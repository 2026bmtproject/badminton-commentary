"""Validate and aggregate v3 human references for the C hold-out review set."""

from __future__ import annotations

import argparse
import json
import runpy
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER = runpy.run_path(str(SCRIPT_DIR / "analyze_forehand_backhand_review.py"))
ExperimentOutput = ANALYZER["ExperimentOutput"]
HumanReview = ANALYZER["HumanReview"]
analyze_review = ANALYZER["analyze_review"]

DEFAULT_RESULT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "ttyvsasy"
    / "forehand_backhand"
    / "holdout_c_review"
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    labeled = [row for row in rows if row["inferred_reference_side"] is not None]
    predicted = [row for row in labeled if row["predicted_side"] is not None]
    correct = [
        row
        for row in labeled
        if row["predicted_side"] == row["inferred_reference_side"]
    ]
    binary_errors = [
        row
        for row in predicted
        if row["predicted_side"] != row["inferred_reference_side"]
    ]
    incorrect_abstentions = [
        row for row in labeled if row["predicted_side"] is None
    ]
    confusion = {
        truth: {prediction: 0 for prediction in ("forehand", "backhand", "unknown")}
        for truth in ("forehand", "backhand")
    }
    for row in labeled:
        truth = str(row["inferred_reference_side"])
        prediction = str(row["predicted_side"] or "unknown")
        confusion[truth][prediction] += 1
    return {
        "counts": {
            "total_events": len(rows),
            "human_labeled": len(labeled),
            "human_uncertain": sum(
                row["verdict"] == "uncertain" for row in rows
            ),
            "human_unreviewed": sum(
                row["verdict"] == "unreviewed" for row in rows
            ),
            "candidate_predicted": len(predicted),
            "candidate_unknown_on_labeled": len(incorrect_abstentions),
            "correct": len(correct),
            "binary_errors": len(binary_errors),
        },
        "metrics": {
            "human_decidable_rate": _ratio(len(labeled), len(rows)),
            "classifier_coverage": _ratio(len(predicted), len(labeled)),
            "fixed_denominator_accuracy": _ratio(len(correct), len(labeled)),
            "selective_accuracy": _ratio(len(correct), len(predicted)),
        },
        "confusion_matrix": confusion,
        "binary_errors": binary_errors,
        "incorrect_abstentions": incorrect_abstentions,
    }


def aggregate_reviews(
    rows_by_segment: dict[int, list[dict[str, object]]],
) -> dict[str, object]:
    new_holdout_segments = sorted(
        segment for segment in rows_by_segment if segment != 144
    )
    recheck_segments = [144] if 144 in rows_by_segment else []

    def collect(segments: list[int]) -> list[dict[str, object]]:
        return [row for segment in segments for row in rows_by_segment[segment]]

    by_player: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_stroke: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in collect(new_holdout_segments):
        by_player[str(row["player"])].append(row)
        by_stroke[str(row["stroke_type"])].append(row)
    return {
        "schema_version": "forehand-backhand-c-holdout-review-metrics-v1",
        "variant": "C_court_prior",
        "groups": {
            "new_holdout": {
                "segments": new_holdout_segments,
                **score_rows(collect(new_holdout_segments)),
            },
            "seg144_recheck": {
                "segments": recheck_segments,
                **score_rows(collect(recheck_segments)),
            },
            "all_reviewed": {
                "segments": sorted(rows_by_segment),
                **score_rows(collect(sorted(rows_by_segment))),
            },
        },
        "by_segment": {
            str(segment): score_rows(rows)
            for segment, rows in sorted(rows_by_segment.items())
        },
        "new_holdout_by_player": {
            player: score_rows(rows)
            for player, rows in sorted(by_player.items())
        },
        "new_holdout_by_stroke_type": {
            stroke: score_rows(rows)
            for stroke, rows in sorted(by_stroke.items())
        },
        "notes": [
            "Primary generalization metrics use only new hold-out segments and exclude SEG144.",
            "Fixed-denominator accuracy counts a candidate unknown as incorrect when the human supplied a side.",
            "Selective accuracy excludes candidate unknowns and uses only human-decidable events.",
            "Human-uncertain events are excluded from accuracy denominators.",
        ],
    }


def _percent(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def render_markdown(summary: dict[str, object]) -> str:
    group_rows = []
    for name, group in summary["groups"].items():
        counts = group["counts"]
        metrics = group["metrics"]
        group_rows.append(
            f"| {name} | {group['segments']} | {counts['human_labeled']} | "
            f"{counts['candidate_predicted']} | "
            f"{_percent(metrics['classifier_coverage'])} | "
            f"{_percent(metrics['fixed_denominator_accuracy'])} | "
            f"{_percent(metrics['selective_accuracy'])} |"
        )
    segment_rows = []
    for segment, score in summary["by_segment"].items():
        counts = score["counts"]
        metrics = score["metrics"]
        segment_rows.append(
            f"| {segment} | {counts['total_events']} | {counts['human_labeled']} | "
            f"{counts['candidate_predicted']} | "
            f"{_percent(metrics['fixed_denominator_accuracy'])} | "
            f"{_percent(metrics['selective_accuracy'])} |"
        )
    holdout = summary["groups"]["new_holdout"]
    binary_errors = "\n".join(
        f"- Event {row['event_index']}: player {row['player']}, "
        f"{row['stroke_type']}, predicted {row['predicted_side']}, "
        f"reference {row['inferred_reference_side']}, "
        f"margin {row['heuristic_margin']:.4f}."
        for row in holdout["binary_errors"]
    ) or "- None."
    abstentions = "\n".join(
        f"- Event {row['event_index']}: player {row['player']}, "
        f"{row['stroke_type']}, candidate unknown, "
        f"reference {row['inferred_reference_side']}."
        for row in holdout["incorrect_abstentions"]
    ) or "- None."
    return f"""# C Court-Prior Forehand/Backhand Hold-out Review

## Primary results

| Group | Segments | Human labeled | Candidate predicted | Coverage | Fixed accuracy | Selective accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(group_rows)}

`new_holdout` is the primary generalization result. SEG144 is reported
separately because its earlier labels participated in selecting C.

## Per segment

| Segment | Events | Human labeled | Candidate predicted | Fixed accuracy | Selective accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(segment_rows)}

## New hold-out binary errors

{binary_errors}

## New hold-out candidate unknowns with a human side

{abstentions}

## Metric boundary

Fixed accuracy uses every human-labeled event and counts candidate `unknown`
as incorrect. Selective accuracy uses only non-null candidates. Human-uncertain
events are excluded from both accuracy denominators.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-files", type=Path, nargs="+", required=True)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows_by_segment = {}
    provenance = []
    for review_path in args.review_files:
        review = HumanReview.model_validate_json(
            review_path.read_text(encoding="utf-8")
        )
        if review.schema_version != "experimental-forehand-backhand-human-reference-v3":
            raise ValueError(f"hold-out review must use v3: {review_path}")
        if review.segment_index is None:
            raise ValueError(f"review has no segment_index: {review_path}")
        segment_index = review.segment_index
        if segment_index in rows_by_segment:
            raise ValueError(f"duplicate review for segment {segment_index}")
        folder = args.result_root / f"seg{segment_index:04d}_C_court_prior"
        result_path = folder / "forehand_backhand_results.json"
        result = ExperimentOutput.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        metrics = analyze_review(result, review)
        rows_by_segment[segment_index] = metrics["review_rows"]
        provenance.append(
            {
                "segment_index": segment_index,
                "review_path": str(review_path),
                "result_path": str(result_path),
                "reviewed_at": review.reviewed_at,
            }
        )
        folder.joinpath("human_reference.json").write_text(
            review.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    summary = aggregate_reviews(rows_by_segment)
    summary["provenance"] = sorted(
        provenance, key=lambda item: int(item["segment_index"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "c_holdout_review_summary.json"
    report_path = args.output / "c_holdout_review_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["groups"], ensure_ascii=False, indent=2))
    print(f"summary: {json_path.resolve()}")
    print(f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()

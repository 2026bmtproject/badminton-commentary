"""Aggregate confidence-gated forehand/backhand human reviews by segment."""

from __future__ import annotations

import argparse
import json
import runpy
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER = runpy.run_path(str(SCRIPT_DIR / "analyze_forehand_backhand_review.py"))
HumanReview = ANALYZER["HumanReview"]
ExperimentOutput = ANALYZER["ExperimentOutput"]
analyze_review = ANALYZER["analyze_review"]
render_markdown = ANALYZER["render_markdown"]
summarize_rows = ANALYZER["_summary"]

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ttyvsasy" / "forehand_backhand"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _format_ratio(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def aggregate_metrics(
    compatible: list[dict[str, object]],
    statuses: list[dict[str, object]],
) -> dict[str, object]:
    rows = [
        row
        for metrics in compatible
        for row in metrics["review_rows"]
    ]
    total = len(rows)
    predicted = [row for row in rows if row["predicted_side"] is not None]
    decided = [
        row
        for row in predicted
        if row["verdict"] in {"correct", "incorrect"}
    ]
    abstained = [row for row in rows if row["predicted_side"] is None]
    correct = sum(row["verdict"] == "correct" for row in decided)
    appropriate = sum(
        row["verdict"] in {"correct", "uncertain"} for row in abstained
    )
    unnecessary = sum(row["verdict"] == "incorrect" for row in abstained)

    confusion = {
        truth: {prediction: 0 for prediction in ("forehand", "backhand")}
        for truth in ("forehand", "backhand")
    }
    by_player: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_stroke: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_orientation: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_player[str(row["player"])].append(row)
        by_stroke[str(row["stroke_type"])].append(row)
        flipped = row["body_flipped_from_court_prior"]
        orientation_key = (
            "flipped_from_court_prior"
            if flipped is True
            else "followed_court_prior"
            if flipped is False
            else "unavailable"
        )
        by_orientation[orientation_key].append(row)
        truth = row["inferred_reference_side"]
        prediction = row["predicted_side"]
        if (
            truth is not None
            and prediction is not None
            and row["verdict"] in {"correct", "incorrect"}
        ):
            confusion[str(truth)][str(prediction)] += 1

    return {
        "schema_version": "forehand-backhand-review-set-metrics-v1",
        "segment_statuses": statuses,
        "compatible_segments": [metrics["segment_index"] for metrics in compatible],
        "counts": {
            "total_hits": total,
            "binary_predictions": len(predicted),
            "unknown_predictions": len(abstained),
            "binary_reviewed": len(decided),
            "binary_correct": correct,
            "binary_incorrect": len(decided) - correct,
            "appropriate_abstentions": appropriate,
            "unnecessary_abstentions": unnecessary,
        },
        "metrics": {
            "classifier_coverage": _ratio(len(predicted), total),
            "selective_accuracy": _ratio(correct, len(decided)),
            "abstention_appropriateness": _ratio(appropriate, len(abstained)),
        },
        "confusion_matrix": confusion,
        "by_segment": {
            str(metrics["segment_index"]): {
                "counts": metrics["counts"],
                "metrics": metrics["metrics"],
            }
            for metrics in compatible
        },
        "by_player": {
            key: summarize_rows(value) for key, value in sorted(by_player.items())
        },
        "by_stroke_type": {
            key: summarize_rows(value) for key, value in sorted(by_stroke.items())
        },
        "by_orientation_decision": {
            key: summarize_rows(value)
            for key, value in sorted(by_orientation.items())
        },
        "binary_errors": [
            row for row in predicted if row["verdict"] == "incorrect"
        ],
        "unnecessary_abstentions": [
            row for row in abstained if row["verdict"] == "incorrect"
        ],
        "notes": [
            "Only reviews matching the exact confidence-gated result are aggregated.",
            "Selective accuracy excludes unknown, uncertain, and unreviewed events.",
            "An incorrect unknown prediction is an unnecessary abstention; its true side is not available in review schema v2.",
            "These segments come from one match and are an experiment, not a production accuracy claim.",
        ],
    }


def render_set_markdown(summary: dict[str, object]) -> str:
    counts = summary["counts"]
    metrics = summary["metrics"]
    status_lines = []
    for status in summary["segment_statuses"]:
        detail = status.get("reason", "matched confidence-gated result")
        status_lines.append(
            f"| {status['segment_index']} | {status['status']} | {detail} |"
        )
    segment_lines = []
    for segment, item in summary["by_segment"].items():
        segment_counts = item["counts"]
        segment_metrics = item["metrics"]
        segment_lines.append(
            f"| {segment} | {segment_counts['total']} | "
            f"{_format_ratio(segment_metrics['classifier_coverage'])} | "
            f"{_format_ratio(segment_metrics['selective_accuracy'])} | "
            f"{segment_counts['appropriate_abstentions']} / "
            f"{segment_counts['unnecessary_abstentions']} |"
        )
    error_lines = "\n".join(
        f"- SEG row event {row['event_index']}: player {row['player']}, "
        f"{row['stroke_type']}, predicted {row['predicted_side']}, "
        f"margin {row['heuristic_margin']:.4f}."
        for row in summary["binary_errors"]
    ) or "- None."
    orientation_lines = []
    for decision, item in summary["by_orientation_decision"].items():
        orientation_lines.append(
            f"| {decision} | {item['total']} | {item['binary_predictions']} | "
            f"{_format_ratio(item['selective_accuracy'])} |"
        )
    return f"""# Forehand/Backhand Confidence-Gate Review Set

## Input compatibility

| Segment | Status | Detail |
| ---: | --- | --- |
{chr(10).join(status_lines)}

## Aggregate

- Compatible segments: {summary['compatible_segments']}
- Total hits: {counts['total_hits']}
- Binary predictions / unknown: {counts['binary_predictions']} / {counts['unknown_predictions']}
- Binary correct / incorrect: {counts['binary_correct']} / {counts['binary_incorrect']}
- Appropriate / unnecessary abstentions: {counts['appropriate_abstentions']} / {counts['unnecessary_abstentions']}
- Classifier coverage: {_format_ratio(metrics['classifier_coverage'])}
- Selective accuracy: {_format_ratio(metrics['selective_accuracy'])}
- Abstention appropriateness: {_format_ratio(metrics['abstention_appropriateness'])}

| Segment | Hits | Coverage | Selective accuracy | Appropriate / unnecessary abstentions |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(segment_lines)}

## Orientation decision diagnostic

| Orientation decision | Hits | Binary predictions | Selective accuracy |
| --- | ---: | ---: | ---: |
{chr(10).join(orientation_lines)}

## Binary errors

{error_lines}

## Interpretation

The confidence gate can turn low-confidence racket-arm observations into
`unknown`, but it does not validate the orientation or forehand/backhand rule.
High-margin binary errors remain part of the measured result. An `incorrect`
verdict on an `unknown` event means the abstention was unnecessary; review v2
does not contain the corrected side, so those events cannot be inserted into
the binary confusion matrix.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=[140, 141, 143, 144],
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    compatible = []
    statuses = []
    for segment_index in args.segments:
        run_dir = args.output_root / f"seg{segment_index:04d}_confidence_gate"
        result_path = run_dir / "forehand_backhand_results.json"
        review_path = (
            args.review_dir
            / f"seg{segment_index:04d}_forehand_backhand_human_review.json"
        )
        status: dict[str, object] = {"segment_index": segment_index}
        try:
            result = ExperimentOutput.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            review = HumanReview.model_validate_json(
                review_path.read_text(encoding="utf-8")
            )
            metrics = analyze_review(result, review)
        except (OSError, ValueError) as exc:
            status.update(status="incompatible", reason=str(exc))
            statuses.append(status)
            continue

        status["status"] = "compatible"
        statuses.append(status)
        compatible.append(metrics)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "human_review.json").write_text(
            review.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "review_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "review_analysis.md").write_text(
            render_markdown(metrics), encoding="utf-8"
        )

    summary = aggregate_metrics(compatible, statuses)
    summary_path = args.output_root / "confidence_gate_review_summary.json"
    report_path = args.output_root / "confidence_gate_review_summary.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_set_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path.resolve()}")
    print(f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()

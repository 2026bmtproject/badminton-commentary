"""Compare four orientation policies against existing human review labels."""

from __future__ import annotations

import argparse
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from badminton_commentary.adapters import CourtPositionToPlayer


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT = runpy.run_path(str(SCRIPT_DIR / "experiment_forehand_backhand.py"))
REVIEW_ANALYZER = runpy.run_path(
    str(SCRIPT_DIR / "analyze_forehand_backhand_review.py")
)
Config = EXPERIMENT["Config"]
ExperimentOutput = EXPERIMENT["ExperimentOutput"]
analyze_segment = EXPERIMENT["analyze_segment"]
HumanReview = REVIEW_ANALYZER["HumanReview"]
analyze_review = REVIEW_ANALYZER["analyze_review"]

DEFAULT_STAGE_ROOT = (
    REPO_ROOT / "experiments" / "ttyvsasy" / "workspace" / "stages"
)
DEFAULT_RESULT_ROOT = REPO_ROOT / "outputs" / "ttyvsasy" / "forehand_backhand"
DEFAULT_OUTPUT = DEFAULT_RESULT_ROOT / "orientation_ablation"

Side = Literal["forehand", "backhand"]


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    config: object


VARIANTS = (
    Variant("A", "original", Config()),
    Variant("B", "face_weight_0", Config(w_face=0.0)),
    Variant(
        "C",
        "court_prior_only",
        Config(orientation_policy="court_prior"),
    ),
    Variant(
        "D",
        "invert_true_branch",
        Config(orientation_policy="invert_disagreement"),
    ),
)


def _reference_result_path(result_root: Path, segment_index: int) -> Path:
    suffix = "seg0144" if segment_index == 144 else (
        f"seg{segment_index:04d}_confidence_gate"
    )
    return result_root / suffix / "forehand_backhand_results.json"


def load_reference_labels(
    *,
    review_dir: Path,
    result_root: Path,
    segments: list[int],
) -> tuple[dict[tuple[int, int], Side], list[dict[str, object]]]:
    references: dict[tuple[int, int], Side] = {}
    provenance = []
    for segment_index in segments:
        result_path = _reference_result_path(result_root, segment_index)
        review_path = (
            review_dir
            / f"seg{segment_index:04d}_forehand_backhand_human_review.json"
        )
        result = ExperimentOutput.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        review = HumanReview.model_validate_json(
            review_path.read_text(encoding="utf-8")
        )
        metrics = analyze_review(result, review)
        segment_labels = 0
        for row in metrics["review_rows"]:
            truth = row["inferred_reference_side"]
            if truth is None:
                continue
            references[(segment_index, int(row["event_index"]))] = truth
            segment_labels += 1
        provenance.append(
            {
                "segment_index": segment_index,
                "labels": segment_labels,
                "review_schema": review.schema_version,
                "review_path": str(review_path),
                "reviewed_result_path": str(result_path),
                "note": (
                    "SEG144 labels are inferred from the reviewed pre-gate baseline; "
                    "the event identity is reused to score current variants."
                    if segment_index == 144
                    else "Labels are inferred from the reviewed confidence-gated A result."
                ),
            }
        )
    return references, provenance


def score_variant(
    *,
    variant: Variant,
    results: dict[int, object],
    references: dict[tuple[int, int], Side],
    original_predictions: dict[tuple[int, int], Side | None] | None,
) -> tuple[dict[str, object], dict[tuple[int, int], Side | None]]:
    predictions = {
        (segment_index, shot.event_index): shot.side
        for segment_index, result in results.items()
        for shot in result.shots
        if (segment_index, shot.event_index) in references
    }
    if set(predictions) != set(references):
        missing = sorted(set(references) - set(predictions))
        extra = sorted(set(predictions) - set(references))
        raise ValueError(
            f"variant {variant.key} event mismatch; missing={missing}, extra={extra}"
        )

    evaluated = len(references)
    predicted = sum(side is not None for side in predictions.values())
    correct = sum(
        predictions[key] == truth for key, truth in references.items()
    )
    confusion = {
        truth: {prediction: 0 for prediction in ("forehand", "backhand", "unknown")}
        for truth in ("forehand", "backhand")
    }
    for key, truth in references.items():
        prediction = predictions[key] or "unknown"
        confusion[truth][prediction] += 1

    by_segment = {}
    for segment_index in sorted(results):
        keys = [key for key in references if key[0] == segment_index]
        segment_predicted = sum(predictions[key] is not None for key in keys)
        segment_correct = sum(
            predictions[key] == references[key] for key in keys
        )
        by_segment[str(segment_index)] = {
            "evaluated": len(keys),
            "predicted": segment_predicted,
            "correct": segment_correct,
            "accuracy_with_unknown_incorrect": (
                segment_correct / len(keys) if keys else None
            ),
            "selective_accuracy": (
                segment_correct / segment_predicted
                if segment_predicted
                else None
            ),
        }

    changed = (
        sum(predictions[key] != original_predictions[key] for key in references)
        if original_predictions is not None
        else 0
    )
    return (
        {
            "variant": variant.key,
            "label": variant.label,
            "config": {
                "face_weight": variant.config.w_face,
                "orientation_policy": variant.config.orientation_policy,
            },
            "evaluated": evaluated,
            "predicted": predicted,
            "unknown": evaluated - predicted,
            "correct": correct,
            "incorrect_or_unknown": evaluated - correct,
            "accuracy_with_unknown_incorrect": (
                correct / evaluated if evaluated else None
            ),
            "selective_accuracy": correct / predicted if predicted else None,
            "changed_from_A": changed,
            "confusion_matrix": confusion,
            "by_segment": by_segment,
        },
        predictions,
    )


def _percentage(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def render_markdown(summary: dict[str, object]) -> str:
    rows = []
    for item in summary["variants"]:
        rows.append(
            f"| {item['variant']} | {item['label']} | {item['correct']} / "
            f"{item['evaluated']} | "
            f"{_percentage(item['accuracy_with_unknown_incorrect'])} | "
            f"{item['predicted']} / {item['evaluated']} | "
            f"{_percentage(item['selective_accuracy'])} | "
            f"{item['changed_from_A']} |"
        )
    segment_rows = []
    for item in summary["variants"]:
        for segment, score in item["by_segment"].items():
            segment_rows.append(
                f"| {item['variant']} | {segment} | {score['correct']} / "
                f"{score['evaluated']} | "
                f"{_percentage(score['accuracy_with_unknown_incorrect'])} | "
                f"{score['predicted']} / {score['evaluated']} |"
            )
    equivalence = summary["equivalence"]
    return f"""# Forehand/Backhand Orientation Ablation

## Fixed human-reference set

- Segments: {summary['segments']}
- Human-labeled binary events: {summary['reference_label_count']}
- Excluded: review rows without an inferable forehand/backhand reference.
- Accuracy counts `unknown` as incorrect so every variant uses the same denominator.
- Selective accuracy scores only non-null predictions and is reported separately.

| Variant | Policy | Correct | Fixed-set accuracy | Coverage | Selective accuracy | Changed from A |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Per-segment fixed-set accuracy

| Variant | Segment | Correct | Accuracy | Coverage |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(segment_rows)}

## C / D equivalence

- Identical predictions: {equivalence['C_and_D_predictions_identical']}
- Reason: with signs restricted to `-1` and `+1`, reversing the voted sign only
  when it disagrees with the court prior always produces the court-prior sign.

## Interpretation boundary

This is a diagnostic ablation on labels that motivated the orientation
hypothesis. It is not a hold-out test and must not be reported as production
accuracy. A winning policy needs validation on newly reviewed rallies or a
different match before it can replace the original rule.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=[140, 141, 143, 144],
    )
    parser.add_argument("--top-player", choices=("a", "b"), default="b")
    parser.add_argument("--bottom-player", choices=("a", "b"), default="a")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    references, provenance = load_reference_labels(
        review_dir=args.review_dir,
        result_root=args.result_root,
        segments=args.segments,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = CourtPositionToPlayer(
        top=args.top_player,
        bottom=args.bottom_player,
    )
    variant_scores = []
    predictions_by_variant = {}
    original_predictions = None
    for variant in VARIANTS:
        variant_dir = args.output / f"{variant.key}_{variant.label}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for segment_index in args.segments:
            result, _ = analyze_segment(
                stage_root=args.stage_root,
                segment_index=segment_index,
                mapping=mapping,
                left_handed_players=set(),
                config=variant.config,
            )
            results[segment_index] = result
            result_path = variant_dir / f"seg{segment_index:04d}_results.json"
            result_path.write_text(
                result.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
        score, predictions = score_variant(
            variant=variant,
            results=results,
            references=references,
            original_predictions=original_predictions,
        )
        if original_predictions is None:
            original_predictions = predictions
        predictions_by_variant[variant.key] = predictions
        variant_scores.append(score)
        print(
            f"{variant.key}: {score['correct']}/{score['evaluated']} "
            f"({_percentage(score['accuracy_with_unknown_incorrect'])})"
        )

    summary = {
        "schema_version": "forehand-backhand-orientation-ablation-v1",
        "segments": args.segments,
        "reference_label_count": len(references),
        "reference_provenance": provenance,
        "variants": variant_scores,
        "equivalence": {
            "C_and_D_predictions_identical": (
                predictions_by_variant["C"] == predictions_by_variant["D"]
            )
        },
        "notes": [
            "Accuracy uses one fixed human-reference denominator and counts unknown as incorrect.",
            "Selective accuracy excludes unknown predictions.",
            "SEG144 references come from the reviewed pre-gate baseline, matched by event identity.",
            "This is a diagnostic development-set ablation, not hold-out validation.",
        ],
    }
    json_path = args.output / "orientation_ablation_summary.json"
    report_path = args.output / "orientation_ablation_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"summary: {json_path.resolve()}")
    print(f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()

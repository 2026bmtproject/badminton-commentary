"""Validate and score a human review of forehand/backhand candidates."""

from __future__ import annotations

import argparse
import json
import runpy
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = runpy.run_path(
    str(Path(__file__).with_name("experiment_forehand_backhand.py"))
)
ExperimentOutput = EXPERIMENT["ExperimentOutput"]
DEFAULT_RESULT = (
    REPO_ROOT
    / "outputs"
    / "ttyvsasy"
    / "forehand_backhand"
    / "seg0144"
    / "forehand_backhand_results.json"
)
DEFAULT_OUTPUT = DEFAULT_RESULT.parent

Side = Literal["forehand", "backhand"]
Verdict = Literal["correct", "incorrect", "uncertain", "unreviewed"]
ReviewStatus = Literal["labeled", "uncertain", "unreviewed"]


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewEntry(ReviewModel):
    event_index: int = Field(ge=0)
    local_frame: int = Field(ge=0)
    player: Literal["a", "b"]
    stroke_type: str
    side: Side | None
    side_zh: str
    margin: float = Field(ge=0)
    verdict: Verdict
    reference_side: Side | None = None
    review_status: ReviewStatus | None = None


class HumanReview(ReviewModel):
    schema_version: Literal[
        "seg144-forehand-backhand-human-review-v1",
        "experimental-forehand-backhand-human-review-v2",
        "experimental-forehand-backhand-human-reference-v3",
    ]
    segment_index: int | None = Field(default=None, ge=0)
    fps: float = Field(gt=0)
    reviewed_at: str
    reviews: list[ReviewEntry]

    @model_validator(mode="after")
    def validate_unique_events(self) -> HumanReview:
        indexes = [item.event_index for item in self.reviews]
        if len(indexes) != len(set(indexes)):
            raise ValueError("human review contains duplicate event_index values")
        if (
            self.schema_version
            in {
                "experimental-forehand-backhand-human-review-v2",
                "experimental-forehand-backhand-human-reference-v3",
            }
            and self.segment_index is None
        ):
            raise ValueError("v2/v3 human review requires segment_index")
        if self.schema_version == "experimental-forehand-backhand-human-reference-v3":
            for item in self.reviews:
                if item.review_status is None:
                    raise ValueError("v3 review entries require review_status")
                if (
                    item.review_status == "labeled"
                    and item.reference_side is None
                ):
                    raise ValueError(
                        "a labeled v3 review entry requires reference_side"
                    )
                if (
                    item.review_status != "labeled"
                    and item.reference_side is not None
                ):
                    raise ValueError(
                        "only labeled v3 review entries may have reference_side"
                    )
                expected_verdict: Verdict = (
                    "correct"
                    if item.review_status == "labeled"
                    and item.side == item.reference_side
                    else "incorrect"
                    if item.review_status == "labeled"
                    else item.review_status
                )
                if item.verdict != expected_verdict:
                    raise ValueError(
                        "v3 verdict is inconsistent with reference_side"
                    )
        return self


def _opposite(side: Side) -> Side:
    return "backhand" if side == "forehand" else "forehand"


def _summary(items: list[dict[str, object]]) -> dict[str, object]:
    binary = [item for item in items if item["predicted_side"] is not None]
    decided = [
        item
        for item in binary
        if item["verdict"] in {"correct", "incorrect"}
    ]
    correct = sum(item["verdict"] == "correct" for item in decided)
    abstained = [item for item in items if item["predicted_side"] is None]
    return {
        "total": len(items),
        "binary_predictions": len(binary),
        "binary_reviewed": len(decided),
        "binary_correct": correct,
        "binary_incorrect": len(decided) - correct,
        "selective_accuracy": correct / len(decided) if decided else None,
        "abstentions": len(abstained),
        "appropriate_abstentions": sum(
            item["verdict"] in {"correct", "uncertain"} for item in abstained
        ),
        "unnecessary_abstentions": sum(
            item["verdict"] == "incorrect" for item in abstained
        ),
    }


def analyze_review(result, review: HumanReview) -> dict[str, object]:
    if review.segment_index is not None and review.segment_index != result.segment_index:
        raise ValueError("review segment_index does not match experiment result")
    result_by_event = {item.event_index: item for item in result.shots}
    review_by_event = {item.event_index: item for item in review.reviews}
    if set(result_by_event) != set(review_by_event):
        missing = sorted(set(result_by_event) - set(review_by_event))
        extra = sorted(set(review_by_event) - set(result_by_event))
        raise ValueError(f"review event mismatch; missing={missing}, extra={extra}")
    if review.fps != result.fps:
        raise ValueError("review fps does not match experiment result")

    rows = []
    for event_index, shot in result_by_event.items():
        item = review_by_event[event_index]
        comparisons = {
            "local_frame": (item.local_frame, shot.local_frame),
            "player": (item.player, shot.player),
            "stroke_type": (item.stroke_type, shot.stroke_type),
            "side": (item.side, shot.side),
        }
        mismatches = [
            name for name, (actual, expected) in comparisons.items() if actual != expected
        ]
        if abs(item.margin - shot.heuristic_margin) > 1e-6:
            mismatches.append("margin")
        if mismatches:
            raise ValueError(
                f"review event {event_index} does not match result fields: {mismatches}"
            )
        inferred_reference: Side | None = item.reference_side
        if inferred_reference is not None:
            pass
        elif item.side is not None and item.verdict == "correct":
            inferred_reference = item.side
        elif item.side is not None and item.verdict == "incorrect":
            inferred_reference = _opposite(item.side)
        rows.append(
            {
                "event_index": event_index,
                "player": item.player,
                "stroke_type": item.stroke_type,
                "predicted_side": item.side,
                "heuristic_margin": item.margin,
                "verdict": item.verdict,
                "inferred_reference_side": inferred_reference,
                "body_flipped_from_court_prior": shot.detail.get(
                    "body_flipped_from_court_prior"
                ),
                "flip_confidence": shot.detail.get("flip_confidence"),
                "accepted_racket_frames": shot.detail.get(
                    "accepted_racket_frames"
                ),
            }
        )

    total = len(rows)
    predicted = [item for item in rows if item["predicted_side"] is not None]
    decided = [
        item
        for item in predicted
        if item["verdict"] in {"correct", "incorrect"}
    ]
    correct = sum(item["verdict"] == "correct" for item in decided)
    abstained = [item for item in rows if item["predicted_side"] is None]
    appropriate_abstentions = [
        item
        for item in abstained
        if item["verdict"] in {"correct", "uncertain"}
    ]
    unnecessary_abstentions = [
        item for item in abstained if item["verdict"] == "incorrect"
    ]
    confusion = {
        truth: {prediction: 0 for prediction in ("forehand", "backhand")}
        for truth in ("forehand", "backhand")
    }
    for item in decided:
        truth = item["inferred_reference_side"]
        prediction = item["predicted_side"]
        if truth is not None and prediction is not None:
            confusion[truth][prediction] += 1

    by_player: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_stroke: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_margin: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in rows:
        by_player[str(item["player"])].append(item)
        by_stroke[str(item["stroke_type"])].append(item)
        margin = float(item["heuristic_margin"])
        bucket = (
            "<0.08"
            if margin < 0.08
            else "0.08-0.40"
            if margin < 0.40
            else "0.40-0.60"
            if margin < 0.60
            else ">=0.60"
        )
        by_margin[bucket].append(item)

    true_forehand = confusion["forehand"]
    true_backhand = confusion["backhand"]
    forehand_precision_denominator = (
        true_forehand["forehand"] + true_backhand["forehand"]
    )
    forehand_recall_denominator = sum(true_forehand.values())
    backhand_precision_denominator = (
        true_forehand["backhand"] + true_backhand["backhand"]
    )
    backhand_recall_denominator = sum(true_backhand.values())
    forehand_precision = (
        true_forehand["forehand"] / forehand_precision_denominator
        if forehand_precision_denominator
        else None
    )
    forehand_recall = (
        true_forehand["forehand"] / forehand_recall_denominator
        if forehand_recall_denominator
        else None
    )
    backhand_precision = (
        true_backhand["backhand"] / backhand_precision_denominator
        if backhand_precision_denominator
        else None
    )
    backhand_recall = (
        true_backhand["backhand"] / backhand_recall_denominator
        if backhand_recall_denominator
        else None
    )

    def f1(precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    forehand_f1 = f1(forehand_precision, forehand_recall)
    backhand_f1 = f1(backhand_precision, backhand_recall)
    return {
        "schema_version": "forehand-backhand-review-metrics-v1",
        "segment_index": result.segment_index,
        "reviewed_at": review.reviewed_at,
        "counts": {
            "total": total,
            "predicted": len(predicted),
            "unknown_prediction": total - len(predicted),
            "review_correct": correct,
            "review_incorrect": len(decided) - correct,
            "review_uncertain": sum(
                item["verdict"] == "uncertain" for item in rows
            ),
            "review_unreviewed": sum(
                item["verdict"] == "unreviewed" for item in rows
            ),
            "binary_reviewed": len(decided),
            "binary_correct": correct,
            "binary_incorrect": len(decided) - correct,
            "appropriate_abstentions": len(appropriate_abstentions),
            "unnecessary_abstentions": len(unnecessary_abstentions),
        },
        "metrics": {
            "classifier_coverage": len(predicted) / total if total else 0,
            "binary_review_coverage": len(decided) / total if total else 0,
            "selective_accuracy": correct / len(decided) if decided else None,
            "overall_correct_fraction": correct / total if total else None,
            "abstention_appropriateness": (
                len(appropriate_abstentions) / len(abstained)
                if abstained
                else None
            ),
            "forehand_precision": forehand_precision,
            "forehand_recall": forehand_recall,
            "forehand_f1": forehand_f1,
            "backhand_precision": backhand_precision,
            "backhand_recall": backhand_recall,
            "backhand_f1": backhand_f1,
            "macro_f1": (
                (forehand_f1 + backhand_f1) / 2
                if forehand_f1 is not None and backhand_f1 is not None
                else None
            ),
        },
        "confusion_matrix": confusion,
        "by_player": {
            key: _summary(value) for key, value in sorted(by_player.items())
        },
        "by_stroke_type": {
            key: _summary(value) for key, value in sorted(by_stroke.items())
        },
        "by_margin": {
            key: _summary(by_margin.get(key, []))
            for key in ("<0.08", "0.08-0.40", "0.40-0.60", ">=0.60")
        },
        "review_rows": rows,
        "errors": [
            item
            for item in predicted
            if item["verdict"] == "incorrect"
        ],
        "unnecessary_abstentions": unnecessary_abstentions,
        "appropriate_abstentions": appropriate_abstentions,
        "uncertain": [
            item for item in rows if item["verdict"] == "uncertain"
        ],
        "notes": [
            "Selective accuracy uses only non-null binary predictions reviewed as correct or incorrect.",
            "A null prediction reviewed as incorrect is an unnecessary abstention, not a binary confusion-matrix error, because review v2 does not record the corrected side.",
            "For a binary non-null prediction, an incorrect verdict infers the opposite side as the reference label.",
            "Metrics describe one rally and do not establish cross-match generalization.",
        ],
    }


def render_markdown(metrics: dict[str, object]) -> str:
    counts = metrics["counts"]
    scores = metrics["metrics"]
    confusion = metrics["confusion_matrix"]
    errors = metrics["errors"]
    uncertain = metrics["uncertain"]
    unnecessary_abstentions = metrics["unnecessary_abstentions"]
    error_lines = "\n".join(
        f"- Event {item['event_index']}: predicted {item['predicted_side']}, "
        f"inferred reference {item['inferred_reference_side']}, "
        f"stroke {item['stroke_type']}, margin {item['heuristic_margin']:.4f}."
        for item in errors
    ) or "- None."
    uncertain_lines = "\n".join(
        f"- Event {item['event_index']}: prediction {item['predicted_side']}, "
        f"stroke {item['stroke_type']}, margin {item['heuristic_margin']:.4f}."
        for item in uncertain
    ) or "- None."
    abstention_lines = "\n".join(
        f"- Event {item['event_index']}: stroke {item['stroke_type']} was unknown, "
        "but the reviewer marked the abstention incorrect."
        for item in unnecessary_abstentions
    ) or "- None."
    macro_f1 = scores["macro_f1"]
    macro_f1_text = f"{macro_f1:.4f}" if macro_f1 is not None else "n/a"
    selective_accuracy = scores["selective_accuracy"]
    selective_accuracy_text = (
        f"{selective_accuracy:.2%}" if selective_accuracy is not None else "n/a"
    )
    return f"""# SEG{metrics['segment_index']} Forehand/Backhand Human Review Metrics

Review time: {metrics['reviewed_at']}

## Summary

- Total hits: {counts['total']}
- Predicted side: {counts['predicted']}
- Unknown prediction: {counts['unknown_prediction']}
- Binary correct / incorrect: {counts['binary_correct']} / {counts['binary_incorrect']}
- Appropriate / unnecessary abstentions: {counts['appropriate_abstentions']} / {counts['unnecessary_abstentions']}
- Classifier coverage: {scores['classifier_coverage']:.2%}
- Selective accuracy: {selective_accuracy_text}
- Macro F1 on decided binary hits: {macro_f1_text}

## Confusion matrix

| Reference \\ Prediction | Forehand | Backhand |
| --- | ---: | ---: |
| Forehand | {confusion['forehand']['forehand']} | {confusion['forehand']['backhand']} |
| Backhand | {confusion['backhand']['forehand']} | {confusion['backhand']['backhand']} |

The reference side for an `incorrect` binary candidate is inferred as the
opposite side because the review schema records verdict rather than an explicit
corrected label.

## Errors

{error_lines}

## Uncertain

{uncertain_lines}

## Unnecessary abstentions

{abstention_lines}

## Interpretation boundary

This is one rally. Selective accuracy excludes null, uncertain and unreviewed hits and must not
be reported as a production accuracy estimate or cross-match validation result.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = ExperimentOutput.model_validate_json(
        args.result.read_text(encoding="utf-8")
    )
    review = HumanReview.model_validate_json(args.review.read_text(encoding="utf-8"))
    metrics = analyze_review(result, review)
    args.output.mkdir(parents=True, exist_ok=True)
    review_copy = args.output / "human_review.json"
    metrics_path = args.output / "review_metrics.json"
    report_path = args.output / "review_analysis.md"
    review_copy.write_text(
        review.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"review: {review_copy.resolve()}")
    print(f"metrics: {metrics_path.resolve()}")
    print(f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()

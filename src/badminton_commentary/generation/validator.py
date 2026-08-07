from __future__ import annotations

import re

from badminton_commentary.schemas import (
    CommentaryPlan,
    GeneratedCommentary,
    RallyAnalysis,
    ScoredRallyFact,
)


class CommentaryValidationError(ValueError):
    """Raised when generated text is not supported by its structured facts."""


def _sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[。！？!?]+", text) if part.strip()]
    return max(len(parts), 1)


def validate_commentary(
    *,
    generated: GeneratedCommentary,
    scored: ScoredRallyFact,
    plan: CommentaryPlan,
    analysis: RallyAnalysis,
) -> None:
    if generated.segment_index != plan.segment_index:
        raise CommentaryValidationError("generated segment_index does not match plan")
    disallowed = set(generated.source_fact_ids) - set(plan.allowed_fact_ids)
    if disallowed:
        raise CommentaryValidationError(
            f"generated commentary cites disallowed fact ids: {sorted(disallowed)}"
        )
    if _sentence_count(generated.text) > plan.max_sentences:
        raise CommentaryValidationError(
            f"generated commentary exceeds {plan.max_sentences} sentences"
        )

    unsupported_outcome_claims = (
        "致勝",
        "拿下這一分",
        "拿下分數",
        "得分",
        "最後一拍",
        "最後以",
    )
    if any(claim in generated.text for claim in unsupported_outcome_claims):
        raise CommentaryValidationError(
            "generated commentary makes an unsupported rally outcome claim"
        )

    expected_score = (scored.fact.score.a, scored.fact.score.b)
    written_scores = [
        (int(a), int(b))
        for a, b in re.findall(r"(\d+)\s*比\s*(\d+)", generated.text)
    ]
    if (
        written_scores
        and None not in expected_score
        and any(score != expected_score for score in written_scores)
    ):
        raise CommentaryValidationError(
            f"generated score does not match rally score {expected_score}"
        )

    cautious_ids = {
        stroke.fact_id
        for stroke in analysis.notable_strokes
        if stroke.confidence_band == "cautious"
    }
    if cautious_ids.intersection(generated.source_fact_ids) and not re.search(
        r"可能|似乎|看來|辨識結果|研判", generated.text
    ):
        raise CommentaryValidationError(
            "cautious stroke commentary requires uncertainty wording"
        )

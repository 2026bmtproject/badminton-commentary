import re

from badminton_commentary.schemas import (
    GeneratedStrokeText,
    RallyScore,
    StrokeEventAnalysis,
    StrokeEventPlan,
)

from .validator import CommentaryValidationError, validate_language_safety


def validate_stroke_commentary(
    *,
    generated: GeneratedStrokeText,
    analysis: StrokeEventAnalysis,
    plan: StrokeEventPlan,
    score: RallyScore,
) -> None:
    disallowed = set(generated.source_fact_ids) - set(plan.allowed_fact_ids)
    if disallowed:
        raise CommentaryValidationError(
            f"stroke commentary cites disallowed fact ids: {sorted(disallowed)}"
        )
    if analysis.current_stroke.fact_id not in generated.source_fact_ids:
        raise CommentaryValidationError(
            "stroke commentary must cite the current stroke fact"
        )

    written_scores = [
        (int(a), int(b))
        for a, b in re.findall(r"(\d+)\s*比\s*(\d+)", generated.text)
    ]
    score_fact_id = f"rally:{analysis.segment_index}:score"
    expected = (score.a, score.b)
    if score_fact_id in generated.source_fact_ids and None in expected:
        raise CommentaryValidationError(
            "stroke commentary cites unavailable score provenance"
        )
    if written_scores:
        if score_fact_id not in generated.source_fact_ids:
            raise CommentaryValidationError(
                "stroke commentary score is missing score provenance"
            )
        if any(item != expected for item in written_scores):
            raise CommentaryValidationError(
                f"stroke commentary score does not match rally score {expected}"
            )

    planned_local = {
        fact.fact_id: fact
        for fact in analysis.local_facts
        if fact.fact_id in plan.allowed_fact_ids
    }
    cited_local = {
        fact_id
        for fact_id in generated.source_fact_ids
        if fact_id in planned_local
    }
    if planned_local and not cited_local:
        raise CommentaryValidationError(
            "stroke commentary omits an available local sequence fact"
        )
    for fact_id in cited_local:
        missing_support = set(planned_local[fact_id].supporting_fact_ids) - set(
            generated.source_fact_ids
        )
        if missing_support:
            raise CommentaryValidationError(
                f"local sequence citation is missing support: {sorted(missing_support)}"
            )

    sentence_endings = sum(generated.text.count(mark) for mark in "。！？!?")
    if sentence_endings > 1:
        raise CommentaryValidationError("stroke commentary must be at most one sentence")

    validate_language_safety(
        text=generated.text,
        allow_exclamation=analysis.speaking_score >= 0.9,
    )

from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.schemas import (
    CommentaryPlan,
    RallyAnalysis,
    ScoredRallyFact,
)


def _allowed_fact_ids(
    scored: ScoredRallyFact,
    analysis: RallyAnalysis,
    *,
    stroke_limit: int,
    pattern_limit: int,
) -> list[str]:
    fact = scored.fact
    prefix = f"rally:{fact.segment_index}"
    allowed: list[str] = []
    if fact.score.a is not None and fact.score.b is not None:
        allowed.append(f"{prefix}:score")
    if fact.events and any(
        reason in scored.importance.reasons for reason in ("medium_rally", "long_rally")
    ):
        allowed.append(f"{prefix}:length")
    selected_patterns = analysis.patterns[:pattern_limit]
    allowed.extend(pattern.fact_id for pattern in selected_patterns)

    candidate_by_id = {
        stroke.fact_id: stroke for stroke in analysis.candidate_strokes
    }
    if selected_patterns:
        representative_ids = {
            pattern.representative_fact_id
            for pattern in selected_patterns
            if pattern.representative_fact_id is not None
        }
        representative_strokes = sorted(
            (
                candidate_by_id[fact_id]
                for fact_id in representative_ids
                if fact_id in candidate_by_id
            ),
            key=lambda stroke: (stroke.salience, stroke.confidence),
            reverse=True,
        )
        allowed.extend(
            stroke.fact_id for stroke in representative_strokes[:stroke_limit]
        )
    else:
        allowed.extend(
            stroke.fact_id for stroke in analysis.notable_strokes[:stroke_limit]
        )
    if fact.highlight_score is not None:
        allowed.append(f"{prefix}:highlight")
    return allowed


def plan_commentary(
    scored: ScoredRallyFact,
    analysis: RallyAnalysis | None = None,
) -> CommentaryPlan:
    """Create a deterministic plan from importance and available facts."""
    importance = scored.importance.score
    if importance >= 0.7:
        should_comment, style, max_sentences = True, "excited", 2
        stroke_limit, pattern_limit = 1, 2
    elif importance >= 0.5:
        should_comment, style, max_sentences = True, "analytical", 2
        stroke_limit, pattern_limit = 1, 1
    elif importance >= 0.25:
        should_comment, style, max_sentences = True, "concise", 1
        stroke_limit, pattern_limit = 1, 1
    else:
        should_comment, style, max_sentences = False, "neutral", 1
        stroke_limit, pattern_limit = 0, 0

    resolved_analysis = analysis or analyze_rally(scored.fact)
    focus = list(scored.importance.reasons)
    focus.extend(pattern.name for pattern in resolved_analysis.patterns[:pattern_limit])
    allowed_fact_ids = _allowed_fact_ids(
        scored,
        resolved_analysis,
        stroke_limit=stroke_limit,
        pattern_limit=pattern_limit,
    )
    if any(
        fact_id.startswith(f"rally:{scored.fact.segment_index}:stroke:")
        for fact_id in allowed_fact_ids
    ):
        focus.append("notable_stroke")

    return CommentaryPlan(
        segment_index=scored.fact.segment_index,
        should_comment=should_comment,
        style=style,
        focus=focus,
        max_sentences=max_sentences,
        allowed_fact_ids=allowed_fact_ids,
    )

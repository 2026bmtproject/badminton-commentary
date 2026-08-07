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
    notable_limit: int,
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
    allowed.extend(
        pattern.fact_id for pattern in analysis.patterns[:pattern_limit]
    )
    allowed.extend(
        stroke.fact_id for stroke in analysis.notable_strokes[:notable_limit]
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
        notable_limit, pattern_limit = 3, 2
    elif importance >= 0.5:
        should_comment, style, max_sentences = True, "analytical", 2
        notable_limit, pattern_limit = 2, 1
    elif importance >= 0.25:
        should_comment, style, max_sentences = True, "concise", 1
        notable_limit, pattern_limit = 1, 1
    else:
        should_comment, style, max_sentences = False, "neutral", 1
        notable_limit, pattern_limit = 0, 0

    resolved_analysis = analysis or analyze_rally(scored.fact)
    focus = list(scored.importance.reasons)
    focus.extend(pattern.name for pattern in resolved_analysis.patterns[:pattern_limit])
    if resolved_analysis.notable_strokes[:notable_limit]:
        focus.append("notable_stroke")

    return CommentaryPlan(
        segment_index=scored.fact.segment_index,
        should_comment=should_comment,
        style=style,
        focus=focus,
        max_sentences=max_sentences,
        allowed_fact_ids=_allowed_fact_ids(
            scored,
            resolved_analysis,
            notable_limit=notable_limit,
            pattern_limit=pattern_limit,
        ),
    )

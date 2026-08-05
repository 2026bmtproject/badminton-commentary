from badminton_commentary.schemas import ImportanceResult, RallyFact


def score_importance(fact: RallyFact) -> ImportanceResult:
    """Score a rally with deterministic, explainable rules."""
    score = 0.0
    reasons: list[str] = []

    if fact.score.a is not None and fact.score.b is not None:
        if abs(fact.score.a - fact.score.b) <= 2:
            score += 0.25
            reasons.append("close_score")
        if max(fact.score.a, fact.score.b) >= 18:
            score += 0.25
            reasons.append("late_game_score")

    if fact.rally_length >= 15:
        score += 0.25
        reasons.append("long_rally")
    elif fact.rally_length >= 8:
        score += 0.15
        reasons.append("medium_rally")

    if fact.highlight_score is not None:
        if fact.highlight_score >= 0.75:
            score += 0.25
            reasons.append("high_highlight_score")
        elif fact.highlight_score >= 0.5:
            score += 0.1
            reasons.append("medium_highlight_score")

    return ImportanceResult(score=min(score, 1.0), reasons=reasons)

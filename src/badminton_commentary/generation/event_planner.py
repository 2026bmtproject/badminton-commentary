from badminton_commentary.schemas import StrokeEventAnalysis, StrokeEventPlan


def plan_stroke_commentary(
    analysis: StrokeEventAnalysis,
    *,
    importance_score: float | None = None,
    force_commentary: bool = False,
) -> StrokeEventPlan:
    """Plan one event; forced mode bypasses legacy importance/salience filters."""
    low_importance_suppressed = (
        importance_score is not None
        and importance_score < 0.25
        and analysis.speaking_score < 0.9
    )
    if not force_commentary and (
        not analysis.should_speak or low_importance_suppressed
    ):
        return StrokeEventPlan(
            segment_index=analysis.segment_index,
            stroke_index=analysis.stroke_index,
            frame=analysis.frame,
            time_sec=analysis.time_sec,
            should_comment=False,
            style="neutral",
            max_sentences=1,
            focus=[],
            allowed_fact_ids=[],
        )

    selected_local = analysis.local_facts[0] if analysis.local_facts else None
    if selected_local is not None:
        allowed = [selected_local.fact_id, *selected_local.supporting_fact_ids]
        focus = [selected_local.name]
    else:
        allowed = [analysis.current_stroke.fact_id]
        focus = [analysis.current_stroke.stroke_type]
    score_fact_id = f"rally:{analysis.segment_index}:score"
    allowed.append(score_fact_id)
    allowed = list(dict.fromkeys(allowed))

    return StrokeEventPlan(
        segment_index=analysis.segment_index,
        stroke_index=analysis.stroke_index,
        frame=analysis.frame,
        time_sec=analysis.time_sec,
        should_comment=True,
        style=(
            "excited"
            if analysis.current_stroke.confidence_band == "reliable"
            and analysis.speaking_score >= 0.9
            else "concise"
        ),
        max_sentences=1,
        focus=focus,
        allowed_fact_ids=allowed,
    )

from __future__ import annotations

from collections.abc import Callable

from badminton_commentary.schemas import (
    AnalyzedStroke,
    RallyFact,
    StrokeEventAnalysis,
    StrokeLocalFact,
)

from .rally_analyzer import (
    ATTACK_STROKES,
    FRONT_STROKES,
    LIFT_STROKES,
    REAR_STROKES,
    analyze_stroke,
)


SPEAKING_THRESHOLD = 0.65
DRIVE_STROKES = {"平快球"}
DROP_STROKES = {"切球", "小球"}


def _local_fact(
    fact: RallyFact,
    *,
    name: str,
    strokes: list[AnalyzedStroke],
    salience: float,
    commentary_hint: str,
) -> StrokeLocalFact:
    return StrokeLocalFact(
        fact_id=(
            f"rally:{fact.segment_index}:local:"
            f"{strokes[0].event_index}-{strokes[-1].event_index}:{name}"
        ),
        name=name,
        start_stroke_index=strokes[0].event_index,
        end_stroke_index=strokes[-1].event_index,
        salience=salience,
        commentary_hint=commentary_hint,
        supporting_fact_ids=[stroke.fact_id for stroke in strokes],
    )


def _adjacent_pair(
    previous: AnalyzedStroke | None,
    current: AnalyzedStroke,
    *,
    previous_types: set[str],
    current_types: set[str],
) -> list[AnalyzedStroke] | None:
    if (
        previous is not None
        and previous.event_index + 1 == current.event_index
        and previous.player != current.player
        and previous.stroke_type in previous_types
        and current.stroke_type in current_types
    ):
        return [previous, current]
    return None


def _append_pair_fact(
    local_facts: list[StrokeLocalFact],
    fact: RallyFact,
    pair: list[AnalyzedStroke] | None,
    *,
    name: str,
    salience: float,
    commentary_hint: str,
) -> None:
    if pair:
        local_facts.append(
            _local_fact(
                fact,
                name=name,
                strokes=pair,
                salience=salience,
                commentary_hint=commentary_hint,
            )
        )


def _three_stroke_sequence(
    previous_two: list[AnalyzedStroke],
    current: AnalyzedStroke,
    predicates: tuple[Callable[[str], bool], Callable[[str], bool], Callable[[str], bool]],
) -> list[AnalyzedStroke] | None:
    if len(previous_two) < 2:
        return None
    first, second = previous_two[-2:]
    if (
        first.event_index + 1 == second.event_index
        and second.event_index + 1 == current.event_index
        and first.player == current.player
        and first.player != second.player
        and predicates[0](first.stroke_type)
        and predicates[1](second.stroke_type)
        and predicates[2](current.stroke_type)
    ):
        return [first, second, current]
    return None


def _analyze_local_facts(
    fact: RallyFact,
    previous: list[AnalyzedStroke],
    current: AnalyzedStroke,
) -> list[StrokeLocalFact]:
    local_facts: list[StrokeLocalFact] = []
    last = previous[-1] if previous else None
    pair_rules = (
        (
            "rear_exchange_continuation",
            LIFT_STROKES,
            LIFT_STROKES,
            0.58,
            "雙方繼續以後場球周旋",
        ),
        (
            "rear_court_stroke_to_front_court_stroke",
            REAR_STROKES,
            FRONT_STROKES,
            0.82,
            "球路由後場球轉入網前處理",
        ),
        (
            "net_exchange_continuation",
            FRONT_STROKES,
            FRONT_STROKES,
            0.80,
            "雙方在網前連續處理",
        ),
        (
            "flat_exchange_continuation",
            DRIVE_STROKES,
            DRIVE_STROKES,
            0.82,
            "雙方以平快球連續對抽",
        ),
        (
            "net_to_lift_transition",
            FRONT_STROKES,
            LIFT_STROKES,
            0.82,
            "網前球後以挑球或高遠球重新帶回後場球路",
        ),
        (
            "lift_to_attack_transition",
            LIFT_STROKES,
            ATTACK_STROKES,
            0.95,
            "挑球或高遠球後緊接進攻球",
        ),
    )
    for name, previous_types, current_types, salience, hint in pair_rules:
        _append_pair_fact(
            local_facts,
            fact,
            _adjacent_pair(
                last,
                current,
                previous_types=previous_types,
                current_types=current_types,
            ),
            name=name,
            salience=salience,
            commentary_hint=hint,
        )

    triple = _three_stroke_sequence(
        previous,
        current,
        (
            lambda stroke: stroke in DROP_STROKES,
            lambda stroke: stroke in LIFT_STROKES,
            lambda stroke: stroke in ATTACK_STROKES,
        ),
    )
    if triple:
        local_facts.append(
            _local_fact(
                fact,
                name="drop_lift_attack_sequence",
                strokes=triple,
                salience=1.0,
                commentary_hint="網前球、挑高與進攻球連續銜接",
            )
        )
    return sorted(local_facts, key=lambda item: item.salience, reverse=True)


def analyze_stroke_events(
    fact: RallyFact,
    *,
    context_size: int = 4,
    include_all_strokes: bool = False,
) -> list[StrokeEventAnalysis]:
    """Analyze strokes in time order; all-strokes mode bypasses speaking filters."""
    if not 2 <= context_size <= 4:
        raise ValueError("context_size must be between 2 and 4")

    usable = [
        stroke
        for event in fact.events
        if (
            stroke := analyze_stroke(
                fact,
                event,
                include_low_confidence=include_all_strokes,
            )
        )
        is not None
    ]
    event_by_index = {event.event_index: event for event in fact.events}
    analyses = []
    for position, current in enumerate(usable):
        previous = usable[max(0, position - context_size) : position]
        local_facts = (
            _analyze_local_facts(fact, previous, current)
            if current.confidence_band == "reliable"
            else []
        )
        speaking_score = max(
            [current.salience, *(item.salience for item in local_facts)]
        )
        event = event_by_index[current.event_index]
        analyses.append(
            StrokeEventAnalysis(
                segment_index=fact.segment_index,
                stroke_index=current.event_index,
                frame=event.frame,
                time_sec=event.time_sec,
                current_stroke=current,
                previous_strokes=previous,
                local_facts=local_facts,
                speaking_score=speaking_score,
                should_speak=(
                    include_all_strokes
                    or (
                        current.confidence_band == "reliable"
                        and speaking_score >= SPEAKING_THRESHOLD
                    )
                ),
            )
        )
    if include_all_strokes:
        return analyses
    for current, following in zip(analyses, analyses[1:]):
        if not current.should_speak or not following.local_facts:
            continue
        following_fact = following.local_facts[0]
        if (
            current.current_stroke.fact_id in following_fact.supporting_fact_ids
            and following_fact.salience > current.speaking_score
        ):
            current.should_speak = False

    last_spoken_index: int | None = None
    last_focus: str | None = None
    for analysis in analyses:
        if not analysis.should_speak:
            continue
        focus = (
            analysis.local_facts[0].name
            if analysis.local_facts
            else analysis.current_stroke.stroke_type
        )
        if analysis.speaking_score < 0.9 and last_spoken_index is not None:
            distance = analysis.stroke_index - last_spoken_index
            if distance < 2 or (focus == last_focus and distance <= 4):
                analysis.should_speak = False
                continue
        last_spoken_index = analysis.stroke_index
        last_focus = focus
    return analyses

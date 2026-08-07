from __future__ import annotations

from collections.abc import Iterable

from badminton_commentary.schemas import (
    AnalyzedStroke,
    RallyAnalysis,
    RallyFact,
    RallyFactEvent,
    StrokePattern,
)


RELIABLE_CONFIDENCE = 0.70
CAUTIOUS_CONFIDENCE = 0.50
NET_STROKES = {"小球", "撲球"}
ATTACK_STROKES = {"殺球", "撲球"}
CLEAR_STROKES = {"高遠球"}


def _analyzed_stroke(fact: RallyFact, event: RallyFactEvent) -> AnalyzedStroke | None:
    if event.stroke_type is None or event.stroke_confidence is None:
        return None
    if event.stroke_confidence < CAUTIOUS_CONFIDENCE:
        return None
    band = (
        "reliable"
        if event.stroke_confidence >= RELIABLE_CONFIDENCE
        else "cautious"
    )
    return AnalyzedStroke(
        fact_id=f"rally:{fact.segment_index}:stroke:{event.event_index}",
        event_index=event.event_index,
        player=event.player,
        stroke_type=event.stroke_type,
        confidence=event.stroke_confidence,
        confidence_band=band,
    )


def _consecutive_pattern(
    strokes: list[AnalyzedStroke],
    *,
    accepted_types: set[str],
) -> list[AnalyzedStroke] | None:
    for first, second in zip(strokes, strokes[1:]):
        if (
            first.stroke_type in accepted_types
            and second.stroke_type in accepted_types
            and first.player != second.player
            and second.event_index == first.event_index + 1
        ):
            return [first, second]
    return None


def _pattern(
    fact: RallyFact,
    *,
    name: str,
    strokes: Iterable[AnalyzedStroke],
) -> StrokePattern:
    support = list(strokes)
    return StrokePattern(
        fact_id=f"rally:{fact.segment_index}:pattern:{name}",
        name=name,
        supporting_fact_ids=[stroke.fact_id for stroke in support],
    )


def analyze_rally(fact: RallyFact) -> RallyAnalysis:
    """Extract traceable stroke observations without inferring a rally winner."""
    usable = [
        stroke
        for event in fact.events
        if (stroke := _analyzed_stroke(fact, event)) is not None
    ]
    reliable = [stroke for stroke in usable if stroke.confidence_band == "reliable"]
    cautious = [stroke for stroke in usable if stroke.confidence_band == "cautious"]
    recognized_count = sum(
        event.stroke_type is not None and event.stroke_confidence is not None
        for event in fact.events
    )
    excluded_count = recognized_count - len(usable)

    patterns: list[StrokePattern] = []
    net = _consecutive_pattern(reliable, accepted_types=NET_STROKES)
    if net:
        patterns.append(_pattern(fact, name="net_exchange", strokes=net))
    attack = _consecutive_pattern(reliable, accepted_types=ATTACK_STROKES)
    if attack:
        patterns.append(_pattern(fact, name="attack_sequence", strokes=attack))
    clear = _consecutive_pattern(reliable, accepted_types=CLEAR_STROKES)
    if clear:
        patterns.append(_pattern(fact, name="clear_exchange", strokes=clear))
    varied_types: dict[str, AnalyzedStroke] = {}
    for stroke in reliable:
        varied_types.setdefault(stroke.stroke_type, stroke)
    if len(varied_types) >= 4:
        patterns.append(
            _pattern(
                fact,
                name="varied_strokes",
                strokes=list(varied_types.values())[:4],
            )
        )

    notable: list[AnalyzedStroke] = []
    for candidate in (
        reliable[0] if reliable else None,
        max(
            (stroke for stroke in reliable if stroke.stroke_type in ATTACK_STROKES),
            key=lambda stroke: stroke.confidence,
            default=None,
        ),
        reliable[-1] if reliable else None,
    ):
        if candidate is not None and candidate.fact_id not in {
            stroke.fact_id for stroke in notable
        }:
            notable.append(candidate)
    if not notable and cautious:
        notable.append(max(cautious, key=lambda stroke: stroke.confidence))

    warnings = []
    if cautious:
        warnings.append("cautious_strokes_present")
    if excluded_count:
        warnings.append("low_confidence_strokes_excluded")
    if not usable:
        warnings.append("no_usable_strokes")

    return RallyAnalysis(
        segment_index=fact.segment_index,
        reliable_stroke_count=len(reliable),
        cautious_stroke_count=len(cautious),
        excluded_stroke_count=excluded_count,
        opening_observed_stroke=usable[0] if usable else None,
        final_observed_stroke=usable[-1] if usable else None,
        notable_strokes=notable,
        patterns=patterns,
        warnings=warnings,
    )

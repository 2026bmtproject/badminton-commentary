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
ATTACK_STROKES = {"殺球", "撲球"}
LIFT_STROKES = {"挑球", "高遠球"}
REAR_STROKES = {"高遠球", "殺球", "切球"}
FRONT_STROKES = {"小球", "撲球"}
STROKE_CATEGORIES = {
    "發球": "serve",
    "高遠球": "rear_court",
    "挑球": "rear_court",
    "殺球": "attack",
    "撲球": "attack",
    "切球": "placement",
    "小球": "front_court",
    "平快球": "drive",
}
STROKE_SALIENCE = {
    "發球": 0.10,
    "殺球": 1.00,
    "撲球": 0.95,
    "平快球": 0.80,
    "切球": 0.75,
    "小球": 0.70,
    "挑球": 0.55,
    "高遠球": 0.50,
}
PATTERN_METADATA = {
    "serve_return_pattern": (0.55, "發接發後連續銜接第三拍"),
    "lift_to_attack_transition": (0.95, "高遠球或挑球後緊接進攻球"),
    "sustained_attack": (1.00, "同一球員在數拍內兩度使用殺球或撲球"),
    "rear_court_stroke_to_front_court_stroke": (
        0.85,
        "球路從後場球轉為網前處理",
    ),
    "stroke_diversity": (0.75, "雙方交替運用至少三類不同球路"),
}


def analyze_stroke(
    fact: RallyFact,
    event: RallyFactEvent,
    *,
    include_low_confidence: bool = False,
) -> AnalyzedStroke | None:
    if (
        event.player is None
        or event.stroke_type is None
        or event.stroke_confidence is None
    ):
        return None
    if event.stroke_confidence < CAUTIOUS_CONFIDENCE and not include_low_confidence:
        return None
    band = (
        "reliable"
        if event.stroke_confidence >= RELIABLE_CONFIDENCE
        else (
            "cautious"
            if event.stroke_confidence >= CAUTIOUS_CONFIDENCE
            else "low"
        )
    )
    return AnalyzedStroke(
        fact_id=f"rally:{fact.segment_index}:stroke:{event.event_index}",
        event_index=event.event_index,
        player=event.player,
        stroke_type=event.stroke_type,
        confidence=event.stroke_confidence,
        confidence_band=band,
        salience=min(
            STROKE_SALIENCE.get(event.stroke_type, 0.25)
            * (0.8 + 0.2 * event.stroke_confidence),
            1.0,
        ),
    )


def _serve_return_pattern(strokes: list[AnalyzedStroke]) -> list[AnalyzedStroke] | None:
    if len(strokes) < 3:
        return None
    first, second, third = strokes[:3]
    if (
        first.stroke_type == "發球"
        and second.event_index == first.event_index + 1
        and third.event_index == second.event_index + 1
        and first.player == third.player
        and first.player != second.player
    ):
        return [first, second, third]
    return None


def _lift_to_attack(strokes: list[AnalyzedStroke]) -> list[AnalyzedStroke] | None:
    for first, second in zip(strokes, strokes[1:]):
        if (
            first.stroke_type in LIFT_STROKES
            and second.stroke_type in ATTACK_STROKES
            and first.player != second.player
            and second.event_index == first.event_index + 1
        ):
            return [first, second]
    return None


def _sustained_attack(strokes: list[AnalyzedStroke]) -> list[AnalyzedStroke] | None:
    for index, first in enumerate(strokes):
        if first.stroke_type not in ATTACK_STROKES:
            continue
        for second in strokes[index + 1 :]:
            if second.event_index - first.event_index > 5:
                break
            if (
                second.player == first.player
                and second.stroke_type in ATTACK_STROKES
            ):
                return [first, second]
    return None


def _rear_stroke_to_front_stroke(
    strokes: list[AnalyzedStroke],
) -> list[AnalyzedStroke] | None:
    for index, first in enumerate(strokes):
        if first.stroke_type not in REAR_STROKES:
            continue
        for second in strokes[index + 1 :]:
            if second.event_index - first.event_index > 6:
                break
            if second.player == first.player and second.stroke_type in FRONT_STROKES:
                return [first, second]
    return None


def _stroke_diversity(strokes: list[AnalyzedStroke]) -> list[AnalyzedStroke] | None:
    for start in range(len(strokes)):
        window = strokes[start : start + 5]
        if len(window) < 3 or window[-1].event_index - window[0].event_index > 8:
            continue
        category_examples: dict[str, AnalyzedStroke] = {}
        for stroke in window:
            category = STROKE_CATEGORIES.get(stroke.stroke_type)
            if category is not None:
                category_examples.setdefault(category, stroke)
        if len(category_examples) >= 3:
            return list(category_examples.values())
    return None


def _pattern(
    fact: RallyFact,
    *,
    name: str,
    strokes: Iterable[AnalyzedStroke],
) -> StrokePattern:
    support = list(strokes)
    salience, commentary_hint = PATTERN_METADATA[name]
    representative = max(
        (stroke for stroke in support if stroke.stroke_type != "發球"),
        key=lambda stroke: stroke.salience,
        default=None,
    )
    return StrokePattern(
        fact_id=f"rally:{fact.segment_index}:pattern:{name}",
        name=name,
        salience=salience,
        commentary_hint=commentary_hint,
        supporting_fact_ids=[stroke.fact_id for stroke in support],
        representative_fact_id=(
            representative.fact_id if representative is not None else None
        ),
    )


def analyze_rally(fact: RallyFact) -> RallyAnalysis:
    """Extract traceable stroke observations without inferring a rally winner."""
    usable = [
        stroke
        for event in fact.events
        if (stroke := analyze_stroke(fact, event)) is not None
    ]
    reliable = [stroke for stroke in usable if stroke.confidence_band == "reliable"]
    cautious = [stroke for stroke in usable if stroke.confidence_band == "cautious"]
    recognized_count = sum(
        event.stroke_type is not None and event.stroke_confidence is not None
        for event in fact.events
    )
    excluded_count = recognized_count - len(usable)

    patterns: list[StrokePattern] = []
    pattern_candidates = (
        ("serve_return_pattern", _serve_return_pattern(reliable)),
        ("lift_to_attack_transition", _lift_to_attack(reliable)),
        ("sustained_attack", _sustained_attack(reliable)),
        (
            "rear_court_stroke_to_front_court_stroke",
            _rear_stroke_to_front_stroke(reliable),
        ),
        ("stroke_diversity", _stroke_diversity(reliable)),
    )
    for name, supporting_strokes in pattern_candidates:
        if supporting_strokes:
            patterns.append(
                _pattern(fact, name=name, strokes=supporting_strokes)
            )

    patterns.sort(key=lambda pattern: pattern.salience, reverse=True)
    notable_pool = reliable or cautious
    notable = sorted(
        (stroke for stroke in notable_pool if stroke.stroke_type != "發球"),
        key=lambda stroke: (stroke.salience, stroke.confidence),
        reverse=True,
    )[:5]

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
        candidate_strokes=usable,
        notable_strokes=notable,
        patterns=patterns,
        warnings=warnings,
    )

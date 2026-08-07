import pytest

from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.schemas import (
    ImportanceResult,
    RallyFact,
    RallyFactEvent,
    RallyScore,
    ScoredRallyFact,
)


RALLIES = [
    [("a", "小球"), ("a", "殺球"), ("b", "小球")],
    [("b", "小球"), ("b", "殺球"), ("a", "小球")],
    [
        ("b", "發球"),
        ("b", "高遠球"),
        ("a", "切球"),
        ("b", "平快球"),
        ("b", "平快球"),
        ("b", "小球"),
        ("a", "小球"),
        ("b", "小球"),
        ("a", "平快球"),
    ],
    [
        ("b", "殺球"),
        ("a", "小球"),
        ("b", "平快球"),
        ("b", "小球"),
        ("b", "切球"),
        ("a", "小球"),
        ("b", "小球"),
        ("a", "小球"),
        ("b", "高遠球"),
        ("a", "高遠球"),
    ],
    [("b", "小球"), ("a", "小球"), ("b", "平快球"), ("a", "殺球")],
]


def scored_rally(segment_index, sequence):
    events = [
        RallyFactEvent(
            event_index=index,
            frame=index * 10,
            time_sec=index / 3,
            player=player,
            stroke_type=stroke_type,
            stroke_confidence=0.8,
        )
        for index, (player, stroke_type) in enumerate(sequence)
    ]
    return ScoredRallyFact(
        fact=RallyFact(
            segment_index=segment_index,
            game_index=None,
            start_sec=0,
            end_sec=2,
            duration_sec=2,
            score=RallyScore(a=20 + segment_index // 2, b=20 + segment_index // 2),
            server=None,
            events=events,
            rally_length=len(events),
            highlight_score=None,
        ),
        importance=ImportanceResult(score=0.65, reasons=["late_game_score"]),
    )


@pytest.mark.parametrize(("segment_index", "sequence"), list(enumerate(RALLIES)))
def test_five_rally_patterns_are_semantically_bounded(segment_index, sequence):
    scored = scored_rally(segment_index, sequence)
    analysis = analyze_rally(scored.fact)
    plan = plan_commentary(scored, analysis)

    names = {pattern.name for pattern in analysis.patterns}
    assert "tempo_variation" not in names
    assert "rear_to_front_transition" not in names
    assert all("觀測" not in pattern.commentary_hint for pattern in analysis.patterns)
    assert all("類型" not in pattern.commentary_hint for pattern in analysis.patterns)

    pattern_positions = [
        index for index, fact_id in enumerate(plan.allowed_fact_ids) if ":pattern:" in fact_id
    ]
    stroke_positions = [
        index for index, fact_id in enumerate(plan.allowed_fact_ids) if ":stroke:" in fact_id
    ]
    if pattern_positions and stroke_positions:
        assert max(pattern_positions) < min(stroke_positions)
    assert len(stroke_positions) <= 1

    selected_stroke_ids = {
        fact_id for fact_id in plan.allowed_fact_ids if ":stroke:" in fact_id
    }
    serve_ids = {
        f"rally:{segment_index}:stroke:{index}"
        for index, (_, stroke_type) in enumerate(sequence)
        if stroke_type == "發球"
    }
    assert selected_stroke_ids.isdisjoint(serve_ids)

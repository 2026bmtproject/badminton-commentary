from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.schemas import (
    ImportanceResult,
    RallyFact,
    RallyFactEvent,
    RallyScore,
    ScoredRallyFact,
)


def make_scored_fact(importance: float) -> ScoredRallyFact:
    fact = RallyFact(
        segment_index=3,
        game_index=None,
        start_sec=1,
        end_sec=2,
        duration_sec=1,
        score=RallyScore(a=20, b=20),
        server=None,
        events=[
            RallyFactEvent(
                event_index=7,
                frame=45,
                time_sec=1.5,
                player="a",
                stroke_type="殺球",
                stroke_confidence=0.9,
            )
        ],
        rally_length=1,
        highlight_score=None,
    )
    return ScoredRallyFact(
        fact=fact,
        importance=ImportanceResult(score=importance, reasons=["close_score"]),
    )


def test_high_importance_uses_excited_two_sentence_plan():
    plan = plan_commentary(make_scored_fact(0.75))

    assert plan.should_comment is True
    assert plan.style == "excited"
    assert plan.max_sentences == 2
    assert plan.focus == ["close_score", "notable_stroke"]


def test_low_importance_is_skipped():
    plan = plan_commentary(make_scored_fact(0.1))

    assert plan.should_comment is False
    assert plan.style == "neutral"


def test_plan_only_allows_fact_ids_that_exist():
    plan = plan_commentary(make_scored_fact(0.5))

    assert plan.allowed_fact_ids == [
        "rally:3:score",
        "rally:3:stroke:7",
    ]


def test_ordinary_serve_loses_to_more_salient_stroke():
    scored = make_scored_fact(0.5)
    scored.fact.events = [
        RallyFactEvent(
            event_index=0,
            frame=10,
            time_sec=0.3,
            player="a",
            stroke_type="發球",
            stroke_confidence=0.95,
        ),
        RallyFactEvent(
            event_index=1,
            frame=20,
            time_sec=0.6,
            player="b",
            stroke_type="殺球",
            stroke_confidence=0.8,
        ),
    ]
    scored.fact.rally_length = 2

    plan = plan_commentary(scored)

    assert "rally:3:stroke:0" not in plan.allowed_fact_ids
    assert "rally:3:stroke:1" in plan.allowed_fact_ids


def test_pattern_precedes_one_representative_stroke():
    scored = make_scored_fact(0.5)
    scored.fact.events = [
        RallyFactEvent(
            event_index=0,
            frame=10,
            time_sec=0.3,
            player="a",
            stroke_type="高遠球",
            stroke_confidence=0.9,
        ),
        RallyFactEvent(
            event_index=1,
            frame=20,
            time_sec=0.6,
            player="b",
            stroke_type="殺球",
            stroke_confidence=0.9,
        ),
        RallyFactEvent(
            event_index=2,
            frame=30,
            time_sec=1.0,
            player="a",
            stroke_type="平快球",
            stroke_confidence=0.9,
        ),
    ]
    scored.fact.rally_length = 3

    plan = plan_commentary(scored)

    pattern_id = "rally:3:pattern:lift_to_attack_transition"
    assert plan.allowed_fact_ids.index(pattern_id) < plan.allowed_fact_ids.index(
        "rally:3:stroke:1"
    )
    assert len([item for item in plan.allowed_fact_ids if ":stroke:" in item]) == 1

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

from badminton_commentary.analysis.stroke_event_analyzer import analyze_stroke_events
from badminton_commentary.generation.event_planner import plan_stroke_commentary
from badminton_commentary.schemas import RallyFact, RallyFactEvent, RallyScore


def fact():
    events = [
        RallyFactEvent(
            event_index=index,
            frame=index * 30,
            time_sec=float(index),
            player=player,
            stroke_type=stroke_type,
            stroke_confidence=0.9,
        )
        for index, player, stroke_type in (
            (0, "a", "小球"),
            (1, "b", "挑球"),
            (2, "a", "殺球"),
        )
    ]
    return RallyFact(
        segment_index=1,
        game_index=None,
        start_sec=0,
        end_sec=3,
        duration_sec=3,
        score=RallyScore(a=5, b=5),
        server=None,
        events=events,
        rally_length=3,
        highlight_score=None,
    )


def test_event_plan_prefers_local_sequence_and_keeps_support():
    analysis = analyze_stroke_events(fact())[-1]

    plan = plan_stroke_commentary(analysis)

    assert plan.should_comment is True
    assert plan.allowed_fact_ids[:4] == [
        "rally:1:local:0-2:drop_lift_attack_sequence",
        "rally:1:stroke:0",
        "rally:1:stroke:1",
        "rally:1:stroke:2",
    ]
    assert plan.stroke_index == 2
    assert plan.frame == 60
    assert plan.time_sec == 2


def test_skipped_event_plan_has_no_allowed_facts():
    low_information = fact()
    low_information.events = low_information.events[:1]
    low_information.events[0].stroke_type = "發球"
    analysis = analyze_stroke_events(low_information)[0]

    plan = plan_stroke_commentary(analysis)

    assert plan.should_comment is False
    assert plan.allowed_fact_ids == []


def test_low_importance_reduces_ordinary_event_density():
    ordinary = fact()
    ordinary.events = ordinary.events[:1]
    analysis = analyze_stroke_events(ordinary)[0]

    default_plan = plan_stroke_commentary(analysis)
    low_importance_plan = plan_stroke_commentary(
        analysis,
        importance_score=0.1,
    )

    assert default_plan.should_comment is True
    assert low_importance_plan.should_comment is False

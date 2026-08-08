import pytest

from badminton_commentary.analysis.stroke_event_analyzer import analyze_stroke_events
from badminton_commentary.schemas import RallyFact, RallyFactEvent, RallyScore


def make_fact(strokes):
    events = [
        RallyFactEvent(
            event_index=index,
            frame=index * 30,
            time_sec=float(index),
            player=player,
            stroke_type=stroke_type,
            stroke_confidence=confidence,
        )
        for index, (player, stroke_type, confidence) in enumerate(strokes)
    ]
    return RallyFact(
        segment_index=2,
        game_index=None,
        start_sec=0,
        end_sec=10,
        duration_sec=10,
        score=RallyScore(a=10, b=10),
        server=None,
        events=events,
        rally_length=len(events),
        highlight_score=None,
    )


def test_event_analyses_keep_time_order_and_previous_context():
    analyses = analyze_stroke_events(
        make_fact(
            [
                ("a", "發球", 0.9),
                ("b", "小球", 0.9),
                ("a", "挑球", 0.9),
                ("b", "殺球", 0.9),
            ]
        )
    )

    assert [item.stroke_index for item in analyses] == [0, 1, 2, 3]
    assert [item.time_sec for item in analyses] == [0, 1, 2, 3]
    assert [stroke.event_index for stroke in analyses[3].previous_strokes] == [0, 1, 2]


def test_low_information_serve_and_repeated_clear_are_not_forced_to_speak():
    analyses = analyze_stroke_events(
        make_fact(
            [
                ("a", "發球", 0.95),
                ("b", "高遠球", 0.9),
                ("a", "高遠球", 0.9),
            ]
        )
    )

    assert analyses[0].should_speak is False
    assert analyses[2].local_facts[0].name == "rear_exchange_continuation"
    assert analyses[2].should_speak is False


def test_high_salience_smash_is_more_likely_to_speak():
    analyses = analyze_stroke_events(
        make_fact([("a", "高遠球", 0.9), ("b", "殺球", 0.9)])
    )

    assert analyses[0].should_speak is False
    assert analyses[1].should_speak is True
    assert analyses[1].speaking_score > analyses[0].speaking_score


def test_three_stroke_fact_only_combines_adjacent_events():
    analyses = analyze_stroke_events(
        make_fact(
            [
                ("a", "小球", 0.9),
                ("b", "挑球", 0.9),
                ("a", "殺球", 0.9),
            ]
        )
    )

    local = analyses[2].local_facts[0]
    assert local.name == "drop_lift_attack_sequence"
    assert local.supporting_fact_ids == [
        "rally:2:stroke:0",
        "rally:2:stroke:1",
        "rally:2:stroke:2",
    ]
    assert [item.should_speak for item in analyses] == [False, False, True]


def test_speaking_policy_avoids_adjacent_repeated_low_priority_lines():
    analyses = analyze_stroke_events(
        make_fact(
            [
                ("a", "小球", 0.9),
                ("b", "小球", 0.9),
                ("a", "小球", 0.9),
                ("b", "平快球", 0.9),
            ]
        )
    )

    spoken = [item.stroke_index for item in analyses if item.should_speak]
    assert spoken == [1, 3]


def test_context_size_outside_two_to_four_is_rejected():
    with pytest.raises(ValueError, match="between 2 and 4"):
        analyze_stroke_events(make_fact([]), context_size=5)


def test_non_adjacent_strokes_do_not_form_a_local_sequence():
    analyses = analyze_stroke_events(
        make_fact(
            [
                ("a", "小球", 0.9),
                ("b", "未知球種", 0.2),
                ("a", "挑球", 0.9),
                ("b", "殺球", 0.9),
            ]
        )
    )

    lift = next(item for item in analyses if item.stroke_index == 2)
    smash = next(item for item in analyses if item.stroke_index == 3)
    assert lift.local_facts == []
    assert all(
        fact.name != "drop_lift_attack_sequence" for fact in smash.local_facts
    )

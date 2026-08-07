from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.schemas import RallyFact, RallyFactEvent, RallyScore


def make_fact(events):
    return RallyFact(
        segment_index=4,
        game_index=None,
        start_sec=0,
        end_sec=2,
        duration_sec=2,
        score=RallyScore(a=10, b=10),
        server=None,
        events=events,
        rally_length=len(events),
        highlight_score=None,
    )


def event(index, player, stroke_type, confidence):
    return RallyFactEvent(
        event_index=index,
        frame=index * 10,
        time_sec=index / 3,
        player=player,
        stroke_type=stroke_type,
        stroke_confidence=confidence,
    )


def test_confidence_bands_and_low_confidence_exclusion():
    analysis = analyze_rally(
        make_fact(
            [
                event(0, "a", "發球", 0.9),
                event(1, "b", "小球", 0.6),
                event(2, "a", "殺球", 0.49),
            ]
        )
    )

    assert analysis.reliable_stroke_count == 1
    assert analysis.cautious_stroke_count == 1
    assert analysis.excluded_stroke_count == 1
    assert analysis.opening_observed_stroke.event_index == 0
    assert analysis.final_observed_stroke.event_index == 1
    assert analysis.warnings == [
        "cautious_strokes_present",
        "low_confidence_strokes_excluded",
    ]


def test_patterns_keep_supporting_fact_ids():
    analysis = analyze_rally(
        make_fact(
            [
                event(0, "a", "小球", 0.9),
                event(1, "b", "撲球", 0.8),
                event(2, "a", "殺球", 0.85),
                event(3, "b", "高遠球", 0.9),
            ]
        )
    )

    assert [pattern.name for pattern in analysis.patterns] == [
        "net_exchange",
        "attack_sequence",
        "varied_strokes",
    ]
    assert analysis.patterns[0].supporting_fact_ids == [
        "rally:4:stroke:0",
        "rally:4:stroke:1",
    ]


def test_final_observed_stroke_does_not_claim_a_winner():
    analysis = analyze_rally(make_fact([event(0, "a", "殺球", 0.8)]))

    assert analysis.final_observed_stroke.stroke_type == "殺球"
    assert "winner" not in analysis.model_dump()

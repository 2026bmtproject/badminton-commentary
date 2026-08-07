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


def test_tactical_patterns_keep_supporting_fact_ids():
    analysis = analyze_rally(
        make_fact(
            [
                event(0, "a", "發球", 0.9),
                event(1, "b", "高遠球", 0.8),
                event(2, "a", "殺球", 0.85),
                event(3, "b", "小球", 0.9),
                event(4, "a", "撲球", 0.88),
            ]
        )
    )

    assert [pattern.name for pattern in analysis.patterns] == [
        "sustained_attack",
        "lift_to_attack_transition",
        "rear_court_stroke_to_front_court_stroke",
        "stroke_diversity",
        "serve_return_pattern",
    ]
    serve_pattern = next(
        pattern
        for pattern in analysis.patterns
        if pattern.name == "serve_return_pattern"
    )
    assert serve_pattern.supporting_fact_ids == [
        "rally:4:stroke:0",
        "rally:4:stroke:1",
        "rally:4:stroke:2",
    ]
    assert serve_pattern.salience < analysis.patterns[0].salience
    assert analysis.notable_strokes[0].stroke_type == "殺球"
    assert all(stroke.stroke_type != "發球" for stroke in analysis.notable_strokes)


def test_final_observed_stroke_does_not_claim_a_winner():
    analysis = analyze_rally(make_fact([event(0, "a", "殺球", 0.8)]))

    assert analysis.final_observed_stroke.stroke_type == "殺球"
    assert "winner" not in analysis.model_dump()

from badminton_commentary.analysis.fact_builder import build_rally_facts
from badminton_commentary.analysis.importance import score_importance
from badminton_commentary.schemas import (
    EventsInput,
    HighlightsInput,
    RallyFact,
    RallyScore,
    ScoresInput,
    SegmentsInput,
    StrokesInput,
)


def make_fact(
    *,
    score_a=0,
    score_b=0,
    rally_length=1,
    highlight_score=None,
):
    return RallyFact(
        segment_index=0,
        game_index=None,
        start_sec=0,
        end_sec=1,
        duration_sec=1,
        score=RallyScore(a=score_a, b=score_b),
        server=None,
        events=[],
        rally_length=rally_length,
        highlight_score=highlight_score,
    )


def test_sample_rally_is_close_and_long():
    facts = build_rally_facts(
        segments=SegmentsInput.model_validate(
            {
                "fps": 25,
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 526,
                        "start_sec": 0,
                        "end_sec": 21.08,
                        "duration_sec": 21.08,
                    }
                ],
            }
        ),
        scores=ScoresInput.model_validate(
            {
                "rallies": [
                    {
                        "segment_index": 0,
                        "score_a": 8,
                        "score_b": 6,
                        "server": None,
                        "game_index": None,
                    }
                ]
            }
        ),
        events=EventsInput.model_validate(
            {
                "events": [
                    {"frame": frame, "player": "a", "segment_index": 0}
                    for frame in range(18)
                ]
            }
        ),
        strokes=StrokesInput(strokes=[]),
        highlights=HighlightsInput(highlights=[]),
    )

    importance = score_importance(facts[0])

    assert importance.score == 0.5
    assert importance.reasons == ["close_score", "long_rally"]


def test_medium_rally_uses_only_medium_rule():
    importance = score_importance(
        make_fact(score_a=10, score_b=5, rally_length=8)
    )

    assert importance.score == 0.15
    assert importance.reasons == ["medium_rally"]


def test_late_close_highlighted_rally_reaches_one():
    importance = score_importance(
        make_fact(score_a=20, score_b=20, rally_length=15, highlight_score=0.8)
    )

    assert importance.score == 1.0
    assert importance.reasons == [
        "close_score",
        "late_game_score",
        "long_rally",
        "high_highlight_score",
    ]


def test_medium_highlight_rule():
    importance = score_importance(
        make_fact(score_a=10, score_b=5, highlight_score=0.5)
    )

    assert importance.score == 0.1
    assert importance.reasons == ["medium_highlight_score"]


def test_missing_scores_and_highlight_are_allowed():
    importance = score_importance(
        make_fact(score_a=None, score_b=None, highlight_score=None)
    )

    assert importance.score == 0.0
    assert importance.reasons == []

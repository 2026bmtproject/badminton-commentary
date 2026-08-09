import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from badminton_commentary.schemas import (
    EventsInput,
    HighlightsInput,
    RallyFactsOutput,
    ScoresInput,
    SegmentsInput,
    StrokesInput,
)


FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "sample_match"


def load_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("segments.json", SegmentsInput),
        ("scores.json", ScoresInput),
        ("events.json", EventsInput),
        ("strokes.json", StrokesInput),
        ("highlights.json", HighlightsInput),
    ],
)
def test_sample_match_inputs_are_valid(filename, model):
    model.model_validate_json(load_fixture(filename))


def test_nullable_score_metadata_is_allowed():
    scores = ScoresInput.model_validate_json(load_fixture("scores.json"))

    assert scores.rallies[0].server is None
    assert scores.rallies[0].game_index is None


def test_invalid_player_is_rejected():
    payload = {"events": [{"frame": 31, "player": "top", "segment_index": 0}]}

    with pytest.raises(ValidationError, match="player"):
        EventsInput.model_validate(payload)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_out_of_range_confidence_is_rejected(confidence):
    payload = {
        "strokes": [
            {"event_index": 0, "stroke_type": "發短球", "confidence": confidence}
        ]
    }

    with pytest.raises(ValidationError, match="confidence"):
        StrokesInput.model_validate(payload)


def test_negative_event_index_is_rejected():
    payload = {
        "strokes": [
            {"event_index": -1, "stroke_type": "發短球", "confidence": 0.8}
        ]
    }

    with pytest.raises(ValidationError, match="event_index"):
        StrokesInput.model_validate(payload)


def test_invalid_segment_range_is_rejected():
    payload = json.loads(load_fixture("segments.json"))
    payload["segments"][0]["end_frame"] = -1

    with pytest.raises(ValidationError, match="end_frame"):
        SegmentsInput.model_validate(payload)


def test_segment_accepts_millisecond_quantization_difference():
    segments = SegmentsInput.model_validate(
        {
            "fps": 30,
            "segments": [
                {
                    "start_frame": 22448,
                    "end_frame": 22492,
                    "start_sec": 748.267,
                    "end_sec": 749.733,
                    "duration_sec": 1.467,
                }
            ],
        }
    )

    assert segments.segments[0].duration_sec == 1.467


def test_segment_rejects_duration_beyond_millisecond_quantization():
    with pytest.raises(ValidationError, match="within 0.001 seconds"):
        SegmentsInput.model_validate(
            {
                "fps": 30,
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 30,
                        "start_sec": 0,
                        "end_sec": 1,
                        "duration_sec": 1.01,
                    }
                ],
            }
        )


def test_unknown_fields_are_rejected():
    payload = {"highlights": [], "unexpected": True}

    with pytest.raises(ValidationError, match="unexpected"):
        HighlightsInput.model_validate(payload)


def test_scored_rally_fact_output_requires_importance():
    payload = {
        "rallies": [
            {
                "fact": {
                    "segment_index": 0,
                    "game_index": None,
                    "start_sec": 0,
                    "end_sec": 1,
                    "duration_sec": 1,
                    "score": {"a": 0, "b": 0},
                    "server": None,
                    "events": [],
                    "rally_length": 0,
                    "highlight_score": None,
                }
            }
        ]
    }

    with pytest.raises(ValidationError, match="importance"):
        RallyFactsOutput.model_validate(payload)

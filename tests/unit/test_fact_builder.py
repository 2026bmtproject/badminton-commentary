import json
from pathlib import Path

import pytest

from badminton_commentary.analysis.fact_builder import build_rally_facts
from badminton_commentary.schemas import (
    EventsInput,
    HighlightsInput,
    ScoresInput,
    SegmentsInput,
    StrokesInput,
)


FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "sample_match"


def load_model(filename, model):
    payload = (FIXTURE_DIR / filename).read_text(encoding="utf-8")
    return model.model_validate_json(payload)


def sample_inputs():
    return {
        "segments": load_model("segments.json", SegmentsInput),
        "scores": load_model("scores.json", ScoresInput),
        "events": load_model("events.json", EventsInput),
        "strokes": load_model("strokes.json", StrokesInput),
        "highlights": load_model("highlights.json", HighlightsInput),
    }


def test_builds_single_rally_from_sample_match():
    facts = build_rally_facts(**sample_inputs())

    assert len(facts) == 1
    fact = facts[0]
    assert fact.segment_index == 0
    assert fact.score.a == 8
    assert fact.score.b == 6
    assert fact.server is None
    assert fact.game_index is None
    assert fact.rally_length == 18
    assert fact.highlight_score is None


def test_joins_strokes_to_events_through_event_index():
    fact = build_rally_facts(**sample_inputs())[0]

    first = fact.events[0]
    assert first.event_index == 0
    assert first.frame == 31
    assert first.time_sec == pytest.approx(1.24)
    assert first.player == "a"
    assert first.stroke_type == "發短球"
    assert first.stroke_confidence == pytest.approx(0.6067)

    last = fact.events[-1]
    assert last.event_index == 17
    assert last.frame == 491
    assert last.time_sec == pytest.approx(19.64)
    assert last.player == "b"
    assert last.stroke_type == "殺球"


def test_missing_stroke_degrades_to_unknown_stroke_data():
    inputs = sample_inputs()
    payload = json.loads((FIXTURE_DIR / "strokes.json").read_text(encoding="utf-8"))
    payload["strokes"] = payload["strokes"][1:]
    inputs["strokes"] = StrokesInput.model_validate(payload)

    first = build_rally_facts(**inputs)[0].events[0]

    assert first.stroke_type is None
    assert first.stroke_confidence is None


def test_out_of_range_stroke_event_index_is_rejected():
    inputs = sample_inputs()
    inputs["strokes"] = StrokesInput.model_validate(
        {
            "strokes": [
                {"event_index": 18, "stroke_type": "殺球", "confidence": 0.8}
            ]
        }
    )

    with pytest.raises(ValueError, match="event_index 18 is out of range"):
        build_rally_facts(**inputs)


def test_duplicate_stroke_event_index_is_rejected():
    inputs = sample_inputs()
    inputs["strokes"] = StrokesInput.model_validate(
        {
            "strokes": [
                {"event_index": 0, "stroke_type": "發短球", "confidence": 0.8},
                {"event_index": 0, "stroke_type": "長球", "confidence": 0.7},
            ]
        }
    )

    with pytest.raises(ValueError, match="duplicate stroke event_index 0"):
        build_rally_facts(**inputs)


def test_event_outside_segment_range_is_rejected():
    inputs = sample_inputs()
    inputs["events"] = EventsInput.model_validate(
        {"events": [{"frame": 527, "player": "a", "segment_index": 0}]}
    )
    inputs["strokes"] = StrokesInput(strokes=[])

    with pytest.raises(ValueError, match="outside segment 0 frame range"):
        build_rally_facts(**inputs)

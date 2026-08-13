import runpy
from pathlib import Path

import pytest
from pydantic import ValidationError


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = runpy.run_path(
    str(
        REPO_ROOT
        / "experiments"
        / "ttyvsasy"
        / "scripts"
        / "run_direct_rallyfact_v3.py"
    )
)


def _source_event(event_index, frame, player, depth_zone):
    return {
        "event_index": event_index,
        "frame": frame,
        "time_sec": frame / 30,
        "player": player,
        "stroke": {
            "stroke_type": "smash",
            "confidence": 0.9,
            "source_frame": frame,
        },
        "pose_features": {
            "source_start_frame": frame - 1,
            "source_end_frame": frame + 1,
            "hit_frame": frame,
        },
        "court_position": {
            "source_frame": frame,
            "projection_confidence": 0.8,
            "depth_zone": depth_zone,
            "position_change_from_previous_same_player_hit": "unknown",
            "limitations": [],
        },
        "shuttle_window": {
            "start_frame": frame - 1,
            "end_frame": frame + 1,
            "points": [
                {"frame": frame - 1},
                {"frame": frame},
                {"frame": frame + 1},
            ],
        },
    }


def _generated_event(source):
    return {
        "event_index": source["event_index"],
        "frame": source["frame"],
        "time_sec": source["time_sec"],
        "player": source["player"],
        "stroke_type": source["stroke"]["stroke_type"],
        "stroke_confidence": source["stroke"]["confidence"],
        "pose_observation": {
            "source_start_frame": source["frame"] - 1,
            "source_end_frame": source["frame"] + 1,
            "confidence": 0.7,
            "posture_candidate": "neutral",
            "posture_confidence": 0.6,
            "secondary_cues": [],
            "limitations": [],
        },
        "court_observation": {
            "source_frame": source["frame"],
            "confidence": 0.8,
            "depth_zone": source["court_position"]["depth_zone"],
            "position_change_from_previous_same_player_hit": "unknown",
            "limitations": [],
        },
        "shuttle_observation": {
            "start_frame": source["frame"] - 1,
            "end_frame": source["frame"] + 1,
            "confidence": 0.6,
            "incoming_image_direction": "unknown",
            "outgoing_image_direction": "right",
            "trajectory_change_candidate": "unknown",
            "limitations": [],
        },
        "warnings": [],
    }


def _payloads():
    source_events = [
        _source_event(1280, 156182, "b", "rear"),
        _source_event(1281, 156197, "a", "front"),
    ]
    package = {
        "rally": {
            "segment_index": 144,
            "start_sec": 5204.633,
            "end_sec": 5221.833,
            "duration_sec": 17.2,
            "score": {"a": 22, "b": 21},
            "server": None,
        },
        "events": source_events,
    }
    generated = {
        "schema_version": "experimental-enriched-rally-fact-v3",
        "segment_index": 144,
        "game_index": None,
        "start_sec": 5204.633,
        "end_sec": 5221.833,
        "duration_sec": 17.2,
        "score": {"a": 22, "b": 21},
        "server": None,
        "events": [_generated_event(event) for event in source_events],
        "rally_length": 2,
        "highlight_score": None,
        "tactical_candidates": [
            {
                "pattern_type": "rear_to_front_stroke_transition",
                "description": "球路由後場轉入前場。",
                "confidence": 0.8,
                "salience": 0.7,
                "start_event_index": 1280,
                "end_event_index": 1281,
                "players": ["a", "b"],
                "evidence": [
                    {
                        "stage": "court_detection",
                        "event_indexes": [1280, 1281],
                        "frames": [156182, 156197],
                    }
                ],
                "limitations": [],
            }
        ],
        "warnings": [],
    }
    return package, generated


def test_v3_output_is_schema_and_provenance_validated():
    package, payload = _payloads()
    generated = SCRIPT["EnrichedRallyFactV3"].model_validate(payload)

    SCRIPT["validate_against_package"](generated, package)


def test_v3_schema_rejects_forehand_backhand_field():
    _, payload = _payloads()
    payload["events"][0]["pose_observation"][
        "forehand_backhand_candidate"
    ] = "forehand"

    with pytest.raises(ValidationError, match="forehand_backhand_candidate"):
        SCRIPT["EnrichedRallyFactV3"].model_validate(payload)


def test_v3_schema_rejects_hitting_arm_field():
    _, payload = _payloads()
    payload["events"][0]["pose_observation"][
        "hitting_arm_candidate"
    ] = "right"

    with pytest.raises(ValidationError, match="hitting_arm_candidate"):
        SCRIPT["EnrichedRallyFactV3"].model_validate(payload)


def test_v3_provenance_rejects_changed_rear_zone():
    package, payload = _payloads()
    payload["events"][0]["court_observation"]["depth_zone"] = "front"
    generated = SCRIPT["EnrichedRallyFactV3"].model_validate(payload)

    with pytest.raises(SCRIPT["V3ExperimentError"], match="court observation"):
        SCRIPT["validate_against_package"](generated, package)


def test_v3_provenance_rejects_untraceable_tactical_frame():
    package, payload = _payloads()
    payload["tactical_candidates"][0]["evidence"][0]["frames"] = [999999]
    generated = SCRIPT["EnrichedRallyFactV3"].model_validate(payload)

    with pytest.raises(SCRIPT["V3ExperimentError"], match="not traceable"):
        SCRIPT["validate_against_package"](generated, package)


@pytest.mark.parametrize(
    "pattern_type",
    [
        "front_court_exchange",
        "rear_court_exchange",
        "repeated_posture_pattern",
    ],
)
def test_compact_prompt_pattern_types_are_accepted(pattern_type):
    package, payload = _payloads()
    payload["tactical_candidates"][0]["pattern_type"] = pattern_type
    generated = SCRIPT["EnrichedRallyFactV3"].model_validate(payload)

    SCRIPT["validate_against_package"](generated, package)

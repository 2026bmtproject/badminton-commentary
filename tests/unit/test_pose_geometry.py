import copy
import json
from types import SimpleNamespace

import pytest

from badminton_commentary.analysis.pose_geometry import compute_pose_geometry


def _record(
    delta: int = 0,
    *,
    confidence: float = 0.9,
    shift_x: float = 0.0,
):
    keypoints = {
        "left_shoulder": (-50.0 + shift_x, 0.0, confidence),
        "right_shoulder": (50.0 + shift_x, 0.0, confidence),
        "left_wrist": (-100.0 + shift_x, 0.0, confidence),
        "right_wrist": (100.0 + shift_x, 0.0, confidence),
        "left_hip": (-50.0 + shift_x, 100.0, confidence),
        "right_hip": (50.0 + shift_x, 100.0, confidence),
        "left_knee": (-50.0 + shift_x, 150.0, confidence),
        "right_knee": (50.0 + shift_x, 150.0, confidence),
        "left_ankle": (-75.0 + shift_x, 200.0, confidence),
        "right_ankle": (75.0 + shift_x, 200.0, confidence),
    }
    return SimpleNamespace(
        frame=100 + delta,
        frame_delta=delta,
        keypoints=keypoints,
    )


def _geometry(records):
    return compute_pose_geometry(
        records,
        hit_frame=100,
        source_start_frame=92,
        source_end_frame=110,
    )


def test_pose_geometry_uses_full_minus8_plus10_window():
    records = [_record(delta) for delta in range(-8, 11)]
    records[-1].keypoints["left_ankle"] = (-100.0, 200.0, 0.9)
    records[-1].keypoints["right_ankle"] = (100.0, 200.0, 0.9)

    result = _geometry(records)

    assert result.source_start_frame == 92
    assert result.source_end_frame == 110
    assert result.step_width.max_ratio == pytest.approx(2.0)
    assert result.step_width.max_frame_delta == 10


def test_pose_confidence_is_clamped_for_geometry():
    result = _geometry([_record(confidence=1.4)])

    assert result.step_width.confidence == 1.0
    assert result.feature_confidence == 1.0


def test_pose_raw_artifact_is_not_mutated():
    record = _record(confidence=1.4)
    original = copy.deepcopy(record.keypoints)

    _geometry([record])

    assert record.keypoints == original
    assert record.keypoints["left_shoulder"][2] == 1.4


def test_step_width_is_normalized_by_torso_length():
    result = _geometry([_record()])

    assert result.step_width.at_hit_ratio == pytest.approx(1.5)


def test_knee_angle_geometry():
    bent = _record()
    bent.keypoints["left_ankle"] = (0.0, 150.0, 0.9)
    bent.keypoints["right_ankle"] = (50.0, 200.0, 0.9)
    result = _geometry([bent])

    assert result.knee_flexion.left_angle_deg_at_hit == pytest.approx(90.0)
    assert result.knee_flexion.right_angle_deg_at_hit == pytest.approx(180.0)


def test_body_height_geometry():
    record = _record()
    record.keypoints["left_ankle"] = (-75.0, 250.0, 0.9)
    record.keypoints["right_ankle"] = (75.0, 250.0, 0.9)

    result = _geometry([record])

    assert result.body_height.at_hit_ratio == pytest.approx(1.5)


def test_torso_lean_geometry():
    vertical = _geometry([_record()])
    leaning_record = _record()
    leaning_record.keypoints["left_shoulder"] = (50.0, 0.0, 0.9)
    leaning_record.keypoints["right_shoulder"] = (150.0, 0.0, 0.9)
    leaning = _geometry([leaning_record])

    assert vertical.torso_lean.angle_deg_at_hit == pytest.approx(0.0)
    assert vertical.torso_lean.direction_at_hit == "none"
    assert leaning.torso_lean.angle_deg_at_hit == pytest.approx(45.0)
    assert leaning.torso_lean.direction_at_hit == "right"


def test_wrist_reach_geometry():
    result = _geometry([_record()])

    assert result.wrist_reach.left_ratio_at_hit == pytest.approx(1.0)
    assert result.wrist_reach.right_ratio_at_hit == pytest.approx(1.0)
    assert result.wrist_reach.max_side in {"left", "right"}


def test_body_displacement_geometry():
    records = [_record(delta, shift_x=delta * 10.0) for delta in range(-8, 11)]

    result = _geometry(records)

    assert result.body_displacement.pre_to_hit_ratio == pytest.approx(0.7)
    assert result.body_displacement.hit_to_post_ratio == pytest.approx(0.9)
    assert result.body_displacement.window_total_ratio == pytest.approx(1.6)
    assert result.body_displacement.max_single_frame_ratio == pytest.approx(0.1)


def test_missing_pose_points_return_null_not_guess():
    record = _record()
    record.keypoints["left_shoulder"] = (0.0, 0.0, 0.1)

    result = _geometry([record])

    assert result.step_width.at_hit_ratio is None
    assert result.step_width.confidence == 0.0
    assert "insufficient_valid_step_width_frames" in result.limitations


def test_feature_confidence_is_deterministic():
    records = [_record(delta, confidence=0.73) for delta in range(-8, 11)]

    first = _geometry(records)
    second = _geometry(records)

    assert first.feature_confidence == pytest.approx(0.73)
    assert first == second


def test_no_pose_semantics_are_generated():
    serialized = json.dumps(_geometry([_record()]).model_dump())

    for forbidden in (
        "hitting_arm",
        "posture_candidate",
        "is_lunge",
        "is_jump",
        "is_reaching",
        "is_low",
        "is_defensive",
        "is_aggressive",
    ):
        assert forbidden not in serialized

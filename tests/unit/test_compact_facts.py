from pathlib import Path

from badminton_commentary.adapters import (
    CourtCalibration,
    CourtDetectionStage,
    CourtPositionToPlayer,
    PoseFrame,
    SelectedPoseStage,
    SelectedShuttleStage,
    SelectedVisionStages,
    ShuttlePoint,
    StagePaths,
    read_upstream_stages,
)
from badminton_commentary.facts import build_compact_rally_facts


FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "upstream_stages"


def pose_keypoints(center_x: float, feet_y: float):
    points = [[center_x, feet_y - 2, 0.9] for _ in range(17)]
    points[5] = [center_x - 0.3, feet_y - 1.5, 0.9]
    points[6] = [center_x + 0.3, feet_y - 1.5, 0.9]
    points[9] = [center_x - 0.9, feet_y - 1.0, 0.9]
    points[10] = [center_x + 0.4, feet_y - 0.9, 0.9]
    points[11] = [center_x - 0.2, feet_y - 0.8, 0.9]
    points[12] = [center_x + 0.2, feet_y - 0.8, 0.9]
    points[15] = [center_x - 0.2, feet_y, 0.9]
    points[16] = [center_x + 0.2, feet_y, 0.9]
    return points


def pose(frame: int, player: str, x: float, y: float):
    return PoseFrame(
        frame=frame,
        segment_index=1,
        player=player,
        keypoints=pose_keypoints(x, y),
        bbox=(x - 1, y - 3, x + 1, y),
    )


def shuttle_points():
    points = []
    for event_frame in (110, 130, 160):
        for offset in (-4, -2, 0, 2, 4):
            points.append(
                ShuttlePoint(
                    frame=event_frame + offset,
                    segment_index=1,
                    method="inpaint",
                    x=float(event_frame + offset),
                    y=float(200 - event_frame - offset),
                    visible=True,
                    confidence=0.9,
                )
            )
            points.append(
                ShuttlePoint(
                    frame=event_frame + offset,
                    segment_index=1,
                    method="viterbi",
                    x=float(400 - event_frame - offset),
                    y=float(event_frame + offset),
                    visible=True,
                    confidence=0.95,
                )
            )
    return points


def stages_with_vision(*, court_confirmed=True):
    stages = read_upstream_stages(StagePaths.from_stage_root(FIXTURE_ROOT))
    vision = SelectedVisionStages(
        segment_index=1,
        pose=SelectedPoseStage(
            segment_index=1,
            frames=[
                pose(110, "top", 2.0, 2.0),
                pose(130, "bottom", 4.0, 10.0),
                pose(160, "top", 2.5, 5.0),
            ],
        ),
        court_detection=CourtDetectionStage(
            courts=[
                CourtCalibration(
                    corners=[(0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4)],
                    homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    segment_index=None,
                )
            ],
            detection_failed=False,
            confirmed=court_confirmed,
        ),
        shuttle_tracking=SelectedShuttleStage(
            segment_index=1,
            fps=30,
            points=shuttle_points(),
        ),
    )
    return stages.model_copy(update={"vision": vision})


def test_compact_builder_joins_multimodal_features_without_raw_arrays():
    compact = build_compact_rally_facts(
        stages=stages_with_vision(),
        segment_index=1,
        court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
    )

    assert [item.event_index for item in compact.events] == [2, 3, 4]
    assert all(item.pose is not None for item in compact.events)
    assert all(item.court_position is not None for item in compact.events)
    assert all(item.shuttle_path is not None for item in compact.events)
    assert compact.events[0].pose.hitting_arm_candidate == "left"
    assert compact.events[0].court_position.depth_zone == "rear"
    assert compact.events[2].court_position.depth_zone == "front"
    assert compact.events[2].court_position.displacement_from_previous_hit_m > 0
    assert compact.events[0].shuttle_path.coordinate_space == "image"
    assert compact.events[0].shuttle_path.sample_count == 5
    assert "keypoints" not in compact.model_dump_json()
    assert compact.warnings == []


def test_unconfirmed_court_disables_position_facts():
    compact = build_compact_rally_facts(
        stages=stages_with_vision(court_confirmed=False),
        segment_index=1,
        court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
    )

    assert all(item.court_position is None for item in compact.events)
    assert "court_calibration_unconfirmed" in compact.warnings


def test_missing_vision_stages_degrade_without_losing_strokes():
    compact = build_compact_rally_facts(
        stages=read_upstream_stages(StagePaths.from_stage_root(FIXTURE_ROOT)),
        segment_index=1,
        court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
    )

    assert len(compact.events) == 3
    assert all(item.pose is None for item in compact.events)
    assert all(item.shuttle_path is None for item in compact.events)
    assert compact.warnings == [
        "pose_stage_missing",
        "shuttle_stage_missing",
        "court_stage_missing",
    ]


def test_selected_vision_segment_must_match_requested_rally():
    stages = stages_with_vision()
    stages.vision = stages.vision.model_copy(update={"segment_index": 0})

    try:
        build_compact_rally_facts(
            stages=stages,
            segment_index=1,
            court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
        )
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("mismatched selected vision segment was accepted")

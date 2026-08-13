import json
import runpy
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from badminton_commentary.adapters import CourtPositionToPlayer
from badminton_commentary.adapters.upstream import (
    EventDetectionStage,
    MatchSegmentationStage,
    ScoreRecognitionStage,
    StrokeClassificationStage,
    UpstreamHitEvent,
    UpstreamScoreRally,
    UpstreamStageData,
    UpstreamStroke,
)
from badminton_commentary.adapters.vision import (
    CourtCalibration,
    CourtDetectionStage,
    PoseFrame,
    SelectedPoseStage,
    SelectedShuttleStage,
    SelectedVisionStages,
    ShuttlePoint,
)
from badminton_commentary.schemas import Segment


REPO_ROOT = Path(__file__).parents[2]
FIXTURE_ROOT = REPO_ROOT / "fixtures" / "upstream_stages"
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "ttyvsasy"
    / "scripts"
    / "package_direct_rallyfact.py"
)
SCRIPT = runpy.run_path(str(SCRIPT_PATH))
POSE_KEYPOINTS = set(SCRIPT["POSE_KEYPOINT_INDEXES"])
FORBIDDEN_SEMANTIC_FIELDS = {
    "forehand_backhand_candidate",
    "hitting_arm_candidate",
    "posture_candidate",
    "lunge",
    "deep_lunge",
    "jump",
    "dive",
    "recovery",
    "incoming_direction",
    "outgoing_direction",
    "trajectory_change",
    "tactical_candidates",
    "phase",
    "initiative",
    "winner",
    "ending_stroke",
}


def _keypoints(frame: int, player_offset: int) -> list[list[float]]:
    return [
        [130.0 + index, 200.0 + player_offset + index, 0.8 + index / 100]
        for index in range(17)
    ]


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    root = tmp_path_factory.mktemp("event-centric")
    stage_root = root / "stages"
    shutil.copytree(FIXTURE_ROOT, stage_root)

    events_path = stage_root / "event_detection/events.json"
    events_payload = json.loads(events_path.read_text(encoding="utf-8"))
    events_payload["events"][2]["frame"] = 100
    events_payload["events"][4]["frame"] = 199
    events_path.write_text(json.dumps(events_payload), encoding="utf-8")
    strokes_path = stage_root / "stroke_classification/strokes.json"
    strokes_payload = json.loads(strokes_path.read_text(encoding="utf-8"))
    strokes_payload["strokes"][2]["frame"] = 100
    strokes_payload["strokes"][4]["frame"] = 199
    strokes_path.write_text(json.dumps(strokes_payload), encoding="utf-8")

    pose_frames = []
    for event_frame, hitter in ((100, "top"), (130, "bottom"), (199, "top")):
        for frame in range(event_frame - 9, event_frame + 12):
            for player in ("top", "bottom"):
                player_offset = 100 if player == "bottom" else 0
                if event_frame == 199 and player == "top":
                    player_offset = 40
                pose_frames.append(
                    {
                        "frame": frame,
                        "segment_index": 1,
                        "player": player,
                        "keypoints": _keypoints(
                            frame,
                            player_offset,
                        ),
                        "bbox": [0, 0, 20, 40],
                    }
                )
    pose_frames.extend(
        [
            {
                "frame": 99,
                "segment_index": 0,
                "player": "top",
                "keypoints": _keypoints(99, 0),
                "bbox": [0, 0, 20, 40],
            },
            {
                "frame": 200,
                "segment_index": 2,
                "player": "top",
                "keypoints": _keypoints(200, 0),
                "bbox": [0, 0, 20, 40],
            },
        ]
    )
    (stage_root / "pose").mkdir()
    (stage_root / "pose/pose.json").write_text(
        json.dumps({"frames": pose_frames}),
        encoding="utf-8",
    )

    homography = [[10, 0, 100], [0, 10, 200], [0, 0, 1]]
    (stage_root / "court_detection").mkdir()
    (stage_root / "court_detection/court.json").write_text(
        json.dumps(
            {
                "courts": [
                    {
                        "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "homography": homography,
                        "segment_index": None,
                    }
                ],
                "detection_failed": False,
                "confirmed": False,
            }
        ),
        encoding="utf-8",
    )

    shuttle_points = []
    for frame in range(100, 200):
        shuttle_points.extend(
            [
                {
                    "frame": frame,
                    "segment_index": 1,
                    "method": "inpaint",
                    "x": float(frame) if frame % 4 else None,
                    "y": float(frame + 1) if frame % 4 else None,
                    "visible": frame % 4 != 0,
                    "confidence": 0 if frame == 130 else 0.9,
                },
                {
                    "frame": frame,
                    "segment_index": 1,
                    "method": "viterbi",
                    "x": float(-frame),
                    "y": float(-frame),
                    "visible": True,
                    "confidence": 0.7,
                },
            ]
        )
    (stage_root / "shuttle_tracking").mkdir()
    (stage_root / "shuttle_tracking/shuttle.json").write_text(
        json.dumps({"points": shuttle_points}),
        encoding="utf-8",
    )

    output_dir = root / "seg0001"
    result = SCRIPT["build_package"](
        stage_root=stage_root,
        output_dir=output_dir,
        segment_index=1,
        mapping=CourtPositionToPlayer(top="b", bottom="a"),
    )
    payload = json.loads(
        (output_dir / "rally_stage_input.json").read_text(encoding="utf-8")
    )
    return payload, result, homography


def test_all_selected_rally_events_preserved(packaged):
    payload, _, _ = packaged
    assert [event["frame"] for event in payload["events"]] == [100, 130, 199]


def test_event_indexes_not_renumbered(packaged):
    payload, _, _ = packaged
    assert [event["event_index"] for event in payload["events"]] == [2, 3, 4]


def test_strokes_joined_by_event_index(packaged):
    payload, _, _ = packaged
    assert [event["stroke"]["stroke_type"] for event in payload["events"]] == [
        "高遠球",
        "殺球",
        "小球",
    ]
    assert [event["stroke"]["source_frame"] for event in payload["events"]] == [
        100,
        130,
        199,
    ]


def test_pose_window_is_minus8_plus10(packaged):
    _, result, _ = packaged
    payload = json.loads(
        (result.directory / "rally_stage_input_debug.json").read_text(
            encoding="utf-8"
        )
    )
    assert [pose["frame_delta"] for pose in payload["events"][0]["pose_window"]] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert [pose["frame_delta"] for pose in payload["events"][1]["pose_window"]] == [
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert [pose["frame_delta"] for pose in payload["events"][2]["pose_window"]] == [
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
    ]


def test_pose_window_contains_only_hitting_player(packaged):
    _, result, _ = packaged
    payload = json.loads(
        (result.directory / "rally_stage_input_debug.json").read_text(
            encoding="utf-8"
        )
    )
    for event in payload["events"]:
        assert {pose["player"] for pose in event["pose_window"]} == {
            event["player"]
        }
    assert [event["player"] for event in payload["events"]] == ["b", "a", "b"]


def test_pose_window_contains_required_posture_keypoints(packaged):
    _, result, _ = packaged
    payload = json.loads(
        (result.directory / "rally_stage_input_debug.json").read_text(
            encoding="utf-8"
        )
    )
    for event in payload["events"]:
        for pose in event["pose_window"]:
            assert set(pose["keypoints"]) == POSE_KEYPOINTS
            assert pose["keypoints"]["nose"][0] == pytest.approx(130)
            assert pose["keypoints"]["left_ankle"][2] == pytest.approx(0.95)


def test_pose_window_contains_knees(packaged):
    _, result, _ = packaged
    payload = json.loads(
        (result.directory / "rally_stage_input_debug.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        {"left_knee", "right_knee"} <= set(pose["keypoints"])
        for event in payload["events"]
        for pose in event["pose_window"]
    )


def test_pose_window_excludes_eyes_and_ears(packaged):
    _, result, _ = packaged
    payload = json.loads(
        (result.directory / "rally_stage_input_debug.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {"left_eye", "right_eye", "left_ear", "right_ear"}
    assert all(
        not excluded & set(pose["keypoints"])
        for event in payload["events"]
        for pose in event["pose_window"]
    )


def test_pose_window_never_crosses_segment_boundary(packaged):
    _, result, _ = packaged
    payload = json.loads(
        (result.directory / "rally_stage_input_debug.json").read_text(
            encoding="utf-8"
        )
    )
    start = payload["rally"]["start_frame"]
    end = payload["rally"]["end_frame"]
    assert all(
        start <= pose["frame"] <= end
        for event in payload["events"]
        for pose in event["pose_window"]
    )


def test_compact_package_has_pose_features(packaged):
    payload, _, _ = packaged

    assert payload["context"]["package_version"] == (
        "direct-rallyfact-event-centric-v4"
    )
    assert payload["context"]["pose_geometry_precomputed"] is True
    assert all(event["pose_features"] is not None for event in payload["events"])
    assert payload["events"][1]["pose_features"]["source_start_frame"] == 122
    assert payload["events"][1]["pose_features"]["source_end_frame"] == 140


def test_compact_package_has_fixed_pose_keyframes(packaged):
    payload, _, _ = packaged

    assert payload["context"]["pose_keyframe_deltas"] == [-8, -4, 0, 4, 8, 10]
    assert [
        item["frame_delta"] for item in payload["events"][1]["pose_keyframes"]
    ] == [-8, -4, 0, 4, 8, 10]
    assert set(payload["events"][1]["pose_keyframes"][0]["keypoints"]) == {
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }


def test_compact_package_does_not_have_full_pose_window(packaged):
    payload, _, _ = packaged

    assert all("pose_window" not in event for event in payload["events"])


def test_debug_package_keeps_full_pose_window(packaged):
    _, result, _ = packaged
    debug = json.loads(
        (result.directory / "rally_stage_input_debug.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(debug["events"][1]["pose_window"]) == 19
    assert result.full_debug_bytes > result.llm_input_bytes
    assert result.reduction_ratio > 1


def test_shuttle_window_is_plus_minus_six(packaged):
    payload, _, _ = packaged
    for event in payload["events"]:
        shuttle = event["shuttle_window"]
        assert shuttle["start_frame"] == max(100, event["frame"] - 6)
        assert shuttle["end_frame"] == min(199, event["frame"] + 6)
        assert all(
            shuttle["start_frame"] <= point["frame"] <= shuttle["end_frame"]
            for point in shuttle["points"]
        )


def test_only_selected_shuttle_method_is_included(packaged):
    payload, _, _ = packaged
    for event in payload["events"]:
        shuttle = event["shuttle_window"]
        assert shuttle["method"] == "inpaint"
        assert all(point["x"] > 0 for point in shuttle["points"])


def test_invalid_shuttle_points_are_excluded(packaged):
    payload, _, _ = packaged
    middle = payload["events"][1]["shuttle_window"]
    assert 130 not in {point["frame"] for point in middle["points"]}
    assert middle["excluded_points"] > 0


def test_court_projection_uses_inverse_homography(packaged):
    payload, _, _ = packaged
    court = payload["events"][0]["court_position"]
    assert court["image_point"] == pytest.approx([145.5, 215.5])
    assert court["court_point_m"] == pytest.approx([4.55, 1.55])
    assert "court_calibration" not in payload


def test_confirmed_false_calibration_is_still_usable(packaged):
    payload, _, _ = packaged
    assert all(event["court_position"] is not None for event in payload["events"])


def test_depth_zone_is_player_relative(packaged):
    payload, _, _ = packaged
    assert payload["events"][0]["stage_player"] == "top"
    assert payload["events"][0]["court_position"]["depth_zone"] == "rear"
    assert payload["events"][1]["stage_player"] == "bottom"
    assert payload["events"][1]["court_position"]["depth_zone"] == "rear"


def test_same_player_depth_change_is_deterministic(packaged):
    payload, _, _ = packaged
    first, _, third = payload["events"]
    assert first["court_position"][
        "position_change_from_previous_same_player_hit"
    ] == "unknown"
    assert third["court_position"]["depth_zone"] == "front"
    assert third["court_position"][
        "position_change_from_previous_same_player_hit"
    ] == "forward"
    depth_change = SCRIPT["_depth_change"]
    assert depth_change(0.50, 0.57) == "stable"
    assert depth_change(0.50, 0.59) == "backward"
    assert depth_change(0.50, 0.41) == "forward"


def test_position_behind_own_baseline_is_kept_as_rear_court():
    keypoints = [(0.0, 0.0, 0.0) for _ in range(17)]
    keypoints[15] = (2.6, -1.17, 0.9)
    keypoints[16] = (3.0, -1.17, 0.9)
    pose = SimpleNamespace(
        frame=100,
        keypoints=keypoints,
        bbox=(2.0, -2.0, 3.5, -1.0),
    )

    position, normalized_depth = SCRIPT["_court_position_for_event"](
        event_frame=100,
        stage_player="top",
        pose_window=[pose],
        inverse_homography=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        previous_depth=None,
    )

    assert position is not None
    assert position.court_point_m == pytest.approx((2.8, -1.17))
    assert position.depth_zone == "rear"
    assert "projected_point_behind_own_baseline" in position.limitations
    assert position.projection_confidence == pytest.approx(0.9 * 0.85)
    assert normalized_depth is not None and normalized_depth > 1


def test_baseline_extension_does_not_accept_lateral_projection_outlier():
    keypoints = [(0.0, 0.0, 0.0) for _ in range(17)]
    keypoints[15] = (-0.4, -1.0, 0.9)
    keypoints[16] = (-0.2, -1.0, 0.9)
    pose = SimpleNamespace(
        frame=100,
        keypoints=keypoints,
        bbox=(-0.5, -2.0, -0.1, -1.0),
    )

    position, normalized_depth = SCRIPT["_court_position_for_event"](
        event_frame=100,
        stage_player="top",
        pose_window=[pose],
        inverse_homography=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        previous_depth=None,
    )

    assert position is None
    assert normalized_depth is None


def test_baseline_extension_only_applies_behind_players_own_baseline():
    keypoints = [(0.0, 0.0, 0.0) for _ in range(17)]
    keypoints[15] = (2.8, 14.0, 0.9)
    keypoints[16] = (3.0, 14.0, 0.9)
    pose = SimpleNamespace(
        frame=100,
        keypoints=keypoints,
        bbox=(2.0, 13.0, 3.5, 14.0),
    )

    position, normalized_depth = SCRIPT["_court_position_for_event"](
        event_frame=100,
        stage_player="top",
        pose_window=[pose],
        inverse_homography=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        previous_depth=None,
    )

    assert position is None
    assert normalized_depth is None


def test_no_forehand_backhand_fields_generated(packaged):
    payload, _, _ = packaged
    serialized = json.dumps(payload["events"], ensure_ascii=False)
    assert "forehand" not in serialized.lower()
    assert "backhand" not in serialized.lower()


def test_no_posture_semantic_generated(packaged):
    payload, _, _ = packaged
    serialized = json.dumps(payload["events"], ensure_ascii=False)
    for field in FORBIDDEN_SEMANTIC_FIELDS - {"tactical_candidates"}:
        assert f'"{field}"' not in serialized


def test_no_hitting_arm_semantic_generated(packaged):
    payload, _, _ = packaged
    serialized = json.dumps(payload["events"], ensure_ascii=False).lower()

    assert "hitting_arm" not in serialized
    assert "forehand" not in serialized
    assert "backhand" not in serialized


def test_no_tactical_fields_are_generated(packaged):
    payload, result, _ = packaged
    assert set(payload) == {"context", "rally", "events"}
    assert "tactical_candidates" not in payload
    assert result.raw_slice_estimated_bytes > result.output_bytes
    assert result.reduction_ratio > 1
    assert result.zip_path.is_file()
    with zipfile.ZipFile(result.zip_path) as archive:
        names = archive.namelist()
    assert "seg0001/rally_stage_input.json" in names
    assert "seg0001/rally_stage_input_debug.json" in names
    assert "seg0001/prompt_with_rally_stage_input.txt" in names
    assert not any("stages/pose" in name for name in names)


def test_compact_v3_prompt_contains_current_contract():
    prompt = SCRIPT["_render_prompt"](
        segment_index=144,
        mapping=CourtPositionToPlayer(top="b", bottom="a"),
    )

    assert prompt.startswith(
        "# Experimental Enriched RallyFact v3 — Compact Prompt"
    )
    assert '"schema_version": "experimental-enriched-rally-fact-v3"' in prompt
    assert '"front_court_exchange"' in prompt
    assert '"rear_court_exchange"' in prompt
    assert '"repeated_posture_pattern"' in prompt
    assert "hitting_arm_candidate" not in prompt
    assert "`pose_features`" in prompt
    assert "`pose_keyframes`" in prompt
    assert "Never infer winner" in prompt
    assert "{{" not in prompt


def test_package_overwrite_preserves_gemini_experiment_outputs(tmp_path):
    for filename in (
        "gemini_enriched_rally_fact_v3.json",
        "gemini_response_v3_raw.txt",
        "gemini_v3_run_metadata.json",
    ):
        (tmp_path / filename).write_text("stale", encoding="utf-8")

    SCRIPT["_remove_legacy_outputs"](tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "gemini_enriched_rally_fact_v3.json",
        "gemini_response_v3_raw.txt",
        "gemini_v3_run_metadata.json",
    }


def _seg144_stages() -> UpstreamStageData:
    segment = Segment(
        start_frame=156139,
        end_frame=156655,
        start_sec=5204.633,
        end_sec=5221.833,
        duration_sec=17.2,
    )
    segments = [
        Segment(
            start_frame=0,
            end_frame=1,
            start_sec=0,
            end_sec=1 / 30,
            duration_sec=1 / 30,
        )
        for _ in range(144)
    ] + [segment]
    selected_frames = [
        156182,
        156197,
        156217,
        156231,
        156254,
        156280,
        156327,
        156348,
        156372,
        156400,
        156426,
        156464,
        156506,
        156525,
        156550,
        156559,
        156580,
    ]
    events = [UpstreamHitEvent(frame=0) for _ in range(1280)] + [
        UpstreamHitEvent(frame=frame) for frame in selected_frames
    ]
    strokes = []
    poses = []
    shuttle = []
    for offset, frame in enumerate(selected_frames):
        event_index = 1280 + offset
        position = "top" if offset % 2 == 0 else "bottom"
        strokes.append(
            UpstreamStroke(
                event_index=event_index,
                frame=frame,
                segment_index=144,
                player=position,
                stroke_type="測試球種",
                confidence=0.8,
            )
        )
        for pose_frame in range(frame - 2, frame + 3):
            poses.append(
                PoseFrame(
                    frame=pose_frame,
                    segment_index=144,
                    player=position,
                    keypoints=_keypoints(
                        pose_frame,
                        0 if position == "top" else 100,
                    ),
                    bbox=(0, 0, 20, 40),
                )
            )
        for shuttle_frame in range(frame - 6, frame + 7):
            if any(point.frame == shuttle_frame for point in shuttle):
                continue
            shuttle.append(
                ShuttlePoint(
                    frame=shuttle_frame,
                    segment_index=144,
                    method="inpaint",
                    x=float(shuttle_frame),
                    y=float(shuttle_frame),
                    visible=True,
                    confidence=0.9,
                )
            )
    return UpstreamStageData(
        match_segmentation=MatchSegmentationStage(fps=30, segments=segments),
        event_detection=EventDetectionStage(events=events),
        score_recognition=ScoreRecognitionStage(
            rallies=[
                UpstreamScoreRally(
                    segment_index=144,
                    score_a=22,
                    score_b=21,
                )
            ]
        ),
        stroke_classification=StrokeClassificationStage(
            shuttle_method="inpaint",
            strokes=strokes,
        ),
        vision=SelectedVisionStages(
            segment_index=144,
            pose=SelectedPoseStage(segment_index=144, frames=poses),
            court_detection=CourtDetectionStage(
                courts=[
                    CourtCalibration(
                        corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
                        homography=[
                            [10, 0, 100],
                            [0, 10, 200],
                            [0, 0, 1],
                        ],
                        segment_index=None,
                    )
                ],
                detection_failed=False,
                confirmed=False,
            ),
            shuttle_tracking=SelectedShuttleStage(
                segment_index=144,
                fps=30,
                points=shuttle,
            ),
        ),
    )


def test_seg0144_all_17_events_preserved():
    debug_input = SCRIPT["_build_event_centric_input"](
        stages=_seg144_stages(),
        segment_index=144,
        mapping=CourtPositionToPlayer(top="b", bottom="a"),
    )
    payload = SCRIPT["_to_llm_input"](debug_input).model_dump(mode="json")

    assert len(payload["events"]) == 17
    assert [event["event_index"] for event in payload["events"][-3:]] == [
        1294,
        1295,
        1296,
    ]
    assert all(event["pose_features"] for event in payload["events"][-3:])
    assert all(event["pose_keyframes"] for event in payload["events"][-3:])
    assert all("pose_window" not in event for event in payload["events"])
    assert all(event["court_position"] for event in payload["events"][-3:])
    assert all(event["shuttle_window"] is not None for event in payload["events"])


def test_seg0144_last_events_have_pose_and_court_data():
    payload = SCRIPT["_build_event_centric_input"](
        stages=_seg144_stages(),
        segment_index=144,
        mapping=CourtPositionToPlayer(top="b", bottom="a"),
    ).model_dump(mode="json")

    for event in payload["events"][-3:]:
        assert event["event_index"] in {1294, 1295, 1296}
        assert event["pose_window"]
        assert event["court_position"] is not None
        assert event["shuttle_window"] is not None


def test_stroke_frame_mismatch_is_reported_without_rewriting_source():
    stages = _seg144_stages()
    strokes = list(stages.stroke_classification.strokes)
    strokes[0] = strokes[0].model_copy(update={"frame": 156183})
    stages = stages.model_copy(
        update={
            "stroke_classification": stages.stroke_classification.model_copy(
                update={"strokes": strokes}
            )
        }
    )

    payload = SCRIPT["_build_event_centric_input"](
        stages=stages,
        segment_index=144,
        mapping=CourtPositionToPlayer(top="b", bottom="a"),
    ).model_dump(mode="json")

    assert payload["events"][0]["frame"] == 156182
    assert payload["events"][0]["stroke"]["source_frame"] == 156183
    assert "does not match" in payload["events"][0]["warnings"][0]


def test_detection_failed_court_is_rejected():
    stages = _seg144_stages()
    court = stages.vision.court_detection.model_copy(
        update={"detection_failed": True}
    )
    stages = stages.model_copy(
        update={
            "vision": stages.vision.model_copy(
                update={"court_detection": court}
            )
        }
    )

    payload = SCRIPT["_build_event_centric_input"](
        stages=stages,
        segment_index=144,
        mapping=CourtPositionToPlayer(top="b", bottom="a"),
    ).model_dump(mode="json")

    assert all(event["court_position"] is None for event in payload["events"])

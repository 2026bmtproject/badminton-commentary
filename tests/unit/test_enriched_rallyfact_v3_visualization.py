import runpy
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = runpy.run_path(
    str(
        REPO_ROOT
        / "experiments"
        / "ttyvsasy"
        / "scripts"
        / "visualize_enriched_rallyfact_v3.py"
    )
)


def _event(event_index, frame, player, posture, depth):
    return SimpleNamespace(
        event_index=event_index,
        frame=frame,
        player=player,
        stroke_type="殺球",
        stroke_confidence=0.8,
        pose_observation=SimpleNamespace(
            source_start_frame=frame - 1,
            source_end_frame=frame + 1,
            confidence=0.7,
            posture_candidate=posture,
            posture_confidence=0.6,
            secondary_cues=["airborne_candidate"],
            limitations=[],
        ),
        court_observation=SimpleNamespace(
            source_frame=frame,
            confidence=0.69,
            depth_zone=depth,
            position_change_from_previous_same_player_hit="unknown",
            limitations=["projected_point_behind_own_baseline"],
        ),
        shuttle_observation=SimpleNamespace(
            start_frame=frame - 1,
            end_frame=frame + 1,
            confidence=0.6,
            incoming_image_direction="unknown",
            outgoing_image_direction="down_right",
            trajectory_change_candidate="sharp_redirection",
            limitations=[],
        ),
        warnings=[],
    )


def _pose(frame, player):
    names = {name for edge in SCRIPT["SKELETON_EDGES"] for name in edge}
    return {
        "frame": frame,
        "frame_delta": 0,
        "player": player,
        "keypoints": {
            name: [900 + index * 3, 400 + index * 7, 0.9]
            for index, name in enumerate(sorted(names))
        },
    }


def test_render_ass_places_events_tactics_and_pose_on_local_timeline():
    events = [
        _event(1280, 156182, "b", "jump", "rear"),
        _event(1281, 156197, "a", "neutral", "mid"),
    ]
    enriched = SimpleNamespace(
        events=events,
        tactical_candidates=[
            SimpleNamespace(
                pattern_type="attack_transition",
                description="形成攻擊性 sequence。",
                confidence=0.7,
                salience=0.8,
                start_event_index=1280,
                end_event_index=1281,
                evidence=[SimpleNamespace(stage="stroke_classification")],
            )
        ],
    )
    package = {
        "rally": {
            "start_frame": 156139,
            "end_frame": 156655,
            "fps": 30,
        },
        "events": [
            {"event_index": 1280, "pose_features": {}},
            {"event_index": 1281, "pose_features": {}},
        ],
    }
    debug_package = {
        "events": [
            {"event_index": 1280, "pose_window": [_pose(156182, "b")]},
            {"event_index": 1281, "pose_window": [_pose(156197, "a")]},
        ]
    }

    ass = SCRIPT["render_ass"](
        enriched,
        package,
        debug_package=debug_package,
        model_label="Gemini 3.1 Pro",
    )

    assert "Gemini 3.1 Pro · Enriched RallyFact v3 · SEG144" in ass
    assert "EVENT #1280" in ass
    assert "Pose  jump  conf 0.60" in ass
    assert "Court  rear / unknown  conf 0.69 · behind baseline" in ass
    assert "Shuttle  ? → ↘  change sharp_redirection" in ass
    assert "TACTICAL  attack_transition  E1280–E1281" in ass
    assert "形成攻擊性 sequence。" in ass
    assert r"{\p1}m " in ass
    assert "0:00:01.33" in ass


def test_pose_path_omits_low_confidence_edges():
    pose = _pose(100, "b")
    for keypoint in pose["keypoints"].values():
        keypoint[2] = 0.1

    assert SCRIPT["_pose_path"](pose["keypoints"]) == ""

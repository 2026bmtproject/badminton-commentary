from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "ttyvsasy"
    / "scripts"
    / "experiment_forehand_backhand.py"
)
MODULE = runpy.run_path(str(SCRIPT))
Config = MODULE["Config"]
classify_hit = MODULE["classify_hit"]
render_viewer = MODULE["render_viewer"]
ExperimentOutput = MODULE["ExperimentOutput"]
REVIEW_SCRIPT = SCRIPT.with_name("analyze_forehand_backhand_review.py")
REVIEW_MODULE = runpy.run_path(str(REVIEW_SCRIPT))
HumanReview = REVIEW_MODULE["HumanReview"]
analyze_review = REVIEW_MODULE["analyze_review"]


def _pose(*, elbow_x: float, wrist_x: float):
    keypoints = [(100.0, 100.0, 0.9) for _ in range(17)]
    coordinates = {
        0: (100.0, 75.0, 0.1),
        1: (95.0, 75.0, 0.1),
        2: (105.0, 75.0, 0.1),
        5: (80.0, 100.0, 0.9),
        6: (120.0, 100.0, 0.9),
        7: (60.0, 110.0, 0.9),
        8: (elbow_x, 100.0, 0.9),
        9: (45.0, 110.0, 0.9),
        10: (wrist_x, 100.0, 0.9),
        11: (85.0, 160.0, 0.9),
        12: (115.0, 160.0, 0.9),
    }
    for index, value in coordinates.items():
        keypoints[index] = value
    return SimpleNamespace(
        frame=100,
        keypoints=keypoints,
        bbox=(40.0, 60.0, 175.0, 200.0),
    )


def test_right_handed_lateral_geometry_separates_synthetic_sides():
    config = Config(min_racket_frames=1)
    forehand, forehand_margin, _ = classify_hit(
        hit_frame=100,
        court_position="bottom",
        pose_by_frame={100: _pose(elbow_x=145.0, wrist_x=170.0)},
        left_handed=False,
        config=config,
    )
    backhand, backhand_margin, _ = classify_hit(
        hit_frame=100,
        court_position="bottom",
        pose_by_frame={100: _pose(elbow_x=110.0, wrist_x=80.0)},
        left_handed=False,
        config=config,
    )

    assert forehand == "forehand"
    assert backhand == "backhand"
    assert forehand_margin >= config.min_margin
    assert backhand_margin >= config.min_margin


def test_near_midline_is_unknown():
    side, margin, detail = classify_hit(
        hit_frame=100,
        court_position="bottom",
        pose_by_frame={100: _pose(elbow_x=120.0, wrist_x=120.0)},
        left_handed=False,
        config=Config(min_racket_frames=1),
    )

    assert side is None
    assert margin < Config.min_margin
    assert detail["score_semantics"] == "heuristic_margin_not_probability"


def test_confidence_gate_abstains_when_too_few_racket_frames_are_reliable():
    side, margin, detail = classify_hit(
        hit_frame=100,
        court_position="bottom",
        pose_by_frame={100: _pose(elbow_x=145.0, wrist_x=170.0)},
        left_handed=False,
        config=Config(min_racket_frames=3),
    )

    assert side is None
    assert margin == 0
    assert detail["accepted_racket_frames"] == 1
    assert detail["reason"] == "insufficient high-confidence racket-arm frames"


def test_frame_viewer_contains_review_shortcuts():
    result = ExperimentOutput.model_validate(
        {
            "schema_version": "experimental-forehand-backhand-v1",
            "segment_index": 144,
            "fps": 30,
            "source_start_frame": 100,
            "source_end_frame": 130,
            "player_mapping": {"top": "b", "bottom": "a"},
            "left_handed_players": [],
            "params": {},
            "shots": [
                {
                    "event_index": 5,
                    "frame": 110,
                    "local_frame": 10,
                    "player": "b",
                    "court_position": "top",
                    "hand": "right",
                    "stroke_type": "殺球",
                    "stroke_confidence": 0.9,
                    "side": "forehand",
                    "side_zh": "正手",
                    "heuristic_margin": 0.5,
                    "frames_used": 1,
                    "detail": {},
                }
            ],
            "summary": {},
            "limitations": [],
        }
    )

    rendered = render_viewer(result, "overlay.mp4")

    assert "arrowleft" in rendered
    assert "review('correct')" in rendered
    assert "seg0144_forehand_backhand_human_review.json" in rendered
    assert "experimental-forehand-backhand-human-review-v2" in rendered
    assert "overlay.mp4" in rendered


def test_human_review_metrics_exclude_uncertain_from_selective_accuracy():
    result = ExperimentOutput.model_validate(
        {
            "schema_version": "experimental-forehand-backhand-v1",
            "segment_index": 144,
            "fps": 30,
            "source_start_frame": 100,
            "source_end_frame": 140,
            "player_mapping": {"top": "b", "bottom": "a"},
            "left_handed_players": [],
            "params": {},
            "shots": [
                {
                    "event_index": event_index,
                    "frame": frame,
                    "local_frame": frame - 100,
                    "player": "b",
                    "court_position": "top",
                    "hand": "right",
                    "stroke_type": "平快球",
                    "stroke_confidence": 0.9,
                    "side": side,
                    "side_zh": "未知" if side is None else "正手",
                    "heuristic_margin": margin,
                    "frames_used": 1,
                    "detail": {},
                }
                for event_index, frame, side, margin in (
                    (5, 110, "forehand", 0.5),
                    (6, 120, "backhand", 0.4),
                    (7, 130, None, 0.02),
                )
            ],
            "summary": {},
            "limitations": [],
        }
    )
    review = HumanReview.model_validate(
        {
            "schema_version": "seg144-forehand-backhand-human-review-v1",
            "fps": 30,
            "reviewed_at": "2026-08-13T00:00:00Z",
            "reviews": [
                {
                    "event_index": event_index,
                    "local_frame": frame - 100,
                    "player": "b",
                    "stroke_type": "平快球",
                    "side": side,
                    "side_zh": "未知" if side is None else "正手",
                    "margin": margin,
                    "verdict": verdict,
                }
                for event_index, frame, side, margin, verdict in (
                    (5, 110, "forehand", 0.5, "correct"),
                    (6, 120, "backhand", 0.4, "incorrect"),
                    (7, 130, None, 0.02, "uncertain"),
                )
            ],
        }
    )

    metrics = analyze_review(result, review)

    assert metrics["metrics"]["classifier_coverage"] == 2 / 3
    assert metrics["metrics"]["selective_accuracy"] == 0.5
    assert metrics["confusion_matrix"]["forehand"] == {
        "forehand": 1,
        "backhand": 1,
    }

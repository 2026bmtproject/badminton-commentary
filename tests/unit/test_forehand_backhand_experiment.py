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
REVIEW_SET_SCRIPT = SCRIPT.with_name("analyze_forehand_backhand_review_set.py")
REVIEW_SET_MODULE = runpy.run_path(str(REVIEW_SET_SCRIPT))
aggregate_metrics = REVIEW_SET_MODULE["aggregate_metrics"]
ABLATION_SCRIPT = SCRIPT.with_name(
    "compare_forehand_backhand_orientation_ablation.py"
)
ABLATION_MODULE = runpy.run_path(str(ABLATION_SCRIPT))
Variant = ABLATION_MODULE["Variant"]
score_variant = ABLATION_MODULE["score_variant"]
HOLDOUT_SCRIPT = SCRIPT.with_name(
    "generate_forehand_backhand_c_holdout_review.py"
)
HOLDOUT_MODULE = runpy.run_path(str(HOLDOUT_SCRIPT))
render_holdout_index = HOLDOUT_MODULE["render_index"]
HOLDOUT_ANALYZER_SCRIPT = SCRIPT.with_name(
    "analyze_forehand_backhand_c_holdout_review.py"
)
HOLDOUT_ANALYZER_MODULE = runpy.run_path(str(HOLDOUT_ANALYZER_SCRIPT))
score_holdout_rows = HOLDOUT_ANALYZER_MODULE["score_rows"]


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


def test_court_prior_and_invert_disagreement_are_equivalent_on_true_branch():
    pose_by_frame = {100: _pose(elbow_x=145.0, wrist_x=170.0)}
    original_side, _, original_detail = classify_hit(
        hit_frame=100,
        court_position="top",
        pose_by_frame=pose_by_frame,
        left_handed=False,
        config=Config(min_racket_frames=1),
    )
    prior_side, _, prior_detail = classify_hit(
        hit_frame=100,
        court_position="top",
        pose_by_frame=pose_by_frame,
        left_handed=False,
        config=Config(
            min_racket_frames=1,
            orientation_policy="court_prior",
        ),
    )
    inverted_side, _, inverted_detail = classify_hit(
        hit_frame=100,
        court_position="top",
        pose_by_frame=pose_by_frame,
        left_handed=False,
        config=Config(
            min_racket_frames=1,
            orientation_policy="invert_disagreement",
        ),
    )

    assert original_detail["vote_disagreed_with_court_prior"] is True
    assert original_detail["body_flipped_from_court_prior"] is True
    assert prior_detail["body_flipped_from_court_prior"] is False
    assert inverted_detail["body_flipped_from_court_prior"] is False
    assert original_side != prior_side
    assert prior_side == inverted_side


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
    assert "review('forehand')" in rendered
    assert "review('backhand')" in rendered
    assert "seg0144_forehand_backhand_human_review.json" in rendered
    assert "experimental-forehand-backhand-human-reference-v3" in rendered
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
    assert metrics["metrics"]["abstention_appropriateness"] == 1
    assert metrics["confusion_matrix"]["forehand"] == {
        "forehand": 1,
        "backhand": 1,
    }


def test_incorrect_unknown_is_an_unnecessary_abstention_not_binary_error():
    result = ExperimentOutput.model_validate(
        {
            "schema_version": "experimental-forehand-backhand-v1",
            "segment_index": 140,
            "fps": 30,
            "source_start_frame": 100,
            "source_end_frame": 120,
            "player_mapping": {"top": "b", "bottom": "a"},
            "left_handed_players": [],
            "params": {},
            "shots": [{
                "event_index": 1, "frame": 110, "local_frame": 10,
                "player": "a", "court_position": "bottom", "hand": "right",
                "stroke_type": "小球", "stroke_confidence": 0.9,
                "side": None, "side_zh": "未知", "heuristic_margin": 0,
                "frames_used": 1, "detail": {},
            }],
            "summary": {}, "limitations": [],
        }
    )
    review = HumanReview.model_validate(
        {
            "schema_version": "experimental-forehand-backhand-human-review-v2",
            "segment_index": 140,
            "fps": 30,
            "reviewed_at": "2026-08-13T00:00:00Z",
            "reviews": [{
                "event_index": 1, "local_frame": 10, "player": "a",
                "stroke_type": "小球", "side": None, "side_zh": "未知",
                "margin": 0, "verdict": "incorrect",
            }],
        }
    )

    metrics = analyze_review(result, review)

    assert metrics["metrics"]["selective_accuracy"] is None
    assert metrics["counts"]["binary_incorrect"] == 0
    assert metrics["counts"]["unnecessary_abstentions"] == 1
    assert sum(sum(row.values()) for row in metrics["confusion_matrix"].values()) == 0


def test_review_set_aggregation_exposes_orientation_failure_cluster():
    rows = [
        {
            "event_index": 1,
            "player": "a",
            "stroke_type": "小球",
            "predicted_side": "forehand",
            "heuristic_margin": 0.9,
            "verdict": "incorrect",
            "inferred_reference_side": "backhand",
            "body_flipped_from_court_prior": True,
            "flip_confidence": 1.0,
            "accepted_racket_frames": 5,
        },
        {
            "event_index": 2,
            "player": "b",
            "stroke_type": "殺球",
            "predicted_side": "forehand",
            "heuristic_margin": 0.8,
            "verdict": "correct",
            "inferred_reference_side": "forehand",
            "body_flipped_from_court_prior": False,
            "flip_confidence": 1.0,
            "accepted_racket_frames": 5,
        },
    ]
    summary = aggregate_metrics(
        [{"segment_index": 140, "counts": {}, "metrics": {}, "review_rows": rows}],
        [{"segment_index": 140, "status": "compatible"}],
    )

    flipped = summary["by_orientation_decision"]["flipped_from_court_prior"]
    followed = summary["by_orientation_decision"]["followed_court_prior"]
    assert flipped["selective_accuracy"] == 0
    assert followed["selective_accuracy"] == 1


def test_ablation_accuracy_uses_fixed_denominator_for_unknown():
    result = SimpleNamespace(
        shots=[
            SimpleNamespace(event_index=1, side="forehand"),
            SimpleNamespace(event_index=2, side=None),
        ]
    )
    score, predictions = score_variant(
        variant=Variant("T", "test", Config()),
        results={140: result},
        references={(140, 1): "forehand", (140, 2): "backhand"},
        original_predictions=None,
    )

    assert predictions[(140, 2)] is None
    assert score["accuracy_with_unknown_incorrect"] == 0.5
    assert score["selective_accuracy"] == 1
    assert score["confusion_matrix"]["backhand"]["unknown"] == 1


def test_v3_reference_records_the_side_for_an_incorrect_unknown_candidate():
    result = ExperimentOutput.model_validate(
        {
            "schema_version": "experimental-forehand-backhand-v1",
            "segment_index": 144,
            "fps": 30,
            "source_start_frame": 100,
            "source_end_frame": 120,
            "player_mapping": {"top": "b", "bottom": "a"},
            "left_handed_players": [],
            "params": {},
            "shots": [{
                "event_index": 1, "frame": 110, "local_frame": 10,
                "player": "a", "court_position": "bottom", "hand": "right",
                "stroke_type": "小球", "stroke_confidence": 0.9,
                "side": None, "side_zh": "未知", "heuristic_margin": 0,
                "frames_used": 1, "detail": {},
            }],
            "summary": {}, "limitations": [],
        }
    )
    review = HumanReview.model_validate(
        {
            "schema_version": "experimental-forehand-backhand-human-reference-v3",
            "segment_index": 144,
            "fps": 30,
            "reviewed_at": "2026-08-15T00:00:00Z",
            "reviews": [{
                "event_index": 1, "local_frame": 10, "player": "a",
                "stroke_type": "小球", "side": None, "side_zh": "未知",
                "margin": 0, "reference_side": "backhand",
                "review_status": "labeled", "verdict": "incorrect",
            }],
        }
    )

    metrics = analyze_review(result, review)

    assert metrics["review_rows"][0]["inferred_reference_side"] == "backhand"
    assert metrics["counts"]["unnecessary_abstentions"] == 1


def test_holdout_index_links_each_court_prior_review_page():
    rendered = render_holdout_index(
        {
            "rallies": [
                {
                    "segment_index": 144,
                    "hits": 17,
                    "duration_sec": 17.2,
                    "unknown_candidates": 4,
                    "video_name": "seg0144_C_overlay.mp4",
                }
            ]
        }
    )

    assert "SEG144" in rendered
    assert "seg0144_C_court_prior/frame_review.html" in rendered
    assert "orientation_policy=court_prior" in rendered


def test_holdout_score_counts_unknown_as_incorrect_on_fixed_denominator():
    rows = [
        {
            "event_index": 1,
            "player": "a",
            "stroke_type": "小球",
            "predicted_side": "forehand",
            "inferred_reference_side": "forehand",
            "verdict": "correct",
            "heuristic_margin": 0.8,
        },
        {
            "event_index": 2,
            "player": "b",
            "stroke_type": "高遠球",
            "predicted_side": None,
            "inferred_reference_side": "backhand",
            "verdict": "incorrect",
            "heuristic_margin": 0.0,
        },
        {
            "event_index": 3,
            "player": "a",
            "stroke_type": "平快球",
            "predicted_side": None,
            "inferred_reference_side": None,
            "verdict": "uncertain",
            "heuristic_margin": 0.0,
        },
    ]

    score = score_holdout_rows(rows)

    assert score["counts"]["human_labeled"] == 2
    assert score["metrics"]["classifier_coverage"] == 0.5
    assert score["metrics"]["fixed_denominator_accuracy"] == 0.5
    assert score["metrics"]["selective_accuracy"] == 1

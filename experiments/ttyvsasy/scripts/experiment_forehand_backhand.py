"""Run and visualize an experimental 2D forehand/backhand heuristic.

This adapts the algorithm supplied in ``F:/Downloads/forehand_backhand.py`` to
the commentary repository's typed upstream stages.  The output is deliberately
kept outside production schemas: a score magnitude is a heuristic margin, not a
calibrated probability or a verified forehand/backhand fact.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    read_upstream_stages,
)
from badminton_commentary.adapters.vision import PoseFrame, _iter_top_level_array


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_ROOT = EXPERIMENT_ROOT / "workspace" / "stages"
DEFAULT_VIDEO = EXPERIMENT_ROOT / "workspace" / "video" / "TTYvsASY.mp4"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "ttyvsasy"
    / "forehand_backhand"
)

KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
SKELETON_EDGES = (
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

Point = tuple[float, float]
Side = Literal["forehand", "backhand"]
Player = Literal["a", "b"]
CourtPosition = Literal["top", "bottom"]
OrientationPolicy = Literal[
    "vote",
    "court_prior",
    "invert_disagreement",
]


class ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShotResult(ExperimentModel):
    event_index: int = Field(ge=0)
    frame: int = Field(ge=0)
    local_frame: int = Field(ge=0)
    player: Player
    court_position: CourtPosition
    hand: Literal["left", "right"]
    stroke_type: str
    stroke_confidence: float = Field(ge=0, le=1)
    side: Side | None
    side_zh: str
    heuristic_margin: float = Field(ge=0)
    frames_used: int = Field(ge=0)
    detail: dict[str, object]


class ExperimentOutput(ExperimentModel):
    schema_version: Literal["experimental-forehand-backhand-v1"]
    segment_index: int = Field(ge=0)
    fps: float = Field(gt=0)
    source_start_frame: int = Field(ge=0)
    source_end_frame: int = Field(ge=0)
    player_mapping: dict[CourtPosition, Player]
    left_handed_players: list[Player]
    params: dict[str, object]
    shots: list[ShotResult]
    summary: dict[str, object]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_shots(self) -> ExperimentOutput:
        indexes = [item.event_index for item in self.shots]
        if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
            raise ValueError("shots must preserve unique source event_index order")
        if any(
            not self.source_start_frame <= item.frame <= self.source_end_frame
            for item in self.shots
        ):
            raise ValueError("shot frame is outside selected rally")
        return self


@dataclass(frozen=True)
class Config:
    min_score: float = 0.30
    racket_joint_min_score: float = 0.50
    min_racket_frames: int = 3
    window: int = 2
    max_search: int = 6
    time_sigma: float = 1.5
    soft: float = 0.35
    overhead_h: float = 0.15
    rth_elbow: float = 0.10
    rth_wrist: float = -0.05
    rth_score: float = 0.35
    min_margin: float = 0.08
    w_flat: tuple[float, float, float] = (0.00, 0.45, 0.55)
    w_over: tuple[float, float, float] = (0.35, 0.30, 0.35)
    w_shoulder: float = 1.00
    w_hip: float = 0.60
    w_face: float = 0.35
    w_prior: float = 0.25
    flip_full: float = 0.80
    orientation_policy: OrientationPolicy = "vote"

    def __post_init__(self) -> None:
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if not 0 <= self.racket_joint_min_score <= 1:
            raise ValueError("racket_joint_min_score must be between 0 and 1")
        if self.racket_joint_min_score < self.min_score:
            raise ValueError("racket_joint_min_score cannot be below min_score")
        if self.min_racket_frames < 1:
            raise ValueError("min_racket_frames must be positive")
        if self.window < 0 or self.max_search < 0:
            raise ValueError("frame windows must be non-negative")
        if self.time_sigma <= 0 or self.soft <= 0 or self.flip_full <= 0:
            raise ValueError("scale parameters must be positive")
        if self.min_margin < 0:
            raise ValueError("min_margin must be non-negative")
        if self.orientation_policy not in {
            "vote",
            "court_prior",
            "invert_disagreement",
        }:
            raise ValueError("unsupported orientation_policy")


@dataclass(frozen=True)
class BodyFrame:
    up: Point
    lateral_image_right: Point
    scale: float


@dataclass(frozen=True)
class Sample:
    frame_delta: int
    body: BodyFrame
    keypoints: list[tuple[float, float, float]]
    joint_confidence: float
    torso_confidence: float
    flip_vote: float
    flip_cues: dict[str, float]
    time_weight: float


def _sub(first: Point, second: Point) -> Point:
    return first[0] - second[0], first[1] - second[1]


def _dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _length(value: Point) -> float:
    return math.hypot(value[0], value[1])


def _unit(value: Point) -> Point | None:
    size = _length(value)
    if size < 1e-6:
        return None
    return value[0] / size, value[1] / size


def _midpoint(first: Point | None, second: Point | None) -> Point | None:
    if first is None:
        return second
    if second is None:
        return first
    return (first[0] + second[0]) / 2, (first[1] + second[1]) / 2


def _confidence(keypoints: list[tuple[float, float, float]], name: str) -> float:
    return float(keypoints[KP[name]][2])


def _point(
    keypoints: list[tuple[float, float, float]],
    name: str,
    threshold: float,
) -> Point | None:
    x, y, confidence = keypoints[KP[name]]
    return (float(x), float(y)) if confidence >= threshold else None


def _body_frame(
    keypoints: list[tuple[float, float, float]],
    bbox: tuple[float, float, float, float],
    config: Config,
) -> BodyFrame | None:
    left_shoulder = _point(keypoints, "left_shoulder", config.min_score)
    right_shoulder = _point(keypoints, "right_shoulder", config.min_score)
    shoulder_center = _midpoint(left_shoulder, right_shoulder)
    if shoulder_center is None:
        return None
    hip_center = _midpoint(
        _point(keypoints, "left_hip", config.min_score),
        _point(keypoints, "right_hip", config.min_score),
    )
    torso = 0.0
    up = None
    if hip_center is not None:
        torso_vector = _sub(shoulder_center, hip_center)
        torso = _length(torso_vector)
        up = _unit(torso_vector)
    if up is None:
        up = (0.0, -1.0)
    lateral = (-up[1], up[0])
    if lateral[0] < 0:
        lateral = (-lateral[0], -lateral[1])
    shoulder_width = (
        _length(_sub(right_shoulder, left_shoulder))
        if left_shoulder is not None and right_shoulder is not None
        else 0.0
    )
    bbox_height = float(bbox[3] - bbox[1])
    scale = max(torso, shoulder_width * 1.2, bbox_height * 0.30, 12.0)
    return BodyFrame(up=up, lateral_image_right=lateral, scale=scale)


def _flip_vote(
    body: BodyFrame,
    keypoints: list[tuple[float, float, float]],
    court_position: CourtPosition,
    config: Config,
) -> tuple[float, dict[str, float]]:
    total = 0.0
    cues: dict[str, float] = {}
    for cue, left_name, right_name, weight in (
        (
            "shoulder",
            "left_shoulder",
            "right_shoulder",
            config.w_shoulder,
        ),
        ("hip", "left_hip", "right_hip", config.w_hip),
    ):
        left = _point(keypoints, left_name, config.min_score)
        right = _point(keypoints, right_name, config.min_score)
        if left is None or right is None:
            continue
        span = _sub(right, left)
        span_length = _length(span)
        if span_length < 1e-6:
            continue
        value = _dot(span, body.lateral_image_right) / span_length
        cues[cue] = value
        total += weight * value

    face_confidence = sum(
        _confidence(keypoints, name)
        for name in ("nose", "left_eye", "right_eye")
    ) / 3
    face_value = max(-1.0, min(1.0, (face_confidence - 0.5) / 0.3))
    cues["face"] = -face_value
    total -= config.w_face * face_value
    prior = -1.0 if court_position == "top" else 1.0
    cues["prior"] = prior
    total += config.w_prior * prior
    cues["total"] = total
    return total, cues


def _sample(
    pose,
    *,
    hit_frame: int,
    court_position: CourtPosition,
    left_handed: bool,
    config: Config,
) -> Sample | None:
    body = _body_frame(pose.keypoints, pose.bbox, config)
    if body is None:
        return None
    racket_side = "left" if left_handed else "right"
    joint_names = (
        f"{racket_side}_shoulder",
        f"{racket_side}_elbow",
        f"{racket_side}_wrist",
    )
    if any(
        _point(pose.keypoints, name, config.min_score) is None
        for name in joint_names
    ):
        return None
    vote, cues = _flip_vote(body, pose.keypoints, court_position, config)
    torso_confidence = max(
        0.1,
        sum(
            _confidence(pose.keypoints, name)
            for name in (
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
            )
        )
        / 4,
    )
    delta = pose.frame - hit_frame
    return Sample(
        frame_delta=delta,
        body=body,
        keypoints=pose.keypoints,
        joint_confidence=min(
            _confidence(pose.keypoints, name) for name in joint_names
        ),
        torso_confidence=torso_confidence,
        flip_vote=vote,
        flip_cues=cues,
        time_weight=math.exp(
            -(delta * delta) / (2 * config.time_sigma * config.time_sigma)
        ),
    )


def _soft_sign(value: float, scale: float) -> float:
    return math.tanh(value / scale)


def frame_signals(
    sample: Sample,
    orientation_sign: float,
    *,
    left_handed: bool,
    config: Config,
) -> dict[str, float | bool]:
    body = sample.body
    handedness_sign = -1.0 if left_handed else 1.0
    racket_direction = (
        body.lateral_image_right[0] * orientation_sign * handedness_sign,
        body.lateral_image_right[1] * orientation_sign * handedness_sign,
    )
    side = "left" if left_handed else "right"
    shoulder = _point(sample.keypoints, f"{side}_shoulder", config.min_score)
    elbow = _point(sample.keypoints, f"{side}_elbow", config.min_score)
    wrist = _point(sample.keypoints, f"{side}_wrist", config.min_score)
    if shoulder is None or elbow is None or wrist is None:
        raise ValueError("sample is missing a required racket-arm joint")

    wrist_lateral = _dot(_sub(wrist, shoulder), racket_direction) / body.scale
    elbow_lateral = _dot(_sub(elbow, shoulder), racket_direction) / body.scale
    forearm_lateral = _dot(_sub(wrist, elbow), racket_direction) / body.scale
    wrist_height = _dot(_sub(wrist, shoulder), body.up) / body.scale
    overhead = wrist_height > config.overhead_h
    elbow_weight, wrist_weight, forearm_weight = (
        config.w_over if overhead else config.w_flat
    )
    score = (
        elbow_weight * _soft_sign(elbow_lateral, config.soft)
        + wrist_weight * _soft_sign(wrist_lateral, config.soft)
        + forearm_weight * _soft_sign(forearm_lateral, config.soft)
    )
    round_the_head = (
        overhead
        and elbow_lateral > config.rth_elbow
        and wrist_lateral < config.rth_wrist
    )
    if round_the_head:
        score = max(score, config.rth_score)
    return {
        "score": score,
        "wrist_lateral": wrist_lateral,
        "elbow_lateral": elbow_lateral,
        "forearm_lateral": forearm_lateral,
        "wrist_height": wrist_height,
        "overhead": overhead,
        "round_the_head": round_the_head,
    }


def _collect_samples(
    *,
    hit_frame: int,
    court_position: CourtPosition,
    pose_by_frame: dict[int, object],
    left_handed: bool,
    config: Config,
) -> list[Sample]:
    samples = []
    for delta in range(-config.window, config.window + 1):
        pose = pose_by_frame.get(hit_frame + delta)
        if pose is None:
            continue
        sample = _sample(
            pose,
            hit_frame=hit_frame,
            court_position=court_position,
            left_handed=left_handed,
            config=config,
        )
        if sample is not None:
            samples.append(sample)
    if samples or config.max_search <= config.window:
        return samples
    outside = sorted(
        range(-config.max_search, config.max_search + 1),
        key=lambda value: (abs(value), value),
    )
    for delta in outside:
        if abs(delta) <= config.window:
            continue
        pose = pose_by_frame.get(hit_frame + delta)
        if pose is None:
            continue
        sample = _sample(
            pose,
            hit_frame=hit_frame,
            court_position=court_position,
            left_handed=left_handed,
            config=config,
        )
        if sample is not None:
            return [sample]
    return []


def classify_hit(
    *,
    hit_frame: int,
    court_position: CourtPosition,
    pose_by_frame: dict[int, object],
    left_handed: bool,
    config: Config,
) -> tuple[Side | None, float, dict[str, object]]:
    samples = _collect_samples(
        hit_frame=hit_frame,
        court_position=court_position,
        pose_by_frame=pose_by_frame,
        left_handed=left_handed,
        config=config,
    )
    if not samples:
        return None, 0.0, {
            "frames_used": 0,
            "reason": "no usable racket-arm pose near hit frame",
        }
    review_frame_deltas = [item.frame_delta for item in samples]
    accepted_samples = [
        item
        for item in samples
        if item.joint_confidence >= config.racket_joint_min_score
    ]
    quality_detail: dict[str, object] = {
        "candidate_frames": len(samples),
        "accepted_racket_frames": len(accepted_samples),
        "required_racket_frames": config.min_racket_frames,
        "racket_joint_min_score": config.racket_joint_min_score,
        "candidate_joint_confidences": {
            str(item.frame_delta): round(item.joint_confidence, 4)
            for item in samples
        },
        "review_frame_deltas": review_frame_deltas,
        "frame_deltas": [item.frame_delta for item in accepted_samples],
        "score_semantics": "heuristic_margin_not_probability",
    }
    if len(accepted_samples) < config.min_racket_frames:
        return None, 0.0, {
            **quality_detail,
            "frames_used": len(accepted_samples),
            "reason": "insufficient high-confidence racket-arm frames",
        }
    samples = accepted_samples
    flip_weight = sum(
        item.time_weight * item.torso_confidence for item in samples
    )
    flip_total = sum(
        item.flip_vote * item.time_weight * item.torso_confidence
        for item in samples
    ) / max(flip_weight, 1e-6)
    voted_orientation_sign = 1.0 if flip_total >= 0 else -1.0
    flip_confidence = min(1.0, abs(flip_total) / config.flip_full)
    expected_sign = -1.0 if court_position == "top" else 1.0
    vote_disagreed_with_court_prior = voted_orientation_sign != expected_sign
    if config.orientation_policy == "court_prior":
        orientation_sign = expected_sign
    elif (
        config.orientation_policy == "invert_disagreement"
        and vote_disagreed_with_court_prior
    ):
        orientation_sign = -voted_orientation_sign
    else:
        orientation_sign = voted_orientation_sign
    scored = [
        (
            item,
            frame_signals(
                item,
                orientation_sign,
                left_handed=left_handed,
                config=config,
            ),
        )
        for item in samples
    ]
    weights = [
        item.time_weight * item.joint_confidence for item, _ in scored
    ]
    weight_sum = sum(weights)
    if weight_sum < 1e-6:
        return None, 0.0, {
            "frames_used": len(samples),
            "reason": "racket-arm pose confidence is too low",
        }
    aggregate = sum(
        float(signals["score"]) * weight
        for (_, signals), weight in zip(scored, weights, strict=True)
    ) / weight_sum
    aggregate *= 0.35 + 0.65 * flip_confidence
    hit_signals = next(
        (
            signals
            for item, signals in scored
            if item.frame_delta == 0
        ),
        scored[0][1],
    )
    detail: dict[str, object] = {
        **quality_detail,
        "frames_used": len(samples),
        "frame_deltas": [item.frame_delta for item in samples],
        "aggregate_score": round(aggregate, 4),
        "wrist_lateral": round(float(hit_signals["wrist_lateral"]), 3),
        "elbow_lateral": round(float(hit_signals["elbow_lateral"]), 3),
        "forearm_lateral": round(
            float(hit_signals["forearm_lateral"]), 3
        ),
        "wrist_height": round(float(hit_signals["wrist_height"]), 3),
        "overhead": bool(hit_signals["overhead"]),
        "round_the_head": any(
            bool(signals["round_the_head"]) for _, signals in scored
        ),
        "facing": "away" if orientation_sign > 0 else "toward",
        "body_flipped_from_court_prior": orientation_sign != expected_sign,
        "orientation_policy": config.orientation_policy,
        "voted_facing": (
            "away" if voted_orientation_sign > 0 else "toward"
        ),
        "vote_disagreed_with_court_prior": vote_disagreed_with_court_prior,
        "flip_confidence": round(flip_confidence, 3),
        "flip_total": round(flip_total, 3),
        "flip_cues": {
            key: round(value, 3)
            for key, value in next(
                (
                    item.flip_cues
                    for item in samples
                    if item.frame_delta == 0
                ),
                samples[0].flip_cues,
            ).items()
        },
        "frame_scores": {
            str(item.frame_delta): round(float(signals["score"]), 3)
            for item, signals in scored
        },
    }
    if len(samples) == 1 and abs(samples[0].frame_delta) > config.window:
        detail["search_offset"] = samples[0].frame_delta
    if abs(aggregate) < config.min_margin:
        detail["reason"] = "aggregate score is inside the unknown margin"
        return None, abs(aggregate), detail
    return ("forehand" if aggregate > 0 else "backhand"), abs(aggregate), detail


def analyze_segment(
    *,
    stage_root: Path,
    segment_index: int,
    mapping: CourtPositionToPlayer,
    left_handed_players: set[Player],
    config: Config,
) -> tuple[ExperimentOutput, dict[tuple[int, CourtPosition], object]]:
    paths = StagePaths.from_stage_root(stage_root)
    pose_path = paths.pose
    if pose_path is None:
        raise ValueError("selected stage root has no pose artifact")
    stages = read_upstream_stages(
        paths.model_copy(
            update={
                "pose": None,
                "court_detection": None,
                "shuttle_tracking": None,
            }
        ),
        segment_index=segment_index,
    )
    segment = stages.match_segmentation.segments[segment_index]
    pose_frames = []
    skipped_pose_records = 0
    for item in _iter_top_level_array(pose_path, "frames"):
        if not isinstance(item, dict) or item.get("segment_index") != segment_index:
            continue
        if item.get("keypoints") is None or item.get("bbox") is None:
            skipped_pose_records += 1
            continue
        pose_frames.append(PoseFrame.model_validate(item))
    pose_lookup = {
        (pose.frame, pose.player): pose for pose in pose_frames
    }
    event_frames = [item.frame for item in stages.event_detection.events]
    strokes = {
        item.event_index: item
        for item in stages.stroke_classification.strokes
        if item.segment_index == segment_index
    }
    shots = []
    for event_index, stroke in sorted(strokes.items()):
        if stroke.player is None:
            continue
        if event_index >= len(event_frames):
            raise ValueError(f"stroke event_index is out of range: {event_index}")
        frame = event_frames[event_index]
        if stroke.frame != frame:
            raise ValueError(f"stroke/event frame mismatch at {event_index}")
        if not segment.start_frame <= frame <= segment.end_frame:
            raise ValueError(f"stroke {event_index} is outside selected segment")
        identity = mapping.resolve(stroke.player)
        poses = {
            pose_frame: pose
            for (pose_frame, position), pose in pose_lookup.items()
            if position == stroke.player
        }
        left_handed = identity in left_handed_players
        side, margin, detail = classify_hit(
            hit_frame=frame,
            court_position=stroke.player,
            pose_by_frame=poses,
            left_handed=left_handed,
            config=config,
        )
        shots.append(
            ShotResult(
                event_index=event_index,
                frame=frame,
                local_frame=frame - segment.start_frame,
                player=identity,
                court_position=stroke.player,
                hand="left" if left_handed else "right",
                stroke_type=stroke.stroke_type,
                stroke_confidence=stroke.confidence,
                side=side,
                side_zh={"forehand": "正手", "backhand": "反手"}.get(
                    side,
                    "未知",
                ),
                heuristic_margin=round(margin, 4),
                frames_used=int(detail["frames_used"]),
                detail=detail,
            )
        )
    by_side = {
        name: sum((item.side or "unknown") == name for item in shots)
        for name in ("forehand", "backhand", "unknown")
    }
    by_player = {
        player: {
            name: sum(
                item.player == player and (item.side or "unknown") == name
                for item in shots
            )
            for name in ("forehand", "backhand", "unknown")
        }
        for player in ("a", "b")
    }
    result = ExperimentOutput(
        schema_version="experimental-forehand-backhand-v1",
        segment_index=segment_index,
        fps=stages.match_segmentation.fps,
        source_start_frame=segment.start_frame,
        source_end_frame=segment.end_frame,
        player_mapping={"top": mapping.top, "bottom": mapping.bottom},
        left_handed_players=sorted(left_handed_players),
        params=asdict(config),
        shots=shots,
        summary={
            "hits": len(shots),
            "by_side": by_side,
            "by_player": by_player,
            "median_heuristic_margin": (
                round(
                    sorted(item.heuristic_margin for item in shots)[
                        len(shots) // 2
                    ],
                    4,
                )
                if shots
                else 0.0
            ),
            "body_flipped_from_court_prior": sum(
                bool(item.detail.get("body_flipped_from_court_prior"))
                for item in shots
            ),
            "round_the_head": sum(
                bool(item.detail.get("round_the_head")) for item in shots
            ),
            "valid_pose_records": len(pose_frames),
            "skipped_missing_pose_records": skipped_pose_records,
        },
        limitations=[
            "Single-view 2D pose does not observe racket face, grip, handedness, or exact contact phase.",
            "The heuristic margin and flip confidence are not calibrated probabilities.",
            "Face-keypoint confidence is an unreliable proxy for body orientation.",
            "Top/bottom facing is only a weak prior and can be wrong during rotation.",
            "Thresholds and weights came from the supplied script and are not fitted or validated on SEG144 labels.",
            "The output must remain experimental until human frame-level review supplies ground truth.",
            f"Skipped {skipped_pose_records} pose records with null keypoints or bbox in this segment.",
        ],
    )
    return result, pose_lookup


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _ass_escape(value: object) -> str:
    return str(value).replace("{", "｛").replace("}", "｝").replace("\\", "／")


def _dialogue(
    *,
    layer: int,
    start: float,
    end: float,
    style: str,
    text: str,
) -> str:
    return (
        f"Dialogue: {layer},{_ass_time(start)},{_ass_time(end)},"
        f"{style},,0,0,0,,{text}"
    )


def _line_polygon(start: Point, end: Point, thickness: float = 5.0) -> str | None:
    length = _length(_sub(end, start))
    if length < 1e-6:
        return None
    px = -(end[1] - start[1]) * thickness / (2 * length)
    py = (end[0] - start[0]) * thickness / (2 * length)
    points = (
        (start[0] + px, start[1] + py),
        (end[0] + px, end[1] + py),
        (end[0] - px, end[1] - py),
        (start[0] - px, start[1] - py),
    )
    return "m " + " l ".join(f"{round(x)} {round(y)}" for x, y in points)


def _pose_path(pose, *, racket_side: str | None = None) -> str:
    paths = []
    edges = SKELETON_EDGES
    if racket_side is not None:
        edges = (
            (f"{racket_side}_shoulder", f"{racket_side}_elbow"),
            (f"{racket_side}_elbow", f"{racket_side}_wrist"),
        )
    for start_name, end_name in edges:
        start = pose.keypoints[KP[start_name]]
        end = pose.keypoints[KP[end_name]]
        if min(start[2], end[2]) < 0.3:
            continue
        polygon = _line_polygon((start[0], start[1]), (end[0], end[1]))
        if polygon is not None:
            paths.append(polygon)
    return " ".join(paths)


def render_ass(
    result: ExperimentOutput,
    pose_lookup: dict[tuple[int, CourtPosition], object],
) -> str:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Forehand,Microsoft JhengHei,42,&H004ADE80,&H004ADE80,&H00101010,&HA0000000,-1,0,0,0,100,100,0,0,3,2,0,8,60,60,45,1
Style: Backhand,Microsoft JhengHei,42,&H00FB71D6,&H00FB71D6,&H00101010,&HA0000000,-1,0,0,0,100,100,0,0,3,2,0,8,60,60,45,1
Style: Unknown,Microsoft JhengHei,42,&H00C4B5FD,&H00C4B5FD,&H00101010,&HA0000000,-1,0,0,0,100,100,0,0,3,2,0,8,60,60,45,1
Style: Info,Microsoft JhengHei,27,&H00FFFFFF,&H00FFFFFF,&H00101010,&HA0000000,0,0,0,0,100,100,0,0,3,2,0,1,35,35,35,1
Style: Warning,Microsoft JhengHei,21,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,-1,0,0,0,100,100,0,0,3,1,0,7,28,28,25,1
Style: Pose,Arial,10,&H0080DE4A,&H0080DE4A,&H00052E16,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Arm,Arial,10,&H0015CCFA,&H0015CCFA,&H00101010,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    fps = result.fps
    duration = (result.source_end_frame - result.source_start_frame + 1) / fps
    lines = [
        _dialogue(
            layer=7,
            start=0,
            end=duration,
            style="Warning",
            text="EXPERIMENTAL 2D HEURISTIC · NOT GROUND TRUTH",
        )
    ]
    for shot in result.shots:
        local_time = shot.local_frame / fps
        start = max(0.0, local_time - 0.18)
        end = min(duration, local_time + 0.42)
        style = {
            "forehand": "Forehand",
            "backhand": "Backhand",
        }.get(shot.side, "Unknown")
        lines.append(
            _dialogue(
                layer=8,
                start=start,
                end=end,
                style=style,
                text=_ass_escape(
                    f"#{shot.event_index} {shot.side_zh} / "
                    f"{shot.side or 'unknown'}"
                ),
            )
        )
        facing = shot.detail.get("facing", "unknown")
        lines.append(
            _dialogue(
                layer=8,
                start=start,
                end=end,
                style="Info",
                text=r"\N".join(
                    _ass_escape(item)
                    for item in (
                        f"player {shot.player} ({shot.court_position}) · {shot.hand}-handed",
                        f"stroke {shot.stroke_type} · classifier confidence {shot.stroke_confidence:.2f}",
                        f"heuristic margin {shot.heuristic_margin:.3f} · facing {facing}",
                        f"frame {shot.frame} · local {shot.local_frame} · pose frames {shot.frames_used}",
                    )
                ),
            )
        )
        racket_side = "left" if shot.hand == "left" else "right"
        for delta in shot.detail.get(
            "review_frame_deltas",
            shot.detail.get("frame_deltas", []),
        ):
            frame = shot.frame + int(delta)
            pose = pose_lookup.get((frame, shot.court_position))
            if pose is None:
                continue
            pose_start = max(0.0, (frame - result.source_start_frame) / fps)
            pose_end = min(duration, pose_start + 1 / fps)
            skeleton = _pose_path(pose)
            arm = _pose_path(pose, racket_side=racket_side)
            if skeleton:
                lines.append(
                    _dialogue(
                        layer=3,
                        start=pose_start,
                        end=pose_end,
                        style="Pose",
                        text=r"{\p1}" + skeleton,
                    )
                )
            if arm:
                lines.append(
                    _dialogue(
                        layer=4,
                        start=pose_start,
                        end=pose_end,
                        style="Arm",
                        text=r"{\p1}" + arm,
                    )
                )
    return header + "\n".join(lines) + "\n"


def _filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def _video_info(video: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)["streams"][0]


def render_video(
    *,
    video: Path,
    subtitles: Path,
    output: Path,
    source_start_frame: int,
    source_end_frame: int,
    fps: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    expected_frames = source_end_frame - source_start_frame + 1
    info = _video_info(video)
    input_frames = int(info["nb_frames"])
    input_args = ["-i", str(video.resolve())]
    if input_frames != expected_frames:
        input_args = [
            "-ss",
            f"{source_start_frame / fps:.9f}",
            "-i",
            str(video.resolve()),
        ]
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            *input_args,
            "-vf",
            f"ass=filename='{_filter_path(subtitles)}'",
            "-frames:v",
            str(expected_frames),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(output.resolve()),
        ],
        check=True,
    )
    rendered_info = _video_info(output)
    if int(rendered_info["nb_frames"]) != expected_frames:
        raise ValueError(
            f"rendered clip has {rendered_info['nb_frames']} frames; "
            f"expected {expected_frames}"
        )


def render_viewer(result: ExperimentOutput, video_name: str) -> str:
    events_json = json.dumps(
        [
            {
                "event_index": item.event_index,
                "local_frame": item.local_frame,
                "player": item.player,
                "stroke_type": item.stroke_type,
                "side": item.side,
                "side_zh": item.side_zh,
                "margin": item.heuristic_margin,
            }
            for item in result.shots
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    rows = "".join(
        "<tr data-index='{index}'><td><button class='jump' data-index='{index}'>"
        "#{event}</button></td><td>{frame}</td><td>{player}</td><td>{stroke}</td>"
        "<td>{side}</td><td>{margin:.3f}</td><td class='verdict'>—</td></tr>".format(
            index=index,
            event=item.event_index,
            frame=item.local_frame,
            player=html.escape(item.player),
            stroke=html.escape(item.stroke_type),
            side=html.escape(item.side_zh),
            margin=item.heuristic_margin,
        )
        for index, item in enumerate(result.shots)
    )
    segment_label = f"seg{result.segment_index:04d}"
    review_schema = "experimental-forehand-backhand-human-review-v2"
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>SEG{result.segment_index} 正反手逐幀檢查</title>
<style>
body{{font-family:Segoe UI,Microsoft JhengHei,sans-serif;background:#0f172a;color:#e2e8f0;margin:20px}}
main{{display:grid;grid-template-columns:minmax(640px,2fr) minmax(420px,1fr);gap:20px}}
video{{width:100%;background:#000}} .panel{{background:#1e293b;padding:14px;border-radius:10px}}
kbd{{background:#334155;border:1px solid #64748b;border-radius:4px;padding:2px 6px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:6px;border-bottom:1px solid #334155;text-align:left}}
tr.active{{background:#334155}} button{{cursor:pointer}} .good{{color:#4ade80}} .bad{{color:#fb7185}} .unsure{{color:#c4b5fd}}
</style></head><body>
<h1>SEG{result.segment_index} 正反手 heuristic 人工檢查</h1>
<p>輸出是 2D heuristic，不是 ground truth。先在擊球幀前後逐幀看持拍臂，再標人工 verdict。</p>
<main><section><video id="video" controls preload="auto" src="{html.escape(video_name)}"></video>
<div class="panel"><strong id="status">載入中</strong><br>
<kbd>←</kbd>/<kbd>→</kbd> 前後 1 幀　<kbd>Shift</kbd>+方向鍵 10 幀　
<kbd>P</kbd>/<kbd>N</kbd> 上／下一擊　<kbd>Space</kbd> 播放／暫停<br>
<kbd>C</kbd> 正確　<kbd>X</kbd> 錯誤　<kbd>U</kbd> 無法判斷　<kbd>E</kbd> 匯出 review JSON</div></section>
<section class="panel"><table><thead><tr><th>Event</th><th>Local frame</th><th>Player</th><th>Stroke</th><th>判定</th><th>Margin</th><th>人工</th></tr></thead><tbody>{rows}</tbody></table></section></main>
<script>
const fps={result.fps}; const events={events_json}; const video=document.getElementById('video');
const status=document.getElementById('status'); const rows=[...document.querySelectorAll('tbody tr')];
const storageKey='{segment_label}-forehand-backhand-review-v2'; let reviews=JSON.parse(localStorage.getItem(storageKey)||'{{}}'); let selected=0;
function frame(){{return Math.round(video.currentTime*fps)}}
function seekFrame(value){{video.pause(); video.currentTime=Math.max(0,value)/fps}}
function choose(index){{selected=Math.max(0,Math.min(events.length-1,index)); seekFrame(events[selected].local_frame); paint()}}
function paint(){{rows.forEach((row,i)=>{{row.classList.toggle('active',i===selected); const v=reviews[events[i].event_index]; const cell=row.querySelector('.verdict'); cell.textContent=v==='correct'?'正確':v==='incorrect'?'錯誤':v==='uncertain'?'無法判斷':'—'; cell.className='verdict '+(v==='correct'?'good':v==='incorrect'?'bad':v==='uncertain'?'unsure':'')}}); const e=events[selected]; status.textContent=`frame ${{frame()}} · event #${{e.event_index}} hit frame ${{e.local_frame}} · ${{e.player}} ${{e.stroke_type}} · ${{e.side_zh}} margin ${{e.margin.toFixed(3)}}`}}
function review(value){{reviews[events[selected].event_index]=value; localStorage.setItem(storageKey,JSON.stringify(reviews)); paint()}}
function exportReview(){{const payload={{schema_version:'{review_schema}',segment_index:{result.segment_index},fps,reviewed_at:new Date().toISOString(),reviews:events.map(e=>({{...e,verdict:reviews[e.event_index]||'unreviewed'}}))}}; const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}})); a.download='{segment_label}_forehand_backhand_human_review.json'; a.click(); URL.revokeObjectURL(a.href)}}
document.querySelectorAll('.jump').forEach(button=>button.onclick=()=>choose(Number(button.dataset.index)));
document.addEventListener('keydown',event=>{{if(event.target.tagName==='INPUT')return; const key=event.key.toLowerCase(); if(key==='arrowleft'){{event.preventDefault();seekFrame(frame()-(event.shiftKey?10:1))}} else if(key==='arrowright'){{event.preventDefault();seekFrame(frame()+(event.shiftKey?10:1))}} else if(key==='p')choose(selected-1); else if(key==='n')choose(selected+1); else if(key==='c')review('correct'); else if(key==='x')review('incorrect'); else if(key==='u')review('uncertain'); else if(key==='e')exportReview(); else if(event.code==='Space'){{event.preventDefault();video.paused?video.play():video.pause()}}}});
video.addEventListener('timeupdate',paint); video.addEventListener('seeked',paint); choose(0);
</script></body></html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-index", type=int, default=144)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-player", choices=("a", "b"), default="b")
    parser.add_argument("--bottom-player", choices=("a", "b"), default="a")
    parser.add_argument(
        "--left-handed-player",
        choices=("a", "b"),
        action="append",
        default=[],
    )
    parser.add_argument("--window", type=int, default=Config.window)
    parser.add_argument("--max-search", type=int, default=Config.max_search)
    parser.add_argument("--min-score", type=float, default=Config.min_score)
    parser.add_argument(
        "--racket-joint-min-score",
        type=float,
        default=Config.racket_joint_min_score,
    )
    parser.add_argument(
        "--min-racket-frames",
        type=int,
        default=Config.min_racket_frames,
    )
    parser.add_argument("--min-margin", type=float, default=Config.min_margin)
    parser.add_argument("--face-weight", type=float, default=Config.w_face)
    parser.add_argument(
        "--orientation-policy",
        choices=("vote", "court_prior", "invert_disagreement"),
        default=Config.orientation_policy,
    )
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config(
        min_score=args.min_score,
        racket_joint_min_score=args.racket_joint_min_score,
        min_racket_frames=args.min_racket_frames,
        window=args.window,
        max_search=args.max_search,
        min_margin=args.min_margin,
        w_face=args.face_weight,
        orientation_policy=args.orientation_policy,
    )
    result, pose_lookup = analyze_segment(
        stage_root=args.stage_root,
        segment_index=args.segment_index,
        mapping=CourtPositionToPlayer(
            top=args.top_player,
            bottom=args.bottom_player,
        ),
        left_handed_players=set(args.left_handed_player),
        config=config,
    )
    output_dir = args.output or (
        DEFAULT_OUTPUT_ROOT / f"seg{args.segment_index:04d}_confidence_gate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_label = f"seg{args.segment_index:04d}"
    result_path = output_dir / "forehand_backhand_results.json"
    subtitle_path = output_dir / "forehand_backhand_overlay.ass"
    video_path = output_dir / f"{segment_label}_forehand_backhand_overlay.mp4"
    viewer_path = output_dir / "frame_review.html"
    result_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    subtitle_path.write_text(
        render_ass(result, pose_lookup),
        encoding="utf-8-sig",
    )
    if not args.skip_video:
        render_video(
            video=args.video,
            subtitles=subtitle_path,
            output=video_path,
            source_start_frame=result.source_start_frame,
            source_end_frame=result.source_end_frame,
            fps=result.fps,
        )
    viewer_path.write_text(
        render_viewer(result, video_path.name),
        encoding="utf-8",
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"results: {result_path.resolve()}")
    print(f"subtitles: {subtitle_path.resolve()}")
    print(f"video: {video_path.resolve()}")
    print(f"viewer: {viewer_path.resolve()}")


if __name__ == "__main__":
    main()

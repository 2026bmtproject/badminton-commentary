"""Deterministic 2D pose geometry for hit-centred pose windows.

This module intentionally produces measurements, not posture semantics.  Raw
keypoint confidences are clamped only while calculating geometry; callers keep
their source pose records unchanged.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


MIN_KP_CONF = 0.35
MIN_TORSO_LENGTH = 1e-6
POSE_KEYFRAME_DELTAS = (-8, -4, 0, 4, 8, 10)


class PoseGeometryRecord(Protocol):
    frame: int
    frame_delta: int
    keypoints: Mapping[str, Sequence[float]]


class GeometryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StepWidthFeatures(GeometryModel):
    pre_median_ratio: float | None
    at_hit_ratio: float | None
    post_median_ratio: float | None
    max_ratio: float | None
    max_frame_delta: int | None
    hit_feature_frame_delta: int | None
    confidence: float = Field(ge=0, le=1)


class KneeFlexionFeatures(GeometryModel):
    left_angle_deg_at_hit: float | None
    right_angle_deg_at_hit: float | None
    left_min_angle_deg: float | None
    right_min_angle_deg: float | None
    min_angle_deg: float | None
    min_side: Literal["left", "right"] | None
    min_frame_delta: int | None
    confidence: float = Field(ge=0, le=1)


class BodyHeightFeatures(GeometryModel):
    pre_median_ratio: float | None
    at_hit_ratio: float | None
    post_median_ratio: float | None
    min_ratio: float | None
    drop_from_pre_to_hit_ratio: float | None
    max_drop_from_pre_ratio: float | None
    confidence: float = Field(ge=0, le=1)


class TorsoLeanFeatures(GeometryModel):
    angle_deg_at_hit: float | None
    direction_at_hit: Literal["left", "right", "none", "unknown"]
    max_angle_deg: float | None
    max_frame_delta: int | None
    confidence: float = Field(ge=0, le=1)


class WristReachFeatures(GeometryModel):
    left_ratio_at_hit: float | None
    right_ratio_at_hit: float | None
    max_ratio: float | None
    max_side: Literal["left", "right"] | None
    max_frame_delta: int | None
    confidence: float = Field(ge=0, le=1)


class BodyDisplacementFeatures(GeometryModel):
    pre_to_hit_ratio: float | None
    hit_to_post_ratio: float | None
    window_total_ratio: float | None
    max_single_frame_ratio: float | None
    confidence: float = Field(ge=0, le=1)


class PoseGeometryFeatures(GeometryModel):
    source_start_frame: int = Field(ge=0)
    source_end_frame: int = Field(ge=0)
    hit_frame: int = Field(ge=0)
    feature_confidence: float = Field(ge=0, le=1)
    step_width: StepWidthFeatures
    knee_flexion: KneeFlexionFeatures
    body_height: BodyHeightFeatures
    torso_lean: TorsoLeanFeatures
    wrist_reach: WristReachFeatures
    body_displacement: BodyDisplacementFeatures
    limitations: list[str]


Point = tuple[float, float]


@dataclass(frozen=True)
class _Measurement:
    frame: int
    delta: int
    value: float
    confidence: float


@dataclass(frozen=True)
class _SideMeasurement:
    frame: int
    delta: int
    side: Literal["left", "right"]
    value: float
    confidence: float


@dataclass(frozen=True)
class _CenterMeasurement:
    frame: int
    delta: int
    point: Point
    torso_length: float
    confidence: float


def _clamp_confidence(value: float) -> float:
    return min(1.0, max(0.0, value))


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _point(
    record: PoseGeometryRecord,
    name: str,
) -> tuple[float, float, float] | None:
    raw = record.keypoints.get(name)
    if raw is None or len(raw) != 3:
        return None
    x, y, raw_confidence = (float(value) for value in raw)
    if not all(math.isfinite(value) for value in (x, y, raw_confidence)):
        return None
    confidence = _clamp_confidence(raw_confidence)
    if confidence < MIN_KP_CONF:
        return None
    return x, y, confidence


def _midpoint(
    left: tuple[float, float, float] | None,
    right: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if left is None or right is None:
        return None
    return (
        (left[0] + right[0]) / 2,
        (left[1] + right[1]) / 2,
        _mean((left[2], right[2])),
    )


def _centers(
    record: PoseGeometryRecord,
) -> tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    shoulder = _midpoint(
        _point(record, "left_shoulder"),
        _point(record, "right_shoulder"),
    )
    hip = _midpoint(
        _point(record, "left_hip"),
        _point(record, "right_hip"),
    )
    ankle = _midpoint(
        _point(record, "left_ankle"),
        _point(record, "right_ankle"),
    )
    return shoulder, hip, ankle


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _torso(
    record: PoseGeometryRecord,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
] | None:
    shoulder, hip, _ = _centers(record)
    if shoulder is None or hip is None:
        return None
    length = _distance(shoulder, hip)
    if not math.isfinite(length) or length <= MIN_TORSO_LENGTH:
        return None
    return shoulder, hip, length


def _hit_measurement(
    values: Sequence[_Measurement],
) -> _Measurement | None:
    candidates = [item for item in values if -1 <= item.delta <= 1]
    return min(candidates, key=lambda item: (abs(item.delta), item.delta)) if candidates else None


def _hit_side_measurement(
    values: Sequence[_SideMeasurement],
    side: Literal["left", "right"],
) -> _SideMeasurement | None:
    candidates = [
        item for item in values if item.side == side and -1 <= item.delta <= 1
    ]
    return min(candidates, key=lambda item: (abs(item.delta), item.delta)) if candidates else None


def _region_values(
    values: Sequence[_Measurement],
    start: int,
    end: int,
) -> list[float]:
    return [item.value for item in values if start <= item.delta <= end]


def _angle(vertex: Point, first: Point, second: Point) -> float | None:
    vector_a = (first[0] - vertex[0], first[1] - vertex[1])
    vector_b = (second[0] - vertex[0], second[1] - vertex[1])
    length_a = math.hypot(*vector_a)
    length_b = math.hypot(*vector_b)
    if length_a <= MIN_TORSO_LENGTH or length_b <= MIN_TORSO_LENGTH:
        return None
    cosine = (
        vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
    ) / (length_a * length_b)
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def compute_step_width_features(
    records: Sequence[PoseGeometryRecord],
) -> StepWidthFeatures:
    values: list[_Measurement] = []
    for record in records:
        torso = _torso(record)
        left = _point(record, "left_ankle")
        right = _point(record, "right_ankle")
        if torso is None or left is None or right is None:
            continue
        shoulder, hip, torso_length = torso
        values.append(
            _Measurement(
                frame=record.frame,
                delta=record.frame_delta,
                value=_distance(left, right) / torso_length,
                confidence=_mean(
                    (shoulder[2], hip[2], left[2], right[2])
                ),
            )
        )
    hit = _hit_measurement(values)
    maximum = max(values, key=lambda item: item.value) if values else None
    return StepWidthFeatures(
        pre_median_ratio=_median(_region_values(values, -8, -2)),
        at_hit_ratio=hit.value if hit else None,
        post_median_ratio=_median(_region_values(values, 2, 10)),
        max_ratio=maximum.value if maximum else None,
        max_frame_delta=maximum.delta if maximum else None,
        hit_feature_frame_delta=hit.delta if hit else None,
        confidence=_mean([item.confidence for item in values]),
    )


def compute_knee_features(
    records: Sequence[PoseGeometryRecord],
) -> KneeFlexionFeatures:
    values: list[_SideMeasurement] = []
    for record in records:
        for side in ("left", "right"):
            hip = _point(record, f"{side}_hip")
            knee = _point(record, f"{side}_knee")
            ankle = _point(record, f"{side}_ankle")
            if hip is None or knee is None or ankle is None:
                continue
            angle = _angle(knee[:2], hip[:2], ankle[:2])
            if angle is None:
                continue
            values.append(
                _SideMeasurement(
                    frame=record.frame,
                    delta=record.frame_delta,
                    side=side,
                    value=angle,
                    confidence=_mean((hip[2], knee[2], ankle[2])),
                )
            )
    left_hit = _hit_side_measurement(values, "left")
    right_hit = _hit_side_measurement(values, "right")
    left_values = [item for item in values if item.side == "left"]
    right_values = [item for item in values if item.side == "right"]
    left_min = min(left_values, key=lambda item: item.value) if left_values else None
    right_min = min(right_values, key=lambda item: item.value) if right_values else None
    overall = min(values, key=lambda item: item.value) if values else None
    return KneeFlexionFeatures(
        left_angle_deg_at_hit=left_hit.value if left_hit else None,
        right_angle_deg_at_hit=right_hit.value if right_hit else None,
        left_min_angle_deg=left_min.value if left_min else None,
        right_min_angle_deg=right_min.value if right_min else None,
        min_angle_deg=overall.value if overall else None,
        min_side=overall.side if overall else None,
        min_frame_delta=overall.delta if overall else None,
        confidence=_mean([item.confidence for item in values]),
    )


def compute_body_height_features(
    records: Sequence[PoseGeometryRecord],
) -> BodyHeightFeatures:
    values: list[_Measurement] = []
    for record in records:
        torso = _torso(record)
        _, _, ankle = _centers(record)
        if torso is None or ankle is None:
            continue
        shoulder, hip, torso_length = torso
        values.append(
            _Measurement(
                frame=record.frame,
                delta=record.frame_delta,
                value=abs(ankle[1] - hip[1]) / torso_length,
                confidence=_mean((shoulder[2], hip[2], ankle[2])),
            )
        )
    pre = _median(_region_values(values, -8, -2))
    hit = _hit_measurement(values)
    minimum = min(values, key=lambda item: item.value) if values else None
    return BodyHeightFeatures(
        pre_median_ratio=pre,
        at_hit_ratio=hit.value if hit else None,
        post_median_ratio=_median(_region_values(values, 2, 10)),
        min_ratio=minimum.value if minimum else None,
        drop_from_pre_to_hit_ratio=(
            pre - hit.value if pre is not None and hit is not None else None
        ),
        max_drop_from_pre_ratio=(
            pre - minimum.value
            if pre is not None and minimum is not None
            else None
        ),
        confidence=_mean([item.confidence for item in values]),
    )


def compute_torso_lean_features(
    records: Sequence[PoseGeometryRecord],
) -> TorsoLeanFeatures:
    values: list[_Measurement] = []
    directions: dict[tuple[int, int], Literal["left", "right", "none"]] = {}
    for record in records:
        torso = _torso(record)
        if torso is None:
            continue
        shoulder, hip, torso_length = torso
        dx = shoulder[0] - hip[0]
        dy = shoulder[1] - hip[1]
        cosine = (-dy) / torso_length
        angle = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
        direction: Literal["left", "right", "none"]
        if abs(dx) <= MIN_TORSO_LENGTH:
            direction = "none"
        else:
            direction = "left" if dx < 0 else "right"
        values.append(
            _Measurement(
                frame=record.frame,
                delta=record.frame_delta,
                value=angle,
                confidence=_mean((shoulder[2], hip[2])),
            )
        )
        directions[(record.frame, record.frame_delta)] = direction
    hit = _hit_measurement(values)
    maximum = max(values, key=lambda item: item.value) if values else None
    return TorsoLeanFeatures(
        angle_deg_at_hit=hit.value if hit else None,
        direction_at_hit=(
            directions[(hit.frame, hit.delta)] if hit is not None else "unknown"
        ),
        max_angle_deg=maximum.value if maximum else None,
        max_frame_delta=maximum.delta if maximum else None,
        confidence=_mean([item.confidence for item in values]),
    )


def compute_wrist_reach_features(
    records: Sequence[PoseGeometryRecord],
) -> WristReachFeatures:
    values: list[_SideMeasurement] = []
    for record in records:
        torso = _torso(record)
        if torso is None:
            continue
        shoulder, hip, torso_length = torso
        for side in ("left", "right"):
            wrist = _point(record, f"{side}_wrist")
            if wrist is None:
                continue
            values.append(
                _SideMeasurement(
                    frame=record.frame,
                    delta=record.frame_delta,
                    side=side,
                    value=_distance(wrist, shoulder) / torso_length,
                    confidence=_mean((shoulder[2], hip[2], wrist[2])),
                )
            )
    left_hit = _hit_side_measurement(values, "left")
    right_hit = _hit_side_measurement(values, "right")
    maximum = max(values, key=lambda item: item.value) if values else None
    return WristReachFeatures(
        left_ratio_at_hit=left_hit.value if left_hit else None,
        right_ratio_at_hit=right_hit.value if right_hit else None,
        max_ratio=maximum.value if maximum else None,
        max_side=maximum.side if maximum else None,
        max_frame_delta=maximum.delta if maximum else None,
        confidence=_mean([item.confidence for item in values]),
    )


def _median_point(values: Sequence[_CenterMeasurement]) -> Point | None:
    if not values:
        return None
    return (
        statistics.median(item.point[0] for item in values),
        statistics.median(item.point[1] for item in values),
    )


def _hit_center(
    values: Sequence[_CenterMeasurement],
) -> _CenterMeasurement | None:
    candidates = [item for item in values if -1 <= item.delta <= 1]
    return min(candidates, key=lambda item: (abs(item.delta), item.delta)) if candidates else None


def compute_body_displacement_features(
    records: Sequence[PoseGeometryRecord],
) -> BodyDisplacementFeatures:
    values: list[_CenterMeasurement] = []
    for record in records:
        torso = _torso(record)
        if torso is None:
            continue
        shoulder, hip, torso_length = torso
        values.append(
            _CenterMeasurement(
                frame=record.frame,
                delta=record.frame_delta,
                point=hip[:2],
                torso_length=torso_length,
                confidence=_mean((shoulder[2], hip[2])),
            )
        )
    values.sort(key=lambda item: item.delta)
    torso_scale = _median([item.torso_length for item in values])
    if torso_scale is None or torso_scale <= MIN_TORSO_LENGTH:
        return BodyDisplacementFeatures(
            pre_to_hit_ratio=None,
            hit_to_post_ratio=None,
            window_total_ratio=None,
            max_single_frame_ratio=None,
            confidence=0.0,
        )

    pre_values = [item for item in values if -8 <= item.delta <= -2][:3]
    post_values = [item for item in values if 2 <= item.delta <= 10][-3:]
    first_values = values[:3]
    last_values = values[-3:]
    pre = _median_point(pre_values)
    post = _median_point(post_values)
    first = _median_point(first_values)
    last = _median_point(last_values)
    hit = _hit_center(values)
    consecutive = [
        _distance(first_item.point, second_item.point) / torso_scale
        for first_item, second_item in zip(values, values[1:], strict=False)
        if second_item.frame - first_item.frame == 1
    ]
    return BodyDisplacementFeatures(
        pre_to_hit_ratio=(
            _distance(pre, hit.point) / torso_scale
            if pre is not None and hit is not None
            else None
        ),
        hit_to_post_ratio=(
            _distance(hit.point, post) / torso_scale
            if hit is not None and post is not None
            else None
        ),
        window_total_ratio=(
            _distance(first, last) / torso_scale
            if first is not None and last is not None
            else None
        ),
        max_single_frame_ratio=max(consecutive) if consecutive else None,
        confidence=_mean([item.confidence for item in values]),
    )


def compute_pose_geometry(
    records: Sequence[PoseGeometryRecord],
    *,
    hit_frame: int,
    source_start_frame: int | None = None,
    source_end_frame: int | None = None,
) -> PoseGeometryFeatures:
    """Compute deterministic numeric features without mutating ``records``."""

    ordered = sorted(records, key=lambda item: item.frame_delta)
    start = source_start_frame if source_start_frame is not None else hit_frame - 8
    end = source_end_frame if source_end_frame is not None else hit_frame + 10
    if any(not -8 <= item.frame_delta <= 10 for item in ordered):
        raise ValueError("pose geometry records must stay within -8..+10")

    step_width = compute_step_width_features(ordered)
    knee_flexion = compute_knee_features(ordered)
    body_height = compute_body_height_features(ordered)
    torso_lean = compute_torso_lean_features(ordered)
    wrist_reach = compute_wrist_reach_features(ordered)
    body_displacement = compute_body_displacement_features(ordered)
    feature_confidences = [
        step_width.confidence,
        knee_flexion.confidence,
        body_height.confidence,
        torso_lean.confidence,
        wrist_reach.confidence,
        body_displacement.confidence,
    ]
    limitations = []
    for name, feature in (
        ("step_width", step_width),
        ("knee_flexion", knee_flexion),
        ("body_height", body_height),
        ("torso_lean", torso_lean),
        ("wrist_reach", wrist_reach),
        ("body_displacement", body_displacement),
    ):
        if feature.confidence == 0:
            limitations.append(f"insufficient_valid_{name}_frames")
    if not any(item.frame_delta == 0 for item in ordered):
        limitations.append("exact_hit_pose_frame_missing")
    return PoseGeometryFeatures(
        source_start_frame=start,
        source_end_frame=end,
        hit_frame=hit_frame,
        feature_confidence=_mean(feature_confidences),
        step_width=step_width,
        knee_flexion=knee_flexion,
        body_height=body_height,
        torso_lean=torso_lean,
        wrist_reach=wrist_reach,
        body_displacement=body_displacement,
        limitations=limitations,
    )

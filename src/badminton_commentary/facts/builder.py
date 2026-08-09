from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass

from badminton_commentary.adapters.upstream import (
    CourtPositionToPlayer,
    UpstreamStageData,
    build_rally_fact_from_stages,
)
from badminton_commentary.adapters.vision import (
    CourtCalibration,
    CourtPosition,
    PoseFrame,
    ShuttlePoint,
)
from badminton_commentary.schemas import Player, RallyFactEvent

from .schemas import (
    CompactCourtPositionFact,
    CompactPoseFact,
    CompactRallyFacts,
    CompactShuttlePathFact,
    CompactStrokeFact,
    ImageDirection,
)


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16


@dataclass(frozen=True)
class CompactFactConfig:
    pose_max_frame_delta: int = 2
    pose_keypoint_confidence: float = 0.5
    shuttle_window_frames: int = 6
    shuttle_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.pose_max_frame_delta < 0:
            raise ValueError("pose_max_frame_delta must be non-negative")
        if not 0 <= self.pose_keypoint_confidence <= 1:
            raise ValueError("pose_keypoint_confidence must be between 0 and 1")
        if self.shuttle_window_frames < 1:
            raise ValueError("shuttle_window_frames must be positive")
        if not 0 <= self.shuttle_confidence <= 1:
            raise ValueError("shuttle_confidence must be between 0 and 1")


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _midpoint(first: tuple[float, float], second: tuple[float, float]):
    return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)


def _usable_point(
    pose: PoseFrame,
    index: int,
    threshold: float,
) -> tuple[float, float] | None:
    x, y, confidence = pose.keypoints[index]
    return (x, y) if confidence >= threshold else None


def _pose_fact(
    *,
    segment_index: int,
    event: RallyFactEvent,
    pose: PoseFrame,
    threshold: float,
) -> CompactPoseFact:
    raw_confidences = [item[2] for item in pose.keypoints]
    confidences = [min(item, 1.0) for item in raw_confidences]
    usable_count = sum(item >= threshold for item in confidences)
    mean_confidence = sum(confidences) / len(confidences)
    quality = (
        "reliable"
        if usable_count >= 12 and mean_confidence >= 0.6
        else "cautious" if usable_count >= 6 else "unavailable"
    )

    left_shoulder = _usable_point(pose, COCO_LEFT_SHOULDER, threshold)
    right_shoulder = _usable_point(pose, COCO_RIGHT_SHOULDER, threshold)
    left_wrist = _usable_point(pose, COCO_LEFT_WRIST, threshold)
    right_wrist = _usable_point(pose, COCO_RIGHT_WRIST, threshold)
    left_hip = _usable_point(pose, COCO_LEFT_HIP, threshold)
    right_hip = _usable_point(pose, COCO_RIGHT_HIP, threshold)
    left_ankle = _usable_point(pose, COCO_LEFT_ANKLE, threshold)
    right_ankle = _usable_point(pose, COCO_RIGHT_ANKLE, threshold)

    shoulder_center = (
        _midpoint(left_shoulder, right_shoulder)
        if left_shoulder is not None and right_shoulder is not None
        else None
    )
    hip_center = (
        _midpoint(left_hip, right_hip)
        if left_hip is not None and right_hip is not None
        else None
    )
    body_center = hip_center or (
        (pose.bbox[0] + pose.bbox[2]) / 2,
        (pose.bbox[1] + pose.bbox[3]) / 2,
    )
    torso_length = (
        _distance(shoulder_center, hip_center)
        if shoulder_center is not None and hip_center is not None
        else 0
    )
    left_extension = (
        _distance(left_shoulder, left_wrist) / torso_length
        if left_shoulder is not None and left_wrist is not None and torso_length > 0
        else None
    )
    right_extension = (
        _distance(right_shoulder, right_wrist) / torso_length
        if right_shoulder is not None
        and right_wrist is not None
        and torso_length > 0
        else None
    )
    extension_values = [
        value for value in (left_extension, right_extension) if value is not None
    ]
    body_extension = max(extension_values) if extension_values else None
    hitting_arm_candidate = "unknown"
    if left_extension is not None and right_extension is not None:
        if left_extension - right_extension >= 0.15:
            hitting_arm_candidate = "left"
        elif right_extension - left_extension >= 0.15:
            hitting_arm_candidate = "right"

    shoulder_width = (
        _distance(left_shoulder, right_shoulder)
        if left_shoulder is not None and right_shoulder is not None
        else 0
    )
    stance_width_ratio = (
        _distance(left_ankle, right_ankle) / shoulder_width
        if left_ankle is not None and right_ankle is not None and shoulder_width > 0
        else None
    )
    shoulder_angle = (
        math.degrees(
            math.atan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0],
            )
        )
        if left_shoulder is not None and right_shoulder is not None
        else None
    )
    limitations = [
        "hitting_arm_is_geometry_candidate_only",
        "forehand_backhand_not_inferred",
    ]
    if any(item > 1 for item in raw_confidences):
        limitations.append("raw_pose_confidence_clamped_to_one")
    if quality != "reliable":
        limitations.append("pose_quality_is_not_reliable")
    return CompactPoseFact(
        fact_id=(
            f"rally:{segment_index}:event:{event.event_index}:pose:{pose.frame}"
        ),
        source_frame=pose.frame,
        frame_delta=abs(pose.frame - event.frame),
        quality=quality,
        mean_keypoint_confidence=mean_confidence,
        usable_keypoint_count=usable_count,
        body_center_image=body_center,
        left_arm_extension=left_extension,
        right_arm_extension=right_extension,
        body_extension=body_extension,
        stance_width_ratio=stance_width_ratio,
        shoulder_angle_deg=shoulder_angle,
        hitting_arm_candidate=hitting_arm_candidate,
        limitations=limitations,
    )


def _inverse_3x3(matrix: list[list[float]]) -> list[list[float]] | None:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (
        d * h - e * g
    )
    if abs(determinant) < 1e-12:
        return None
    return [
        [
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
        ],
        [
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
        ],
        [
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant,
        ],
    ]


def _project(
    inverse: list[list[float]],
    point: tuple[float, float],
) -> tuple[float, float] | None:
    x, y = point
    denominator = inverse[2][0] * x + inverse[2][1] * y + inverse[2][2]
    if abs(denominator) < 1e-12:
        return None
    return (
        (inverse[0][0] * x + inverse[0][1] * y + inverse[0][2])
        / denominator,
        (inverse[1][0] * x + inverse[1][1] * y + inverse[1][2])
        / denominator,
    )


def _court_fact(
    *,
    segment_index: int,
    event: RallyFactEvent,
    pose: PoseFrame,
    position: CourtPosition,
    calibration: CourtCalibration,
    threshold: float,
) -> CompactCourtPositionFact | None:
    inverse = _inverse_3x3(calibration.homography)
    if inverse is None:
        return None
    left_ankle = _usable_point(pose, COCO_LEFT_ANKLE, threshold)
    right_ankle = _usable_point(pose, COCO_RIGHT_ANKLE, threshold)
    if left_ankle is not None and right_ankle is not None:
        image_point = _midpoint(left_ankle, right_ankle)
        source = "ankles_midpoint"
        quality = "reliable"
        limitations: list[str] = []
    else:
        image_point = (
            (pose.bbox[0] + pose.bbox[2]) / 2,
            pose.bbox[3],
        )
        source = "bbox_bottom_center"
        quality = "cautious"
        limitations = ["feet_missing_bbox_bottom_used"]
    projected = _project(inverse, image_point)
    if projected is None:
        return None
    court_x, court_y = projected
    if not 0 <= court_x <= COURT_WIDTH_M or not 0 <= court_y <= COURT_LENGTH_M:
        return None
    normalized_x = court_x / COURT_WIDTH_M
    normalized_y = court_y / COURT_LENGTH_M
    player_depth = (
        court_y / (COURT_LENGTH_M / 2)
        if position == "top"
        else (COURT_LENGTH_M - court_y) / (COURT_LENGTH_M / 2)
    )
    depth_zone = (
        "rear" if player_depth < 1 / 3 else "mid" if player_depth < 2 / 3 else "front"
    )
    width_zone = (
        "left"
        if normalized_x < 1 / 3
        else "center" if normalized_x < 2 / 3 else "right"
    )
    return CompactCourtPositionFact(
        fact_id=f"rally:{segment_index}:event:{event.event_index}:court",
        source_frame=pose.frame,
        quality=quality,
        position_source=source,
        court_x_m=court_x,
        court_y_m=court_y,
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        depth_zone=depth_zone,
        width_zone=width_zone,
        displacement_from_previous_hit_m=None,
        limitations=limitations,
    )


def _unit_vector(points: list[ShuttlePoint]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    if first.x is None or first.y is None or last.x is None or last.y is None:
        return None
    dx, dy = last.x - first.x, last.y - first.y
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-9 else (0.0, 0.0)


def _image_direction(vector: tuple[float, float] | None) -> ImageDirection | None:
    if vector is None:
        return None
    x, y = vector
    if abs(x) < 0.15 and abs(y) < 0.15:
        return "stable"
    horizontal = "left" if x < 0 else "right"
    vertical = "up" if y < 0 else "down"
    if abs(x) >= 2 * abs(y):
        return horizontal
    if abs(y) >= 2 * abs(x):
        return vertical
    return f"{vertical}_{horizontal}"  # type: ignore[return-value]


def _shuttle_fact(
    *,
    segment_index: int,
    event: RallyFactEvent,
    points: list[ShuttlePoint],
    config: CompactFactConfig,
) -> CompactShuttlePathFact:
    frames = [item.frame for item in points]
    start = event.frame - config.shuttle_window_frames
    end = event.frame + config.shuttle_window_frames
    left = bisect_left(frames, max(start, 0))
    right = bisect_left(frames, end + 1)
    window = points[left:right]
    usable = [
        item
        for item in window
        if item.visible
        and item.x is not None
        and item.y is not None
        and item.confidence >= config.shuttle_confidence
    ]
    incoming = _unit_vector([item for item in usable if item.frame <= event.frame])
    outgoing = _unit_vector([item for item in usable if item.frame >= event.frame])
    usable_ratio = len(usable) / len(window) if window else 0
    quality = (
        "reliable"
        if len(usable) >= 4 and usable_ratio >= 0.6
        else "cautious" if len(usable) >= 2 else "unavailable"
    )
    limitations = ["image_coordinates_only", "speed_not_computed"]
    if quality != "reliable":
        limitations.append("shuttle_path_quality_is_not_reliable")
    return CompactShuttlePathFact(
        fact_id=(
            f"rally:{segment_index}:event:{event.event_index}:shuttle:"
            f"{max(start, 0)}-{end}"
        ),
        start_frame=max(start, 0),
        end_frame=end,
        coordinate_space="image",
        quality=quality,
        sample_count=len(window),
        usable_sample_count=len(usable),
        usable_ratio=usable_ratio,
        incoming_unit_vector=incoming,
        outgoing_unit_vector=outgoing,
        incoming_direction=_image_direction(incoming),
        outgoing_direction=_image_direction(outgoing),
        limitations=limitations,
    )


def _nearest_pose(
    poses: list[PoseFrame],
    frame: int,
    max_delta: int,
) -> PoseFrame | None:
    if not poses:
        return None
    frames = [item.frame for item in poses]
    position = bisect_left(frames, frame)
    candidates = poses[max(0, position - 1) : min(len(poses), position + 1)]
    nearest = min(candidates, key=lambda item: abs(item.frame - frame))
    return nearest if abs(nearest.frame - frame) <= max_delta else None


def _position_for_player(
    player: Player | None,
    mapping: CourtPositionToPlayer | None,
) -> CourtPosition | None:
    if player is None or mapping is None:
        return None
    return "top" if mapping.top == player else "bottom"


def _select_calibration(
    stages: UpstreamStageData,
    segment_index: int,
) -> tuple[CourtCalibration | None, str | None]:
    vision = stages.vision
    if vision is None or vision.court_detection is None:
        return None, "court_stage_missing"
    court = vision.court_detection
    if court.detection_failed:
        return None, "court_detection_failed"
    if not court.confirmed:
        return None, "court_calibration_unconfirmed"
    exact = [item for item in court.courts if item.segment_index == segment_index]
    global_items = [item for item in court.courts if item.segment_index is None]
    candidates = exact or global_items
    if len(candidates) != 1:
        return None, "court_calibration_missing_or_ambiguous"
    return candidates[0], None


def build_compact_rally_facts(
    *,
    stages: UpstreamStageData,
    segment_index: int,
    court_position_to_player: CourtPositionToPlayer | None,
    config: CompactFactConfig | None = None,
) -> CompactRallyFacts:
    """Build compact, verified multimodal facts without invoking an LLM."""
    resolved_config = config or CompactFactConfig()
    fact = build_rally_fact_from_stages(
        stages=stages,
        segment_index=segment_index,
        court_position_to_player=court_position_to_player,
    )
    if stages.vision is not None and stages.vision.segment_index != segment_index:
        raise ValueError(
            "selected vision stages do not match the requested segment_index"
        )
    warnings: list[str] = []
    pose_stage = stages.vision.pose if stages.vision is not None else None
    shuttle_stage = (
        stages.vision.shuttle_tracking if stages.vision is not None else None
    )
    if pose_stage is None:
        warnings.append("pose_stage_missing")
    if shuttle_stage is None:
        warnings.append("shuttle_stage_missing")
    calibration, court_warning = _select_calibration(stages, segment_index)
    if court_warning is not None:
        warnings.append(court_warning)

    poses_by_position: dict[CourtPosition, list[PoseFrame]] = {
        "top": [],
        "bottom": [],
    }
    if pose_stage is not None:
        for pose in pose_stage.frames:
            poses_by_position[pose.player].append(pose)
        for poses in poses_by_position.values():
            poses.sort(key=lambda item: item.frame)
    shuttle_method = stages.stroke_classification.shuttle_method
    shuttle_points = sorted(
        (
            item
            for item in (shuttle_stage.points if shuttle_stage is not None else [])
            if shuttle_method is None or item.method == shuttle_method
        ),
        key=lambda item: item.frame,
    )
    if shuttle_stage is not None and shuttle_method is not None and not shuttle_points:
        warnings.append("configured_shuttle_method_has_no_points")

    compact_events: list[CompactStrokeFact] = []
    previous_court_by_player: dict[Player, CompactCourtPositionFact] = {}
    for event in fact.events:
        event_warnings: list[str] = []
        position = _position_for_player(event.player, court_position_to_player)
        raw_pose = (
            _nearest_pose(
                poses_by_position[position],
                event.frame,
                resolved_config.pose_max_frame_delta,
            )
            if position is not None
            else None
        )
        pose_fact = (
            _pose_fact(
                segment_index=segment_index,
                event=event,
                pose=raw_pose,
                threshold=resolved_config.pose_keypoint_confidence,
            )
            if raw_pose is not None
            else None
        )
        if raw_pose is None:
            event_warnings.append("pose_not_available_at_event")

        court_fact = (
            _court_fact(
                segment_index=segment_index,
                event=event,
                pose=raw_pose,
                position=position,
                calibration=calibration,
                threshold=resolved_config.pose_keypoint_confidence,
            )
            if raw_pose is not None
            and position is not None
            and calibration is not None
            else None
        )
        if calibration is not None and raw_pose is not None and court_fact is None:
            event_warnings.append("court_position_projection_failed")
        if court_fact is not None and event.player is not None:
            previous = previous_court_by_player.get(event.player)
            if previous is not None:
                displacement = math.hypot(
                    court_fact.court_x_m - previous.court_x_m,
                    court_fact.court_y_m - previous.court_y_m,
                )
                court_fact = court_fact.model_copy(
                    update={"displacement_from_previous_hit_m": displacement}
                )
            previous_court_by_player[event.player] = court_fact

        shuttle_fact = (
            _shuttle_fact(
                segment_index=segment_index,
                event=event,
                points=shuttle_points,
                config=resolved_config,
            )
            if shuttle_stage is not None
            else None
        )
        if shuttle_fact is not None and shuttle_fact.quality == "unavailable":
            event_warnings.append("shuttle_path_not_available_at_event")
        compact_events.append(
            CompactStrokeFact(
                fact_id=f"rally:{segment_index}:stroke:{event.event_index}",
                event_index=event.event_index,
                frame=event.frame,
                time_sec=event.time_sec,
                player=event.player,
                stroke_type=event.stroke_type,
                stroke_confidence=event.stroke_confidence,
                pose=pose_fact,
                court_position=court_fact,
                shuttle_path=shuttle_fact,
                warnings=event_warnings,
            )
        )

    segment = stages.match_segmentation.segments[segment_index]
    return CompactRallyFacts(
        schema_version="compact-rally-facts-v1",
        segment_index=segment_index,
        fps=stages.match_segmentation.fps,
        start_frame=segment.start_frame,
        end_frame=segment.end_frame,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        score=fact.score,
        server=fact.server,
        events=compact_events,
        warnings=list(dict.fromkeys(warnings)),
    )

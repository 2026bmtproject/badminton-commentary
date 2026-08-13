"""Build an event-centric stage slice for direct Gemini RallyFact experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    UpstreamStageData,
    read_upstream_stages,
)
from badminton_commentary.adapters.upstream import CourtPosition
from badminton_commentary.schemas import (
    NonNegativeFloat,
    NonNegativeInt,
    Player,
    Probability,
)
from badminton_commentary.facts.builder import _inverse_3x3, _project
from badminton_commentary.analysis.pose_geometry import (
    POSE_KEYFRAME_DELTAS,
    PoseGeometryFeatures,
    compute_pose_geometry,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_ROOT = EXPERIMENT_ROOT / "workspace" / "stages"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ttyvsasy" / "direct_rallyfact"
PROMPT_TEMPLATE_PATH = (
    EXPERIMENT_ROOT / "prompts" / "direct_rallyfact_event_centric_v3.txt"
)
PACKAGE_VERSION = "direct-rallyfact-event-centric-v4"
OUTPUT_SCHEMA_VERSION = "experimental-enriched-rally-fact-v3"
POSE_PRE_FRAMES = 8
POSE_POST_FRAMES = 10
SHUTTLE_WINDOW_RADIUS = 6
COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
COURT_HALF_LENGTH_M = COURT_LENGTH_M / 2
COURT_KEYPOINT_CONFIDENCE = 0.5
COURT_SINGLE_ANKLE_PENALTY = 0.75
COURT_BBOX_CONFIDENCE = 0.35
COURT_BASELINE_EXTENSION_M = 1.5
COURT_BASELINE_EXTENSION_CONFIDENCE_PENALTY = 0.85
COURT_DEPTH_FRONT_MAX = 1 / 3
COURT_DEPTH_MID_MAX = 2 / 3
COURT_DEPTH_CHANGE_EPSILON = 0.08

POSE_KEYPOINT_INDEXES = {
    "nose": 0,
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
POSE_KEYFRAME_KEYPOINTS = {
    name: index
    for name, index in POSE_KEYPOINT_INDEXES.items()
    if name not in {"nose", "left_elbow", "right_elbow"}
}
SOURCE_STAGES = [
    "match_segmentation",
    "event_detection",
    "stroke_classification",
    "score_recognition",
    "pose",
    "court_detection",
    "shuttle_tracking",
]


class PackageModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlayerMapping(PackageModel):
    top: Player
    bottom: Player


class PackageContext(PackageModel):
    package_version: Literal["direct-rallyfact-event-centric-v4"]
    segment_index: NonNegativeInt
    player_mapping: PlayerMapping
    pose_pre_frames: Literal[8]
    pose_post_frames: Literal[10]
    pose_keyframe_deltas: tuple[
        Literal[-8],
        Literal[-4],
        Literal[0],
        Literal[4],
        Literal[8],
        Literal[10],
    ]
    shuttle_window_radius_frames: Literal[6]
    pose_semantic_scope: list[Literal["posture_and_movement"]]
    pose_geometry_precomputed: Literal[True]
    court_geometry_precomputed: Literal[True]
    source_stages: list[str]
    notes: list[str]


class RallyScore(PackageModel):
    a: NonNegativeInt | None
    b: NonNegativeInt | None


class RallyMetadata(PackageModel):
    segment_index: NonNegativeInt
    start_frame: NonNegativeInt
    end_frame: NonNegativeInt
    start_sec: NonNegativeFloat
    end_sec: NonNegativeFloat
    duration_sec: NonNegativeFloat
    fps: Annotated[float, Field(gt=0)]
    score: RallyScore
    server: Player | None
    game_index: NonNegativeInt | None


class StrokeSlice(PackageModel):
    stroke_type: Annotated[str, Field(min_length=1)]
    confidence: Probability
    source_frame: NonNegativeInt


RawKeypoint = tuple[float, float, NonNegativeFloat]


class PoseWindowRecord(PackageModel):
    frame: NonNegativeInt
    frame_delta: int
    player: Player
    keypoints: dict[str, RawKeypoint]
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_window_record(self) -> PoseWindowRecord:
        if not -POSE_PRE_FRAMES <= self.frame_delta <= POSE_POST_FRAMES:
            raise ValueError("pose frame_delta exceeds configured window")
        if set(self.keypoints) != set(POSE_KEYPOINT_INDEXES):
            raise ValueError("pose record must contain exactly the selected keypoints")
        return self


class PoseKeyframeRecord(PackageModel):
    frame: NonNegativeInt
    frame_delta: Literal[-8, -4, 0, 4, 8, 10]
    keypoints: dict[str, RawKeypoint]
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_keyframe(self) -> PoseKeyframeRecord:
        if set(self.keypoints) != set(POSE_KEYFRAME_KEYPOINTS):
            raise ValueError("pose keyframe must contain exactly ten keypoints")
        return self


class ShuttleWindowPoint(PackageModel):
    frame: NonNegativeInt
    x: float
    y: float
    confidence: Probability


class ShuttleWindow(PackageModel):
    method: Annotated[str, Field(min_length=1)]
    start_frame: NonNegativeInt
    end_frame: NonNegativeInt
    points: list[ShuttleWindowPoint]
    excluded_points: NonNegativeInt


CourtPositionSource = Literal[
    "ankle_midpoint",
    "single_ankle",
    "bbox_bottom_center",
]
CourtDepthZone = Literal["front", "mid", "rear", "unknown"]
CourtDepthChange = Literal["forward", "backward", "stable", "unknown"]


class CourtPositionSlice(PackageModel):
    source_frame: NonNegativeInt
    frame_delta: int
    image_point: tuple[float, float]
    court_point_m: tuple[float, float]
    position_source: CourtPositionSource
    projection_confidence: Probability
    depth_zone: CourtDepthZone
    position_change_from_previous_same_player_hit: CourtDepthChange
    limitations: list[str]

    @model_validator(mode="after")
    def validate_source_frame(self) -> CourtPositionSlice:
        if not -POSE_PRE_FRAMES <= self.frame_delta <= POSE_POST_FRAMES:
            raise ValueError("court source frame is outside the pose window")
        return self


class EventSlice(PackageModel):
    event_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    stage_player: CourtPosition | None
    player: Player | None
    stroke: StrokeSlice | None
    pose_features: PoseGeometryFeatures | None
    pose_keyframes: list[PoseKeyframeRecord]
    pose_window: list[PoseWindowRecord]
    court_position: CourtPositionSlice | None
    shuttle_window: ShuttleWindow | None
    warnings: list[str]


class EventCentricStageInput(PackageModel):
    context: PackageContext
    rally: RallyMetadata
    events: list[EventSlice]

    @model_validator(mode="after")
    def validate_consistency(self) -> EventCentricStageInput:
        if self.context.segment_index != self.rally.segment_index:
            raise ValueError("context and rally segment_index must match")
        indexes = [event.event_index for event in self.events]
        if len(indexes) != len(set(indexes)):
            raise ValueError("event_index values must be unique")
        if indexes != sorted(indexes):
            raise ValueError("events must preserve source event_index order")
        for event in self.events:
            if not self.rally.start_frame <= event.frame <= self.rally.end_frame:
                raise ValueError("event frame is outside selected rally")
            for pose in event.pose_window:
                if not self.rally.start_frame <= pose.frame <= self.rally.end_frame:
                    raise ValueError("pose frame is outside selected rally")
                if pose.player != event.player:
                    raise ValueError("pose window contains a non-hitting player")
                if pose.frame - event.frame != pose.frame_delta:
                    raise ValueError("pose frame_delta does not match event frame")
            for keyframe in event.pose_keyframes:
                if keyframe.frame - event.frame != keyframe.frame_delta:
                    raise ValueError("pose keyframe delta does not match event frame")
                if keyframe.frame_delta not in POSE_KEYFRAME_DELTAS:
                    raise ValueError("pose keyframe uses a non-fixed delta")
            court = event.court_position
            if court is not None:
                if court.source_frame - event.frame != court.frame_delta:
                    raise ValueError("court frame_delta does not match event frame")
                if not self.rally.start_frame <= court.source_frame <= self.rally.end_frame:
                    raise ValueError("court source frame is outside selected rally")
            shuttle = event.shuttle_window
            if shuttle is None:
                continue
            expected_start = max(
                self.rally.start_frame,
                event.frame - SHUTTLE_WINDOW_RADIUS,
            )
            expected_end = min(
                self.rally.end_frame,
                event.frame + SHUTTLE_WINDOW_RADIUS,
            )
            if (shuttle.start_frame, shuttle.end_frame) != (
                expected_start,
                expected_end,
            ):
                raise ValueError("shuttle window does not match configured radius")
            if any(
                not shuttle.start_frame <= point.frame <= shuttle.end_frame
                for point in shuttle.points
            ):
                raise ValueError("shuttle point is outside its event window")
        return self


class CompactEventSlice(PackageModel):
    event_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    stage_player: CourtPosition | None
    player: Player | None
    stroke: StrokeSlice | None
    pose_features: PoseGeometryFeatures | None
    pose_keyframes: list[PoseKeyframeRecord]
    court_position: CourtPositionSlice | None
    shuttle_window: ShuttleWindow | None
    warnings: list[str]


class LLMEventCentricStageInput(PackageModel):
    context: PackageContext
    rally: RallyMetadata
    events: list[CompactEventSlice]

    @model_validator(mode="after")
    def validate_consistency(self) -> LLMEventCentricStageInput:
        if self.context.segment_index != self.rally.segment_index:
            raise ValueError("context and rally segment_index must match")
        indexes = [event.event_index for event in self.events]
        if len(indexes) != len(set(indexes)) or indexes != sorted(indexes):
            raise ValueError("compact events must preserve unique source order")
        for event in self.events:
            if not self.rally.start_frame <= event.frame <= self.rally.end_frame:
                raise ValueError("event frame is outside selected rally")
            if [item.frame_delta for item in event.pose_keyframes] != sorted(
                item.frame_delta for item in event.pose_keyframes
            ):
                raise ValueError("pose keyframes must be ordered")
        return self


@dataclass(frozen=True)
class PackageResult:
    directory: Path
    zip_path: Path
    file_count: int
    segment_index: int
    event_count: int
    raw_pose_record_count: int
    compact_pose_keyframe_count: int
    pose_features_count: int
    shuttle_point_count: int
    court_position_count: int
    full_debug_bytes: int
    llm_input_bytes: int
    raw_slice_estimated_bytes: int
    reduction_ratio: float

    @property
    def pose_record_count(self) -> int:
        return self.raw_pose_record_count

    @property
    def output_bytes(self) -> int:
        return self.llm_input_bytes


def _write_compact_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_size(payload: object) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _render_prompt(*, segment_index: int, mapping: CourtPositionToPlayer) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{{SEGMENT_INDEX}}", str(segment_index))
        .replace("{{TOP_PLAYER}}", mapping.top)
        .replace("{{BOTTOM_PLAYER}}", mapping.bottom)
    )


def _select_score(stages: UpstreamStageData, segment_index: int):
    scores = [
        score
        for score in stages.score_recognition.rallies
        if score.segment_index == segment_index
    ]
    if len(scores) > 1:
        raise ValueError(f"duplicate score for segment_index {segment_index}")
    return scores[0] if scores else None


def _select_court_calibration(
    stages: UpstreamStageData,
    *,
    segment_index: int,
) -> list[list[float]] | None:
    vision = stages.vision
    court = vision.court_detection if vision is not None else None
    if court is None or court.detection_failed:
        return None

    exact = [item for item in court.courts if item.segment_index == segment_index]
    candidates = exact or [
        item for item in court.courts if item.segment_index is None
    ]
    unique = {
        json.dumps(item.model_dump(mode="json"), sort_keys=True): item
        for item in candidates
    }
    if not unique:
        return None
    if len(unique) > 1:
        raise ValueError(
            "multiple different court calibrations match the selected segment; "
            "the current upstream schema has no frame/calibration id to select one"
        )
    calibration = next(iter(unique.values()))
    return calibration.homography


def _clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _usable_keypoint(
    pose,
    name: Literal["left_ankle", "right_ankle"],
) -> tuple[float, float, float] | None:
    x, y, confidence = pose.keypoints[POSE_KEYPOINT_INDEXES[name]]
    if (
        confidence < COURT_KEYPOINT_CONFIDENCE
        or not math.isfinite(x)
        or not math.isfinite(y)
        or not math.isfinite(confidence)
    ):
        return None
    return x, y, confidence


def _court_source_from_pose(
    pose,
) -> tuple[
    tuple[float, float],
    CourtPositionSource,
    float,
    list[str],
]:
    left = _usable_keypoint(pose, "left_ankle")
    right = _usable_keypoint(pose, "right_ankle")
    if left is not None and right is not None:
        return (
            ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2),
            "ankle_midpoint",
            _clamp_probability((left[2] + right[2]) / 2),
            [],
        )
    available = left or right
    if available is not None:
        return (
            (available[0], available[1]),
            "single_ankle",
            _clamp_probability(available[2] * COURT_SINGLE_ANKLE_PENALTY),
            ["single_ankle_fallback"],
        )
    return (
        ((pose.bbox[0] + pose.bbox[2]) / 2, pose.bbox[3]),
        "bbox_bottom_center",
        COURT_BBOX_CONFIDENCE,
        ["ankles_unavailable_bbox_bottom_center_used"],
    )


def _player_relative_depth(
    court_y: float,
    position: CourtPosition,
) -> float | None:
    if position == "top":
        if not -COURT_BASELINE_EXTENSION_M <= court_y <= COURT_HALF_LENGTH_M:
            return None
        return (COURT_HALF_LENGTH_M - court_y) / COURT_HALF_LENGTH_M
    if not (
        COURT_HALF_LENGTH_M
        <= court_y
        <= COURT_LENGTH_M + COURT_BASELINE_EXTENSION_M
    ):
        return None
    return (court_y - COURT_HALF_LENGTH_M) / COURT_HALF_LENGTH_M


def _is_behind_own_baseline(
    court_y: float,
    position: CourtPosition,
) -> bool:
    if position == "top":
        return -COURT_BASELINE_EXTENSION_M <= court_y < 0
    return COURT_LENGTH_M < court_y <= (
        COURT_LENGTH_M + COURT_BASELINE_EXTENSION_M
    )


def _depth_zone(normalized_depth: float | None) -> CourtDepthZone:
    if normalized_depth is None:
        return "unknown"
    if normalized_depth < COURT_DEPTH_FRONT_MAX:
        return "front"
    if normalized_depth < COURT_DEPTH_MID_MAX:
        return "mid"
    return "rear"


def _depth_change(
    previous: float | None,
    current: float | None,
) -> CourtDepthChange:
    if previous is None or current is None:
        return "unknown"
    delta = current - previous
    if abs(delta) <= COURT_DEPTH_CHANGE_EPSILON:
        return "stable"
    return "backward" if delta > 0 else "forward"


def _court_position_for_event(
    *,
    event_frame: int,
    stage_player: CourtPosition | None,
    pose_window,
    inverse_homography: list[list[float]] | None,
    previous_depth: float | None,
) -> tuple[CourtPositionSlice | None, float | None]:
    if stage_player is None or not pose_window or inverse_homography is None:
        return None, None
    pose = min(
        pose_window,
        key=lambda item: (abs(item.frame - event_frame), item.frame),
    )
    image_point, source, confidence, limitations = _court_source_from_pose(pose)
    projected = _project(inverse_homography, image_point)
    if projected is None or not all(math.isfinite(value) for value in projected):
        return None, None
    court_x, court_y = projected
    within_official_depth = 0 <= court_y <= COURT_LENGTH_M
    within_own_baseline_extension = _is_behind_own_baseline(
        court_y,
        stage_player,
    )
    if not 0 <= court_x <= COURT_WIDTH_M or not (
        within_official_depth or within_own_baseline_extension
    ):
        return None, None
    normalized_depth = _player_relative_depth(court_y, stage_player)
    if within_own_baseline_extension:
        limitations.append("projected_point_behind_own_baseline")
        confidence = _clamp_probability(
            confidence * COURT_BASELINE_EXTENSION_CONFIDENCE_PENALTY
        )
    if normalized_depth is None:
        limitations.append("projected_point_outside_player_half")
        confidence = _clamp_probability(confidence * 0.5)
    return (
        CourtPositionSlice(
            source_frame=pose.frame,
            frame_delta=pose.frame - event_frame,
            image_point=image_point,
            court_point_m=projected,
            position_source=source,
            projection_confidence=confidence,
            depth_zone=_depth_zone(normalized_depth),
            position_change_from_previous_same_player_hit=_depth_change(
                previous_depth,
                normalized_depth,
            ),
            limitations=limitations,
        ),
        normalized_depth,
    )


def _build_event_centric_input(
    *,
    stages: UpstreamStageData,
    segment_index: int,
    mapping: CourtPositionToPlayer,
) -> EventCentricStageInput:
    segments = stages.match_segmentation.segments
    if not 0 <= segment_index < len(segments):
        raise ValueError(
            f"segment_index {segment_index} is out of range for "
            f"{len(segments)} segments"
        )
    segment = segments[segment_index]
    fps = stages.match_segmentation.fps
    selected_events = [
        (event_index, event)
        for event_index, event in enumerate(stages.event_detection.events)
        if segment.start_frame <= event.frame <= segment.end_frame
    ]
    selected_indexes = {event_index for event_index, _ in selected_events}

    strokes_by_event = {}
    for stroke in stages.stroke_classification.strokes:
        if stroke.event_index in strokes_by_event:
            raise ValueError(f"duplicate stroke event_index {stroke.event_index}")
        strokes_by_event[stroke.event_index] = stroke
        if (
            stroke.segment_index == segment_index
            and stroke.event_index not in selected_indexes
        ):
            raise ValueError(
                f"stroke event_index {stroke.event_index} claims segment "
                f"{segment_index} but its event is outside the rally"
            )

    vision = stages.vision
    if vision is not None and vision.segment_index != segment_index:
        raise ValueError("selected vision stages do not match segment_index")

    pose_by_frame_player = {}
    if vision is not None and vision.pose is not None:
        for pose in vision.pose.frames:
            key = (pose.frame, pose.player)
            if key in pose_by_frame_player:
                raise ValueError(
                    f"duplicate pose record for frame {pose.frame} player {pose.player}"
                )
            pose_by_frame_player[key] = pose

    shuttle_method = stages.stroke_classification.shuttle_method
    shuttle_by_frame = {}
    if vision is not None and vision.shuttle_tracking is not None:
        for point in vision.shuttle_tracking.points:
            if point.method != shuttle_method:
                continue
            if point.frame in shuttle_by_frame:
                raise ValueError(
                    f"duplicate shuttle point for method {shuttle_method!r} "
                    f"frame {point.frame}"
                )
            shuttle_by_frame[point.frame] = point

    homography = _select_court_calibration(stages, segment_index=segment_index)
    inverse_homography = _inverse_3x3(homography) if homography is not None else None
    previous_depth_by_position: dict[CourtPosition, float] = {}
    events: list[EventSlice] = []
    for event_index, event in selected_events:
        stroke = strokes_by_event.get(event_index)
        warnings: list[str] = []
        if stroke is not None and stroke.segment_index != segment_index:
            raise ValueError(
                f"stroke event_index {event_index} references segment "
                f"{stroke.segment_index}, expected {segment_index}"
            )
        if stroke is None:
            warnings.append("no stroke_classification record for this event_index")
        elif stroke.frame != event.frame:
            warnings.append(
                f"stroke source_frame {stroke.frame} does not match "
                f"event frame {event.frame}"
            )

        stage_player = stroke.player if stroke is not None else None
        player = mapping.resolve(stage_player) if stage_player is not None else None
        pose_window: list[PoseWindowRecord] = []
        source_pose_window = []
        if stage_player is not None:
            pose_start = max(segment.start_frame, event.frame - POSE_PRE_FRAMES)
            pose_end = min(segment.end_frame, event.frame + POSE_POST_FRAMES)
            for frame in range(pose_start, pose_end + 1):
                pose = pose_by_frame_player.get((frame, stage_player))
                if pose is None:
                    continue
                source_pose_window.append(pose)
                pose_window.append(
                    PoseWindowRecord(
                        frame=pose.frame,
                        frame_delta=pose.frame - event.frame,
                        player=player,
                        keypoints={
                            name: pose.keypoints[index]
                            for name, index in POSE_KEYPOINT_INDEXES.items()
                        },
                        bbox=pose.bbox,
                    )
                )

        pose_features = (
            compute_pose_geometry(
                pose_window,
                hit_frame=event.frame,
                source_start_frame=max(
                    segment.start_frame,
                    event.frame - POSE_PRE_FRAMES,
                ),
                source_end_frame=min(
                    segment.end_frame,
                    event.frame + POSE_POST_FRAMES,
                ),
            )
            if pose_window
            else None
        )
        pose_keyframes = [
            PoseKeyframeRecord(
                frame=pose.frame,
                frame_delta=pose.frame_delta,
                keypoints={
                    name: pose.keypoints[name]
                    for name in POSE_KEYFRAME_KEYPOINTS
                },
                bbox=pose.bbox,
            )
            for pose in pose_window
            if pose.frame_delta in POSE_KEYFRAME_DELTAS
        ]

        previous_depth = (
            previous_depth_by_position.get(stage_player)
            if stage_player is not None
            else None
        )
        court_position, normalized_depth = _court_position_for_event(
            event_frame=event.frame,
            stage_player=stage_player,
            pose_window=source_pose_window,
            inverse_homography=inverse_homography,
            previous_depth=previous_depth,
        )
        if stage_player is not None and normalized_depth is not None:
            previous_depth_by_position[stage_player] = normalized_depth

        shuttle_window = None
        if (
            shuttle_method is not None
            and vision is not None
            and vision.shuttle_tracking is not None
        ):
            shuttle_start = max(
                segment.start_frame,
                event.frame - SHUTTLE_WINDOW_RADIUS,
            )
            shuttle_end = min(
                segment.end_frame,
                event.frame + SHUTTLE_WINDOW_RADIUS,
            )
            points = []
            excluded_points = 0
            for frame in range(shuttle_start, shuttle_end + 1):
                point = shuttle_by_frame.get(frame)
                if point is None:
                    continue
                if (
                    not point.visible
                    or point.x is None
                    or point.y is None
                    or point.confidence <= 0
                    or not math.isfinite(point.x)
                    or not math.isfinite(point.y)
                    or not math.isfinite(point.confidence)
                ):
                    excluded_points += 1
                    continue
                points.append(
                    ShuttleWindowPoint(
                        frame=point.frame,
                        x=point.x,
                        y=point.y,
                        confidence=point.confidence,
                    )
                )
            shuttle_window = ShuttleWindow(
                method=shuttle_method,
                start_frame=shuttle_start,
                end_frame=shuttle_end,
                points=points,
                excluded_points=excluded_points,
            )

        events.append(
            EventSlice(
                event_index=event_index,
                frame=event.frame,
                time_sec=event.frame / fps,
                stage_player=stage_player,
                player=player,
                stroke=(
                    StrokeSlice(
                        stroke_type=stroke.stroke_type,
                        confidence=stroke.confidence,
                        source_frame=stroke.frame,
                    )
                    if stroke is not None
                    else None
                ),
                pose_features=pose_features,
                pose_keyframes=pose_keyframes,
                pose_window=pose_window,
                court_position=court_position,
                shuttle_window=shuttle_window,
                warnings=warnings,
            )
        )

    score = _select_score(stages, segment_index)
    result = EventCentricStageInput(
        context=PackageContext(
            package_version=PACKAGE_VERSION,
            segment_index=segment_index,
            player_mapping=PlayerMapping.model_validate(
                mapping.model_dump(mode="json")
            ),
            pose_pre_frames=POSE_PRE_FRAMES,
            pose_post_frames=POSE_POST_FRAMES,
            pose_keyframe_deltas=POSE_KEYFRAME_DELTAS,
            shuttle_window_radius_frames=SHUTTLE_WINDOW_RADIUS,
            pose_semantic_scope=["posture_and_movement"],
            pose_geometry_precomputed=True,
            court_geometry_precomputed=True,
            source_stages=SOURCE_STAGES,
            notes=[
                "Pose geometry contains deterministic 2D measurements only.",
                "Pose posture semantics are not precomputed.",
                (
                    "No hitting-arm or forehand/backhand inference is "
                    "precomputed or requested."
                ),
                "Court projection and depth zones are deterministic preprocessing.",
                "event_index values are original full-match indexes.",
            ],
        ),
        rally=RallyMetadata(
            segment_index=segment_index,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            duration_sec=segment.duration_sec,
            fps=fps,
            score=RallyScore(
                a=score.score_a if score is not None else None,
                b=score.score_b if score is not None else None,
            ),
            server=score.server if score is not None else None,
            game_index=score.game_index if score is not None else None,
        ),
        events=events,
    )
    if len(result.events) != len(selected_events):
        raise ValueError("event-centric package did not preserve every selected event")
    return result


def _to_llm_input(
    debug_input: EventCentricStageInput,
) -> LLMEventCentricStageInput:
    return LLMEventCentricStageInput(
        context=debug_input.context,
        rally=debug_input.rally,
        events=[
            CompactEventSlice(
                event_index=event.event_index,
                frame=event.frame,
                time_sec=event.time_sec,
                stage_player=event.stage_player,
                player=event.player,
                stroke=event.stroke,
                pose_features=event.pose_features,
                pose_keyframes=event.pose_keyframes,
                court_position=event.court_position,
                shuttle_window=event.shuttle_window,
                warnings=event.warnings,
            )
            for event in debug_input.events
        ],
    )


def _raw_slice_estimated_size(
    *,
    stages: UpstreamStageData,
    segment_index: int,
) -> int:
    segment = stages.match_segmentation.segments[segment_index]
    vision = stages.vision
    payload = {
        "match_segmentation": {
            "fps": stages.match_segmentation.fps,
            "segments": [segment.model_dump(mode="json")],
        },
        "event_detection": {
            "events": [
                {"event_index": index, **event.model_dump(mode="json")}
                for index, event in enumerate(stages.event_detection.events)
                if segment.start_frame <= event.frame <= segment.end_frame
            ]
        },
        "score_recognition": {
            "rallies": [
                score.model_dump(mode="json")
                for score in stages.score_recognition.rallies
                if score.segment_index == segment_index
            ]
        },
        "stroke_classification": {
            "shuttle_method": stages.stroke_classification.shuttle_method,
            "strokes": [
                stroke.model_dump(mode="json")
                for stroke in stages.stroke_classification.strokes
                if stroke.segment_index == segment_index
            ],
        },
        "pose": (
            vision.pose.model_dump(mode="json")
            if vision is not None and vision.pose is not None
            else None
        ),
        "court_detection": (
            vision.court_detection.model_dump(mode="json")
            if vision is not None and vision.court_detection is not None
            else None
        ),
        "shuttle_tracking": (
            vision.shuttle_tracking.model_dump(mode="json")
            if vision is not None and vision.shuttle_tracking is not None
            else None
        ),
    }
    return _json_size(payload)


def _remove_legacy_outputs(output_dir: Path) -> None:
    legacy_files = [
        "context.json",
        "gemini_request_core.json",
        "gemini_request_all_stages.json",
        "prompt_core_only.txt",
        "prompt_with_core_stages.txt",
        "prompt_with_all_stages.txt",
    ]
    for filename in legacy_files:
        (output_dir / filename).unlink(missing_ok=True)
    legacy_stages = output_dir / "stages"
    if legacy_stages.is_dir():
        shutil.rmtree(legacy_stages)


def build_package(
    *,
    stage_root: Path,
    output_dir: Path,
    segment_index: int,
    mapping: CourtPositionToPlayer,
    overwrite: bool = False,
) -> PackageResult:
    zip_path = output_dir.with_suffix(".zip")
    if (output_dir.exists() or zip_path.exists()) and not overwrite:
        raise FileExistsError(
            f"package already exists; pass --overwrite: {output_dir}, {zip_path}"
        )
    if overwrite:
        _remove_legacy_outputs(output_dir)

    stages = read_upstream_stages(
        StagePaths.from_stage_root(stage_root),
        segment_index=segment_index,
    )
    debug_input = _build_event_centric_input(
        stages=stages,
        segment_index=segment_index,
        mapping=mapping,
    )
    llm_input = _to_llm_input(debug_input)
    payload = llm_input.model_dump(mode="json")
    debug_payload = debug_input.model_dump(mode="json")
    raw_slice_estimated_bytes = _raw_slice_estimated_size(
        stages=stages,
        segment_index=segment_index,
    )

    files: list[Path] = []
    input_path = output_dir / "rally_stage_input.json"
    _write_compact_json(input_path, payload)
    files.append(input_path)

    debug_path = output_dir / "rally_stage_input_debug.json"
    _write_compact_json(debug_path, debug_payload)
    files.append(debug_path)

    prompt = _render_prompt(segment_index=segment_index, mapping=mapping)
    prompt_path = output_dir / "prompt.txt"
    _write_text(prompt_path, prompt)
    files.append(prompt_path)

    combined_path = output_dir / "prompt_with_rally_stage_input.txt"
    _write_text(
        combined_path,
        prompt
        + "\n\n以下是 event-centric deterministic stage input JSON：\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    files.append(combined_path)

    pose_record_count = sum(len(event.pose_window) for event in debug_input.events)
    pose_keyframe_count = sum(
        len(event.pose_keyframes) for event in llm_input.events
    )
    pose_features_count = sum(
        event.pose_features is not None for event in llm_input.events
    )
    court_position_count = sum(
        event.court_position is not None for event in llm_input.events
    )
    shuttle_point_count = sum(
        len(event.shuttle_window.points)
        for event in llm_input.events
        if event.shuttle_window is not None
    )
    output_bytes = input_path.stat().st_size
    full_debug_bytes = debug_path.stat().st_size
    reduction_ratio = (
        full_debug_bytes / output_bytes if output_bytes else 0.0
    )
    readme_path = output_dir / "README.md"
    _write_text(
        readme_path,
        f"""# Direct Gemini event-centric RallyFact experiment: SEG{segment_index:04d}

`rally_stage_input.json` 是 v4 LLM-facing transport representation。它包含 deterministic pose geometry、固定 pose keyframes、court projection 與原始 shuttle window，但不包含姿態語義或戰術判斷。

`rally_stage_input_debug.json` 另外保留擊球方完整 -{POSE_PRE_FRAMES}/+{POSE_POST_FRAMES} pose window，只供 debug、provenance 與 visualization 使用，不應送入 LLM。

直接實驗可將 `prompt.txt` 與 `rally_stage_input.json` 一起提供給 Gemini，或貼上已合併的 `prompt_with_rally_stage_input.txt`。預期輸出 schema 是 `{OUTPUT_SCHEMA_VERSION}`。

事件以原始全場 `event_index` 保留；compact pose 每拍最多保留六個固定 delta keyframes，每個 keyframe 有肩、腕、髖、膝、踝共 10 個 keypoints。Shuttle 每拍最多保留指定 method 的 ±{SHUTTLE_WINDOW_RADIUS} frames 有效可見點。Raw court homography 不送入 LLM。

此 package 不含 production RallyFact 或 deterministic Fact Builder reference answer。Gemini 結果需另存並驗證後，才能交給 Planner / Commentator。
""",
    )
    files.append(readme_path)

    manifest_path = output_dir / "manifest.json"
    manifest_payload = {
        "package_version": PACKAGE_VERSION,
        "segment_index": segment_index,
        "player_mapping": mapping.model_dump(mode="json"),
        "counts": {
            "events": len(llm_input.events),
            "raw_pose_records": pose_record_count,
            "compact_pose_keyframes": pose_keyframe_count,
            "pose_features_available": pose_features_count,
            "shuttle_points": shuttle_point_count,
        },
        "window_radii": {
            "pose_pre_frames": POSE_PRE_FRAMES,
            "pose_post_frames": POSE_POST_FRAMES,
            "shuttle_frames": SHUTTLE_WINDOW_RADIUS,
        },
        "court_positions": {
            "available": court_position_count,
            "events": len(llm_input.events),
        },
        "sizes": {
            "raw_slice_estimated_bytes": raw_slice_estimated_bytes,
            "full_debug_bytes": full_debug_bytes,
            "llm_input_bytes": output_bytes,
            "reduction_ratio": reduction_ratio,
        },
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        ],
    }
    _write_json(manifest_path, manifest_payload)
    files.append(manifest_path)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(
                path,
                arcname=(
                    Path(output_dir.name) / path.relative_to(output_dir)
                ).as_posix(),
            )

    return PackageResult(
        directory=output_dir,
        zip_path=zip_path,
        file_count=len(files),
        segment_index=segment_index,
        event_count=len(llm_input.events),
        raw_pose_record_count=pose_record_count,
        compact_pose_keyframe_count=pose_keyframe_count,
        pose_features_count=pose_features_count,
        shuttle_point_count=shuttle_point_count,
        court_position_count=court_position_count,
        full_debug_bytes=full_debug_bytes,
        llm_input_bytes=output_bytes,
        raw_slice_estimated_bytes=raw_slice_estimated_bytes,
        reduction_ratio=reduction_ratio,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one event-centric stage slice for direct Gemini input."
    )
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--top-player", choices=("a", "b"), required=True)
    parser.add_argument("--bottom-player", choices=("a", "b"), required=True)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Accepted for experiment CLI compatibility; no provider is called.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output or (
        DEFAULT_OUTPUT_ROOT / f"seg{args.segment_index:04d}"
    )
    result = build_package(
        stage_root=args.stage_root,
        output_dir=output_dir,
        segment_index=args.segment_index,
        mapping=CourtPositionToPlayer(
            top=args.top_player,
            bottom=args.bottom_player,
        ),
        overwrite=args.overwrite,
    )
    print(f"package: {result.directory.resolve()}")
    print(f"zip: {result.zip_path.resolve()}")
    print(f"segment: {result.segment_index}")
    print(f"events: {result.event_count}")
    print(f"raw_pose_records: {result.raw_pose_record_count}")
    print(f"compact_pose_keyframes: {result.compact_pose_keyframe_count}")
    print(f"full_debug_bytes: {result.full_debug_bytes}")
    print(f"llm_input_bytes: {result.llm_input_bytes}")
    print(f"reduction_ratio: {result.reduction_ratio:.2f}x")
    print(
        "pose_features_available: "
        f"{result.pose_features_count}/{result.event_count}"
    )
    print(
        "court_positions_available: "
        f"{result.court_position_count}/{result.event_count}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from badminton_commentary.schemas import NonNegativeFloat, NonNegativeInt, Probability


CourtPosition = Literal["top", "bottom"]
PoseMode = Literal["fast", "balanced", "accurate"]


class VisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VisionEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PoseFrame(VisionRecord):
    frame: NonNegativeInt
    segment_index: NonNegativeInt
    player: CourtPosition
    keypoints: Annotated[
        list[tuple[float, float, NonNegativeFloat]],
        Field(min_length=17, max_length=17),
    ]
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_bbox(self) -> PoseFrame:
        left, top, right, bottom = self.bbox
        if right < left or bottom < top:
            raise ValueError("pose bbox must use left, top, right, bottom order")
        return self


class SelectedPoseStage(VisionRecord):
    segment_index: NonNegativeInt
    frames: list[PoseFrame]

    @model_validator(mode="after")
    def validate_segment_isolation(self) -> SelectedPoseStage:
        if any(item.segment_index != self.segment_index for item in self.frames):
            raise ValueError("selected pose stage contains another segment")
        return self


class CourtCalibration(VisionRecord):
    corners: Annotated[
        list[tuple[float, float]],
        Field(min_length=4, max_length=4),
    ]
    homography: Annotated[
        list[Annotated[list[float], Field(min_length=3, max_length=3)]],
        Field(min_length=3, max_length=3),
    ]
    segment_index: NonNegativeInt | None


class CourtDetectionStage(VisionEnvelope):
    courts: list[CourtCalibration]
    detection_failed: bool
    confirmed: bool


class ShuttlePoint(VisionRecord):
    frame: NonNegativeInt
    segment_index: NonNegativeInt
    method: Annotated[str, Field(min_length=1)]
    x: float | None
    y: float | None
    visible: bool
    confidence: Probability

    @model_validator(mode="after")
    def validate_coordinates(self) -> ShuttlePoint:
        if self.visible and (self.x is None or self.y is None):
            raise ValueError("visible shuttle point requires x and y")
        return self


class SelectedShuttleStage(VisionRecord):
    segment_index: NonNegativeInt
    fps: Annotated[float, Field(gt=0)]
    points: list[ShuttlePoint]

    @model_validator(mode="after")
    def validate_segment_isolation(self) -> SelectedShuttleStage:
        if any(item.segment_index != self.segment_index for item in self.points):
            raise ValueError("selected shuttle stage contains another segment")
        return self


class SelectedVisionStages(VisionRecord):
    segment_index: NonNegativeInt
    pose: SelectedPoseStage | None = None
    court_detection: CourtDetectionStage | None = None
    shuttle_tracking: SelectedShuttleStage | None = None


def _iter_top_level_array(path: Path, key: str) -> Iterator[object]:
    """Stream one top-level JSON array without retaining the complete artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"vision stage artifact not found: {path}")
    decoder = json.JSONDecoder()
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*\[')
    chunk_size = 1024 * 1024
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"top-level array {key!r} not found in {path}")
            buffer += chunk
            match = pattern.search(buffer)
            if match is not None:
                buffer = buffer[match.end() :]
                break
            buffer = buffer[-(len(key) + 32) :]

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f"invalid or truncated array {key!r} in {path}"
                    ) from exc
                buffer += chunk
            else:
                yield item
                buffer = buffer[end:]


def read_selected_pose_stage(
    path: Path,
    *,
    segment_index: int,
) -> SelectedPoseStage:
    frames = [
        PoseFrame.model_validate(item)
        for item in _iter_top_level_array(path, "frames")
        if isinstance(item, dict) and item.get("segment_index") == segment_index
    ]
    return SelectedPoseStage(segment_index=segment_index, frames=frames)


def read_court_detection_stage(path: Path) -> CourtDetectionStage:
    if not path.is_file():
        raise FileNotFoundError(f"vision stage artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"court stage artifact must contain a JSON object: {path}")
    return CourtDetectionStage.model_validate(payload)


def read_selected_shuttle_stage(
    path: Path,
    *,
    segment_index: int,
    fps: float,
) -> SelectedShuttleStage:
    points = [
        ShuttlePoint.model_validate(item)
        for item in _iter_top_level_array(path, "points")
        if isinstance(item, dict) and item.get("segment_index") == segment_index
    ]
    return SelectedShuttleStage(
        segment_index=segment_index,
        fps=fps,
        points=points,
    )


def read_selected_vision_stages(
    *,
    segment_index: int,
    fps: float,
    pose_path: Path | None,
    court_path: Path | None,
    shuttle_path: Path | None,
) -> SelectedVisionStages:
    return SelectedVisionStages(
        segment_index=segment_index,
        pose=(
            read_selected_pose_stage(pose_path, segment_index=segment_index)
            if pose_path is not None
            else None
        ),
        court_detection=(
            read_court_detection_stage(court_path)
            if court_path is not None
            else None
        ),
        shuttle_tracking=(
            read_selected_shuttle_stage(
                shuttle_path,
                segment_index=segment_index,
                fps=fps,
            )
            if shuttle_path is not None
            else None
        ),
    )

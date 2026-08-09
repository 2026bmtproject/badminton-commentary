from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from badminton_commentary.schemas import (
    NonNegativeFloat,
    NonNegativeInt,
    Player,
    Probability,
    RallyScore,
)


FactQuality = Literal["reliable", "cautious", "unavailable"]
HittingArmCandidate = Literal["left", "right", "unknown"]
CourtDepthZone = Literal["rear", "mid", "front"]
CourtWidthZone = Literal["left", "center", "right"]
CourtPositionSource = Literal["ankles_midpoint", "bbox_bottom_center"]
ImageDirection = Literal[
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
    "stable",
]


class CompactFactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompactPoseFact(CompactFactModel):
    fact_id: str
    source_frame: NonNegativeInt
    frame_delta: NonNegativeInt
    quality: FactQuality
    mean_keypoint_confidence: Probability
    usable_keypoint_count: Annotated[int, Field(ge=0, le=17)]
    body_center_image: tuple[float, float]
    left_arm_extension: NonNegativeFloat | None
    right_arm_extension: NonNegativeFloat | None
    body_extension: NonNegativeFloat | None
    stance_width_ratio: NonNegativeFloat | None
    shoulder_angle_deg: float | None
    hitting_arm_candidate: HittingArmCandidate
    limitations: list[str]


class CompactCourtPositionFact(CompactFactModel):
    fact_id: str
    source_frame: NonNegativeInt
    quality: Literal["reliable", "cautious"]
    position_source: CourtPositionSource
    court_x_m: float
    court_y_m: float
    normalized_x: Annotated[float, Field(ge=0, le=1)]
    normalized_y: Annotated[float, Field(ge=0, le=1)]
    depth_zone: CourtDepthZone
    width_zone: CourtWidthZone
    displacement_from_previous_hit_m: NonNegativeFloat | None
    limitations: list[str]


class CompactShuttlePathFact(CompactFactModel):
    fact_id: str
    start_frame: NonNegativeInt
    end_frame: NonNegativeInt
    coordinate_space: Literal["image"]
    quality: FactQuality
    sample_count: NonNegativeInt
    usable_sample_count: NonNegativeInt
    usable_ratio: Probability
    incoming_unit_vector: tuple[float, float] | None
    outgoing_unit_vector: tuple[float, float] | None
    incoming_direction: ImageDirection | None
    outgoing_direction: ImageDirection | None
    limitations: list[str]


class CompactStrokeFact(CompactFactModel):
    fact_id: str
    event_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    player: Player | None
    stroke_type: str | None
    stroke_confidence: Probability | None
    pose: CompactPoseFact | None
    court_position: CompactCourtPositionFact | None
    shuttle_path: CompactShuttlePathFact | None
    warnings: list[str]


class CompactRallyFacts(CompactFactModel):
    schema_version: Literal["compact-rally-facts-v1"]
    segment_index: NonNegativeInt
    fps: Annotated[float, Field(gt=0)]
    start_frame: NonNegativeInt
    end_frame: NonNegativeInt
    start_sec: NonNegativeFloat
    end_sec: NonNegativeFloat
    score: RallyScore
    server: Player | None
    events: list[CompactStrokeFact]
    warnings: list[str]

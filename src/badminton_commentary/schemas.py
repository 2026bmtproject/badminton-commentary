from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Player = Literal["a", "b"]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(StrictModel):
    start_frame: NonNegativeInt
    end_frame: NonNegativeInt
    start_sec: NonNegativeFloat
    end_sec: NonNegativeFloat
    duration_sec: NonNegativeFloat

    @model_validator(mode="after")
    def validate_range(self) -> "Segment":
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be greater than or equal to start_sec")
        expected_duration = self.end_sec - self.start_sec
        if abs(self.duration_sec - expected_duration) > 1e-6:
            raise ValueError("duration_sec must equal end_sec - start_sec")
        return self


class SegmentsInput(StrictModel):
    fps: Annotated[float, Field(gt=0)]
    segments: Annotated[list[Segment], Field(min_length=1)]


class ScoreRally(StrictModel):
    segment_index: NonNegativeInt
    score_a: NonNegativeInt | None
    score_b: NonNegativeInt | None
    server: Player | None
    game_index: NonNegativeInt | None


class ScoresInput(StrictModel):
    rallies: list[ScoreRally]


class HitEvent(StrictModel):
    frame: NonNegativeInt
    player: Player
    segment_index: NonNegativeInt


class EventsInput(StrictModel):
    events: list[HitEvent]


class Stroke(StrictModel):
    event_index: NonNegativeInt
    stroke_type: Annotated[str, Field(min_length=1)]
    confidence: Probability


class StrokesInput(StrictModel):
    strokes: list[Stroke]


class Highlight(StrictModel):
    segment_index: NonNegativeInt
    score: Probability


class HighlightsInput(StrictModel):
    highlights: list[Highlight]


class RallyScore(StrictModel):
    a: NonNegativeInt | None
    b: NonNegativeInt | None


class RallyFactEvent(StrictModel):
    event_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    player: Player
    stroke_type: str | None
    stroke_confidence: Probability | None


class RallyFact(StrictModel):
    segment_index: NonNegativeInt
    game_index: NonNegativeInt | None
    start_sec: NonNegativeFloat
    end_sec: NonNegativeFloat
    duration_sec: NonNegativeFloat
    score: RallyScore
    server: Player | None
    events: list[RallyFactEvent]
    rally_length: NonNegativeInt
    highlight_score: Probability | None


class ImportanceResult(StrictModel):
    score: Probability
    reasons: list[str]

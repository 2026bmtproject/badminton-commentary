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


class ScoredRallyFact(StrictModel):
    fact: RallyFact
    importance: ImportanceResult


StrokeConfidenceBand = Literal["reliable", "cautious"]
StrokePatternName = Literal[
    "serve_return_pattern",
    "lift_to_attack_transition",
    "sustained_attack",
    "rear_court_stroke_to_front_court_stroke",
    "stroke_diversity",
]


class AnalyzedStroke(StrictModel):
    fact_id: str
    event_index: NonNegativeInt
    player: Player
    stroke_type: str
    confidence: Probability
    confidence_band: StrokeConfidenceBand
    salience: Probability


class StrokePattern(StrictModel):
    fact_id: str
    name: StrokePatternName
    salience: Probability
    commentary_hint: str
    supporting_fact_ids: Annotated[list[str], Field(min_length=2)]
    representative_fact_id: str | None


class RallyAnalysis(StrictModel):
    segment_index: NonNegativeInt
    reliable_stroke_count: NonNegativeInt
    cautious_stroke_count: NonNegativeInt
    excluded_stroke_count: NonNegativeInt
    opening_observed_stroke: AnalyzedStroke | None
    final_observed_stroke: AnalyzedStroke | None
    candidate_strokes: list[AnalyzedStroke]
    notable_strokes: list[AnalyzedStroke]
    patterns: list[StrokePattern]
    warnings: list[str]


class RallyFactsOutput(StrictModel):
    rallies: list[ScoredRallyFact]


class RallyAnalysesOutput(StrictModel):
    analyses: list[RallyAnalysis]


CommentaryStyle = Literal["neutral", "analytical", "excited", "concise"]


class CommentaryPlan(StrictModel):
    segment_index: NonNegativeInt
    should_comment: bool
    style: CommentaryStyle
    focus: list[str]
    max_sentences: Annotated[int, Field(ge=1, le=3)]
    allowed_fact_ids: list[str]


class GeneratedCommentary(StrictModel):
    segment_index: NonNegativeInt
    text: Annotated[str, Field(min_length=1, max_length=240)]
    source_fact_ids: Annotated[list[str], Field(min_length=1)]


class CommentaryPlansOutput(StrictModel):
    plans: list[CommentaryPlan]


class CommentaryOutput(StrictModel):
    lines: list[GeneratedCommentary]


StrokeLocalFactName = Literal[
    "rear_exchange_continuation",
    "rear_court_stroke_to_front_court_stroke",
    "net_exchange_continuation",
    "flat_exchange_continuation",
    "net_to_lift_transition",
    "lift_to_attack_transition",
    "drop_lift_attack_sequence",
]


class StrokeLocalFact(StrictModel):
    fact_id: str
    name: StrokeLocalFactName
    start_stroke_index: NonNegativeInt
    end_stroke_index: NonNegativeInt
    salience: Probability
    commentary_hint: str
    supporting_fact_ids: Annotated[list[str], Field(min_length=2, max_length=3)]


class StrokeEventAnalysis(StrictModel):
    segment_index: NonNegativeInt
    stroke_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    current_stroke: AnalyzedStroke
    previous_strokes: Annotated[list[AnalyzedStroke], Field(max_length=4)]
    local_facts: list[StrokeLocalFact]
    speaking_score: Probability
    should_speak: bool


class StrokeEventPlan(StrictModel):
    segment_index: NonNegativeInt
    stroke_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    should_comment: bool
    style: CommentaryStyle
    max_sentences: Literal[1]
    focus: list[str]
    allowed_fact_ids: list[str]


class GeneratedStrokeText(StrictModel):
    text: Annotated[str, Field(min_length=1, max_length=120)]
    source_fact_ids: Annotated[list[str], Field(min_length=1)]


class StrokeCommentaryLine(StrictModel):
    segment_index: NonNegativeInt
    stroke_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    text: Annotated[str, Field(min_length=1, max_length=120)]
    source_fact_ids: Annotated[list[str], Field(min_length=1)]


class RallyCommentaryBundle(StrictModel):
    segment_index: NonNegativeInt
    events: list[StrokeCommentaryLine]
    summary: GeneratedCommentary | None


class EventDrivenCommentaryOutput(StrictModel):
    rallies: list[RallyCommentaryBundle]


SubtitleKind = Literal["event", "summary"]


class SubtitleCue(StrictModel):
    segment_index: NonNegativeInt
    kind: SubtitleKind
    start_sec: NonNegativeFloat
    end_sec: NonNegativeFloat
    text: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_time_range(self) -> "SubtitleCue":
        if self.end_sec <= self.start_sec:
            raise ValueError("subtitle end_sec must be greater than start_sec")
        return self

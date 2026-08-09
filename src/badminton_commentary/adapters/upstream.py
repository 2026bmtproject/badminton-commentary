from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from badminton_commentary.analysis.fact_builder import build_rally_facts
from badminton_commentary.schemas import (
    EventsInput,
    Highlight,
    HighlightsInput,
    HitEvent,
    NonNegativeInt,
    Player,
    Probability,
    RallyFact,
    ScoreRally,
    ScoresInput,
    Segment,
    SegmentsInput,
    Stroke,
    StrokesInput,
)


CourtPosition = Literal["top", "bottom"]


class UpstreamRecord(BaseModel):
    """Strict record from the Badminton Analysis System stage contracts."""

    model_config = ConfigDict(extra="forbid")


class UpstreamEnvelope(BaseModel):
    """Typed consumed fields while allowing producer metadata at envelope level."""

    model_config = ConfigDict(extra="ignore")


class UpstreamHitEvent(UpstreamRecord):
    frame: NonNegativeInt


class UpstreamStroke(UpstreamRecord):
    event_index: NonNegativeInt
    frame: NonNegativeInt
    segment_index: NonNegativeInt
    player: CourtPosition | None
    stroke_type: Annotated[str, Field(min_length=1)]
    confidence: Probability


class UpstreamScoreRally(UpstreamRecord):
    segment_index: NonNegativeInt
    score_a: NonNegativeInt | None
    score_b: NonNegativeInt | None
    server: Player | None = None
    game_index: NonNegativeInt | None = None
    sub_scores: list[list[NonNegativeInt]] | None = None
    split_secs: list[float] | None = None


class MatchSegmentationStage(UpstreamEnvelope):
    fps: Annotated[float, Field(gt=0)]
    segments: Annotated[list[Segment], Field(min_length=1)]


class EventDetectionStage(UpstreamEnvelope):
    events: list[UpstreamHitEvent]


class ScoreRecognitionStage(UpstreamEnvelope):
    rallies: list[UpstreamScoreRally]


class StrokeClassificationStage(UpstreamEnvelope):
    strokes: list[UpstreamStroke]


class HighlightStage(UpstreamEnvelope):
    highlights: list[Highlight]


class UpstreamStageData(UpstreamRecord):
    """Parsed stage data consumed by the production commentary adapter."""

    match_segmentation: MatchSegmentationStage
    event_detection: EventDetectionStage
    score_recognition: ScoreRecognitionStage
    stroke_classification: StrokeClassificationStage
    highlights: HighlightStage | None = None


class CourtPositionToPlayer(UpstreamRecord):
    """Identity mapping for one segment; court sides may swap during a match."""

    top: Player
    bottom: Player

    @model_validator(mode="after")
    def validate_distinct_players(self) -> CourtPositionToPlayer:
        if self.top == self.bottom:
            raise ValueError("top and bottom must map to different players")
        return self

    def resolve(self, position: CourtPosition) -> Player:
        return self.top if position == "top" else self.bottom


class StagePaths(UpstreamRecord):
    """Filesystem boundary; optional future stages are not parsed or consumed yet."""

    match_segmentation: Path
    event_detection: Path
    score_recognition: Path
    stroke_classification: Path
    highlights: Path | None = None
    court_detection: Path | None = None
    shuttle_tracking: Path | None = None
    pose: Path | None = None

    @classmethod
    def from_stage_root(cls, root: str | Path) -> StagePaths:
        root = Path(root)

        def optional(stage: str, filename: str) -> Path | None:
            path = root / stage / filename
            return path if path.exists() else None

        return cls(
            match_segmentation=root / "match_segmentation" / "segments.json",
            event_detection=root / "event_detection" / "events.json",
            score_recognition=root / "score_recognition" / "scores.json",
            stroke_classification=(
                root / "stroke_classification" / "strokes.json"
            ),
            highlights=optional("audio_highlight", "highlights.json"),
            court_detection=optional("court_detection", "court.json"),
            shuttle_tracking=optional("shuttle_tracking", "shuttle.json"),
            pose=optional("pose", "pose.json"),
        )


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required upstream stage artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"upstream stage artifact must contain a JSON object: {path}")
    return payload


def read_upstream_stages(paths: StagePaths) -> UpstreamStageData:
    """Read only the stages currently consumed by commentary."""
    highlights = (
        HighlightStage.model_validate(_read_json_object(paths.highlights))
        if paths.highlights is not None
        else None
    )
    return UpstreamStageData(
        match_segmentation=MatchSegmentationStage.model_validate(
            _read_json_object(paths.match_segmentation)
        ),
        event_detection=EventDetectionStage.model_validate(
            _read_json_object(paths.event_detection)
        ),
        score_recognition=ScoreRecognitionStage.model_validate(
            _read_json_object(paths.score_recognition)
        ),
        stroke_classification=StrokeClassificationStage.model_validate(
            _read_json_object(paths.stroke_classification)
        ),
        highlights=highlights,
    )


def _stroke_by_event(stages: UpstreamStageData) -> dict[int, UpstreamStroke]:
    event_count = len(stages.event_detection.events)
    indexed: dict[int, UpstreamStroke] = {}
    for stroke in stages.stroke_classification.strokes:
        if stroke.event_index >= event_count:
            raise ValueError(
                f"upstream stroke event_index {stroke.event_index} is out of range "
                f"for {event_count} events"
            )
        if stroke.event_index in indexed:
            raise ValueError(
                f"duplicate upstream stroke event_index {stroke.event_index}"
            )
        indexed[stroke.event_index] = stroke
    return indexed


def build_rally_fact_from_stages(
    *,
    stages: UpstreamStageData,
    segment_index: int,
    court_position_to_player: CourtPositionToPlayer | None,
) -> RallyFact:
    """Normalize one selected segment, then delegate joins to Fact Builder."""
    segments = stages.match_segmentation.segments
    if not 0 <= segment_index < len(segments):
        raise ValueError(
            f"segment_index {segment_index} is out of range for "
            f"{len(segments)} segments"
        )
    segment = segments[segment_index]
    stroke_by_event = _stroke_by_event(stages)
    selected_event_indexes = [
        event_index
        for event_index, event in enumerate(stages.event_detection.events)
        if segment.start_frame <= event.frame <= segment.end_frame
    ]
    selected_event_index_set = set(selected_event_indexes)

    for stroke in stages.stroke_classification.strokes:
        if (
            stroke.segment_index == segment_index
            and stroke.event_index not in selected_event_index_set
        ):
            raise ValueError(
                f"stroke event_index {stroke.event_index} claims segment "
                f"{segment_index} but its event frame is outside that segment"
            )

    normalized_events: list[HitEvent] = []
    normalized_strokes: list[Stroke] = []
    for local_event_index, source_event_index in enumerate(selected_event_indexes):
        raw_event = stages.event_detection.events[source_event_index]
        raw_stroke = stroke_by_event.get(source_event_index)
        player: Player | None = None
        if raw_stroke is not None:
            if raw_stroke.frame != raw_event.frame:
                raise ValueError(
                    f"stroke event_index {source_event_index} frame "
                    f"{raw_stroke.frame} does not match event frame {raw_event.frame}"
                )
            if raw_stroke.segment_index != segment_index:
                raise ValueError(
                    f"stroke event_index {source_event_index} references segment "
                    f"{raw_stroke.segment_index}, expected {segment_index}"
                )
            if raw_stroke.player is not None:
                if court_position_to_player is None:
                    raise ValueError(
                        "court_position_to_player is required because upstream "
                        "strokes identify court position, not player identity"
                    )
                player = court_position_to_player.resolve(raw_stroke.player)
            normalized_strokes.append(
                Stroke(
                    event_index=local_event_index,
                    stroke_type=raw_stroke.stroke_type,
                    confidence=raw_stroke.confidence,
                )
            )
        normalized_events.append(
            HitEvent(
                frame=raw_event.frame,
                player=player,
                segment_index=0,
                source_event_index=source_event_index,
            )
        )

    selected_scores = [
        score
        for score in stages.score_recognition.rallies
        if score.segment_index == segment_index
    ]
    if len(selected_scores) > 1:
        raise ValueError(f"duplicate score for segment_index {segment_index}")
    if selected_scores and selected_scores[0].sub_scores:
        if len(selected_scores[0].sub_scores) > 1:
            raise ValueError(
                "selected segment contains multiple recovered rallies; "
                "segment_index alone is ambiguous"
            )
    normalized_scores = ScoresInput(
        rallies=[
            ScoreRally(
                segment_index=0,
                score_a=score.score_a,
                score_b=score.score_b,
                server=score.server,
                game_index=score.game_index,
            )
            for score in selected_scores
        ]
    )

    selected_highlights = (
        [
            Highlight(segment_index=0, score=highlight.score)
            for highlight in stages.highlights.highlights
            if highlight.segment_index == segment_index
        ]
        if stages.highlights is not None
        else []
    )
    if len(selected_highlights) > 1:
        raise ValueError(f"duplicate highlight for segment_index {segment_index}")

    fact = build_rally_facts(
        segments=SegmentsInput(
            fps=stages.match_segmentation.fps,
            segments=[segment],
        ),
        scores=normalized_scores,
        events=EventsInput(events=normalized_events),
        strokes=StrokesInput(strokes=normalized_strokes),
        highlights=HighlightsInput(highlights=selected_highlights),
    )[0]
    return fact.model_copy(update={"segment_index": segment_index})

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from badminton_commentary.schemas import (
    NonNegativeInt,
    Player,
    Probability,
    StrictModel,
)


TacticalPatternType = Literal[
    "sustained_attack",
    "rear_to_front_stroke_transition",
    "attack_transition",
    "defense_to_counterattack_candidate",
    "front_back_court_displacement",
    "attacking_initiative_candidate",
    "notable_stroke_sequence",
    "rally_tactical_theme",
]


class GeneratedTacticalFact(StrictModel):
    """Provider-produced candidate before deterministic provenance validation."""

    pattern_type: TacticalPatternType
    description: Annotated[str, Field(min_length=1, max_length=160)]
    confidence: Probability
    salience: Probability
    start_event_index: NonNegativeInt
    end_event_index: NonNegativeInt
    players: Annotated[list[Player], Field(max_length=2)]
    evidence_fact_ids: Annotated[list[str], Field(min_length=2, max_length=12)]
    limitations: Annotated[list[str], Field(max_length=6)]

    @model_validator(mode="after")
    def validate_generated_fact(self) -> "GeneratedTacticalFact":
        if self.end_event_index < self.start_event_index:
            raise ValueError(
                "end_event_index must be greater than or equal to start_event_index"
            )
        if len(set(self.players)) != len(self.players):
            raise ValueError("players must not contain duplicates")
        if len(set(self.evidence_fact_ids)) != len(self.evidence_fact_ids):
            raise ValueError("evidence_fact_ids must not contain duplicates")
        return self


class GeneratedTacticalAnalysis(StrictModel):
    segment_index: NonNegativeInt
    facts: Annotated[list[GeneratedTacticalFact], Field(max_length=5)]


class TacticalFact(StrictModel):
    fact_id: Annotated[str, Field(min_length=1)]
    segment_index: NonNegativeInt
    pattern_type: TacticalPatternType
    description: Annotated[str, Field(min_length=1, max_length=160)]
    confidence: Probability
    salience: Probability
    start_event_index: NonNegativeInt
    end_event_index: NonNegativeInt
    players: Annotated[list[Player], Field(max_length=2)]
    evidence_fact_ids: Annotated[list[str], Field(min_length=2, max_length=12)]
    limitations: Annotated[list[str], Field(max_length=6)]

    @model_validator(mode="after")
    def validate_tactical_fact(self) -> "TacticalFact":
        if self.end_event_index < self.start_event_index:
            raise ValueError(
                "end_event_index must be greater than or equal to start_event_index"
            )
        if len(set(self.players)) != len(self.players):
            raise ValueError("players must not contain duplicates")
        if len(set(self.evidence_fact_ids)) != len(self.evidence_fact_ids):
            raise ValueError("evidence_fact_ids must not contain duplicates")
        return self


class TacticalAnalysisResult(StrictModel):
    schema_version: Literal["tactical-facts-v1"]
    prompt_version: Annotated[str, Field(min_length=1)]
    provider_model: str | None = None
    segment_index: NonNegativeInt
    facts: Annotated[list[TacticalFact], Field(max_length=5)]
    warnings: list[str]

    @model_validator(mode="after")
    def validate_result(self) -> "TacticalAnalysisResult":
        if any(fact.segment_index != self.segment_index for fact in self.facts):
            raise ValueError("all tactical facts must match result segment_index")
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("tactical fact ids must be unique")
        return self

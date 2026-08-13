from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from badminton_commentary.facts.schemas import CompactRallyFacts
from badminton_commentary.facts.tactical import (
    GeneratedTacticalAnalysis,
    GeneratedTacticalFact,
    TacticalAnalysisResult,
    TacticalFact,
)
from badminton_commentary.generation.json_response import extract_json_payload
from badminton_commentary.providers import LLMProvider, ProviderError


TACTICAL_PROMPT_VERSION = "gemini-tactical-analyzer-v1"
SYSTEM_PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "tactical_analyzer.txt"
)
ATTACK_STROKES = {"殺球", "撲球"}
REAR_STROKES = {"高遠球", "挑球", "殺球", "切球"}
FRONT_STROKES = {"小球", "撲球"}
UNSUPPORTED_DESCRIPTION_TERMS = {
    "最後一拍",
    "致勝",
    "得分",
    "迫使",
    "逼迫",
    "意圖",
    "球速",
    "速度",
    "加速",
    "減速",
    "正手",
    "反手",
    "因此",
    "所以",
    "導致",
    "靠著",
}


class TacticalAnalysisError(ProviderError):
    """Raised when a tactical response is invalid or not grounded."""


class _TacticalCandidateError(TacticalAnalysisError):
    """A single unsupported candidate that can be safely discarded."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Evidence:
    event_index: int
    player: str | None
    kind: str
    stroke_type: str | None = None
    court_depth_zone: str | None = None


def _evidence_catalog(compact: CompactRallyFacts) -> dict[str, _Evidence]:
    catalog: dict[str, _Evidence] = {}
    for event in compact.events:
        base = _Evidence(
            event_index=event.event_index,
            player=event.player,
            kind="stroke",
            stroke_type=event.stroke_type,
        )
        catalog[event.fact_id] = base
        if event.pose is not None:
            catalog[event.pose.fact_id] = _Evidence(
                event_index=event.event_index,
                player=event.player,
                kind="pose",
            )
        if event.court_position is not None:
            catalog[event.court_position.fact_id] = _Evidence(
                event_index=event.event_index,
                player=event.player,
                kind="court",
                court_depth_zone=event.court_position.depth_zone,
            )
        if event.shuttle_path is not None:
            catalog[event.shuttle_path.fact_id] = _Evidence(
                event_index=event.event_index,
                player=event.player,
                kind="shuttle",
            )
    return catalog


def _validate_pattern_evidence(
    generated: GeneratedTacticalFact,
    evidence: list[_Evidence],
) -> None:
    kinds = {item.kind for item in evidence}
    stroke_types = {
        item.stroke_type for item in evidence if item.stroke_type is not None
    }
    pattern_type = generated.pattern_type
    if pattern_type == "sustained_attack":
        attacking_players = {
            item.player
            for item in evidence
            if item.stroke_type in ATTACK_STROKES and item.player is not None
        }
        if not any(
            sum(
                item.player == player and item.stroke_type in ATTACK_STROKES
                for item in evidence
            )
            >= 2
            for player in attacking_players
        ):
            raise _TacticalCandidateError(
                "sustained_attack_missing_evidence",
                "sustained_attack requires two attack strokes by the same player"
            )
    elif pattern_type == "rear_to_front_stroke_transition":
        if not (stroke_types & REAR_STROKES and stroke_types & FRONT_STROKES):
            raise _TacticalCandidateError(
                "rear_front_missing_stroke_evidence",
                "rear_to_front_stroke_transition requires rear and front stroke evidence"
            )
        rear_events = [
            item.event_index for item in evidence if item.stroke_type in REAR_STROKES
        ]
        front_events = [
            item.event_index for item in evidence if item.stroke_type in FRONT_STROKES
        ]
        if not any(rear < front for rear in rear_events for front in front_events):
            raise _TacticalCandidateError(
                "rear_front_invalid_order",
                "rear_to_front_stroke_transition requires rear-before-front order"
            )
    elif pattern_type in {
        "attack_transition",
        "defense_to_counterattack_candidate",
        "attacking_initiative_candidate",
    }:
        if not stroke_types & ATTACK_STROKES:
            raise _TacticalCandidateError(
                "attack_stroke_missing",
                f"{pattern_type} requires at least one attack stroke"
            )
    elif pattern_type == "front_back_court_displacement":
        court_evidence = [item for item in evidence if item.kind == "court"]
        players_with_depth_change = {
            item.player
            for item in court_evidence
            if item.player is not None
            and len(
                {
                    candidate.court_depth_zone
                    for candidate in court_evidence
                    if candidate.player == item.player
                    and candidate.court_depth_zone is not None
                }
            )
            >= 2
        }
        if len(court_evidence) < 2 or not players_with_depth_change:
            raise _TacticalCandidateError(
                "same_player_depth_change_missing",
                "front_back_court_displacement requires one player's court facts in different depth zones"
            )
    elif pattern_type == "rally_tactical_theme":
        if len({item.event_index for item in evidence}) < 3:
            raise _TacticalCandidateError(
                "tactical_theme_insufficient_events",
                "rally_tactical_theme requires evidence from at least three events"
            )
    elif pattern_type == "notable_stroke_sequence" and "stroke" not in kinds:
        raise _TacticalCandidateError(
            "notable_sequence_stroke_missing",
            "notable_stroke_sequence requires stroke evidence"
        )


def _validate_generated_fact(
    generated: GeneratedTacticalFact,
    *,
    catalog: dict[str, _Evidence],
    valid_event_indexes: set[int],
) -> None:
    if generated.start_event_index not in valid_event_indexes:
        raise _TacticalCandidateError(
            "unknown_start_event",
            f"unknown start_event_index: {generated.start_event_index}"
        )
    if generated.end_event_index not in valid_event_indexes:
        raise _TacticalCandidateError(
            "unknown_end_event",
            f"unknown end_event_index: {generated.end_event_index}"
        )
    unknown_ids = [
        fact_id
        for fact_id in generated.evidence_fact_ids
        if fact_id not in catalog
    ]
    if unknown_ids:
        raise _TacticalCandidateError(
            "unknown_evidence_fact",
            f"unknown evidence fact ids: {unknown_ids}",
        )

    evidence = [catalog[fact_id] for fact_id in generated.evidence_fact_ids]
    evidence_indexes = {item.event_index for item in evidence}
    if len(evidence_indexes) < 2:
        raise _TacticalCandidateError(
            "insufficient_evidence_events",
            "a tactical fact requires evidence from at least two events"
        )
    outside_range = sorted(
        event_index
        for event_index in evidence_indexes
        if not (
            generated.start_event_index
            <= event_index
            <= generated.end_event_index
        )
    )
    if outside_range:
        raise _TacticalCandidateError(
            "evidence_outside_range",
            f"evidence events are outside the declared range: {outside_range}"
        )
    evidence_players = {
        item.player for item in evidence if item.player is not None
    }
    unsupported_players = set(generated.players) - evidence_players
    if unsupported_players:
        raise _TacticalCandidateError(
            "unsupported_player",
            f"players are not supported by evidence: {sorted(unsupported_players)}"
        )
    unsupported_terms = sorted(
        term
        for term in UNSUPPORTED_DESCRIPTION_TERMS
        if term in generated.description
    )
    if unsupported_terms:
        raise _TacticalCandidateError(
            "unsupported_description_claim",
            f"description contains unsupported claims: {unsupported_terms}"
        )
    if generated.pattern_type.endswith("_candidate"):
        if not generated.limitations:
            raise _TacticalCandidateError(
                "candidate_limitation_missing",
                f"{generated.pattern_type} requires an explicit limitation"
            )
        if not any(
            wording in generated.description
            for wording in ("可能", "候選", "看來", "似乎")
        ):
            raise _TacticalCandidateError(
                "candidate_uncertainty_wording_missing",
                f"{generated.pattern_type} requires uncertainty wording"
            )
    _validate_pattern_evidence(generated, evidence)


def _compact_payload(compact: CompactRallyFacts) -> dict[str, object]:
    return {
        "schema_version": compact.schema_version,
        "segment_index": compact.segment_index,
        "fps": compact.fps,
        "start_frame": compact.start_frame,
        "end_frame": compact.end_frame,
        "score": compact.score.model_dump(),
        "events": [event.model_dump(mode="json") for event in compact.events],
        "warnings": compact.warnings,
    }


def analyze_tactical_facts(
    *,
    provider: LLMProvider,
    compact_facts: CompactRallyFacts,
    max_facts: int = 5,
) -> TacticalAnalysisResult:
    """Use one provider call to derive validated, traceable tactical candidates."""
    if not 1 <= max_facts <= 5:
        raise ValueError("max_facts must be between 1 and 5")
    if len(compact_facts.events) < 2:
        return TacticalAnalysisResult(
            schema_version="tactical-facts-v1",
            prompt_version=TACTICAL_PROMPT_VERSION,
            provider_model=None,
            segment_index=compact_facts.segment_index,
            facts=[],
            warnings=["insufficient_events_for_tactical_analysis"],
        )

    payload = {
        "prompt_version": TACTICAL_PROMPT_VERSION,
        "max_facts": max_facts,
        "compact_rally_facts": _compact_payload(compact_facts),
    }
    response = provider.generate(
        system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        user_prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    try:
        generated = GeneratedTacticalAnalysis.model_validate_json(
            extract_json_payload(response)
        )
    except ValidationError as exc:
        raise TacticalAnalysisError(
            f"provider returned invalid tactical JSON: {exc}"
        ) from exc

    if generated.segment_index != compact_facts.segment_index:
        raise TacticalAnalysisError("tactical segment_index does not match input")
    if len(generated.facts) > max_facts:
        raise TacticalAnalysisError(
            f"provider returned more than max_facts={max_facts}"
        )

    catalog = _evidence_catalog(compact_facts)
    valid_event_indexes = {event.event_index for event in compact_facts.events}
    facts: list[TacticalFact] = []
    warnings: list[str] = []
    seen_signatures: set[tuple[str, int, int]] = set()
    for source_index, candidate in enumerate(generated.facts):
        try:
            _validate_generated_fact(
                candidate,
                catalog=catalog,
                valid_event_indexes=valid_event_indexes,
            )
        except _TacticalCandidateError as exc:
            warnings.append(
                "rejected_tactical_fact:"
                f"{source_index}:{candidate.pattern_type}:{exc.code}"
            )
            continue
        signature = (
            candidate.pattern_type,
            candidate.start_event_index,
            candidate.end_event_index,
        )
        if signature in seen_signatures:
            warnings.append(
                "rejected_tactical_fact:"
                f"{source_index}:{candidate.pattern_type}:duplicate_candidate"
            )
            continue
        seen_signatures.add(signature)
        fact_index = len(facts)
        facts.append(
            TacticalFact(
                fact_id=(
                    f"rally:{compact_facts.segment_index}:tactical:"
                    f"{fact_index}:{candidate.pattern_type}"
                ),
                segment_index=compact_facts.segment_index,
                **candidate.model_dump(),
            )
        )

    provider_model = getattr(provider, "last_model_used", None)
    primary_model = getattr(provider, "model", None)
    if not facts:
        warnings.append("no_supported_tactical_patterns")
    if (
        isinstance(provider_model, str)
        and isinstance(primary_model, str)
        and provider_model != primary_model
    ):
        warnings.append(f"provider_model_fallback:{primary_model}->{provider_model}")
    return TacticalAnalysisResult(
        schema_version="tactical-facts-v1",
        prompt_version=TACTICAL_PROMPT_VERSION,
        provider_model=provider_model if isinstance(provider_model, str) else None,
        segment_index=compact_facts.segment_index,
        facts=facts,
        warnings=warnings,
    )

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from badminton_commentary.providers.base import LLMProvider, ProviderError
from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.schemas import (
    CommentaryPlan,
    GeneratedCommentary,
    RallyAnalysis,
    ScoredRallyFact,
)

from .json_response import extract_json_payload
from .validator import (
    CommentaryValidationError,
    commentary_allows_exclamation,
    normalize_exclamation_emphasis,
    validate_commentary,
)


PROMPT_VERSION = "commentator-v3"
SYSTEM_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "commentator.txt"


class CommentaryGenerationError(ProviderError):
    """Raised when a provider response cannot become grounded commentary."""


def build_fact_catalog(
    scored: ScoredRallyFact,
    analysis: RallyAnalysis,
) -> dict[str, object]:
    fact = scored.fact
    prefix = f"rally:{fact.segment_index}"
    catalog: dict[str, object] = {}
    if fact.score.a is not None and fact.score.b is not None:
        catalog[f"{prefix}:score"] = {"a": fact.score.a, "b": fact.score.b}
    if fact.events:
        catalog[f"{prefix}:length"] = fact.rally_length
        for event in fact.events:
            catalog[f"{prefix}:stroke:{event.event_index}"] = {
                "frame": event.frame,
                "time_sec": event.time_sec,
                "player": event.player,
                "stroke_type": event.stroke_type,
                "confidence": event.stroke_confidence,
            }
    for pattern in analysis.patterns:
        candidate_by_id = {
            stroke.fact_id: stroke for stroke in analysis.candidate_strokes
        }
        supporting = {
            fact_id: candidate_by_id[fact_id].model_dump()
            for fact_id in pattern.supporting_fact_ids
            if fact_id in candidate_by_id
        }
        catalog[pattern.fact_id] = {
            "pattern": pattern.name,
            "salience": pattern.salience,
            "commentary_hint": pattern.commentary_hint,
            "evidence_scope": "stroke_sequence_only",
            "does_not_imply": [
                "player_movement",
                "tactical_intent",
                "causality",
                "rally_outcome",
            ],
            "supporting_fact_ids": pattern.supporting_fact_ids,
            "supporting_strokes": supporting,
        }
    for stroke in analysis.notable_strokes:
        catalog[stroke.fact_id] = stroke.model_dump()
    if fact.highlight_score is not None:
        catalog[f"{prefix}:highlight"] = fact.highlight_score
    return catalog


def generate_commentary(
    *,
    provider: LLMProvider,
    scored: ScoredRallyFact,
    plan: CommentaryPlan,
    analysis: RallyAnalysis | None = None,
    player_names: dict[str, str] | None = None,
) -> GeneratedCommentary:
    """Generate a legacy summary-only grounded commentary response."""
    if not plan.should_comment:
        raise ValueError("cannot generate commentary for a skipped plan")
    if plan.segment_index != scored.fact.segment_index:
        raise ValueError("plan segment_index does not match rally fact")

    resolved_analysis = analysis or analyze_rally(scored.fact)
    catalog = build_fact_catalog(scored, resolved_analysis)
    missing = set(plan.allowed_fact_ids) - set(catalog)
    if missing:
        raise ValueError(f"plan references unavailable fact ids: {sorted(missing)}")
    allowed_catalog = {
        fact_id: catalog[fact_id]
        for fact_id in plan.allowed_fact_ids
        if fact_id in catalog
    }
    user_payload = {
        "prompt_version": PROMPT_VERSION,
        "players": player_names or {"a": "player a", "b": "player b"},
        "plan": plan.model_dump(),
        "fact_catalog": allowed_catalog,
    }
    response = provider.generate(
        system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
    )
    try:
        generated = GeneratedCommentary.model_validate_json(
            extract_json_payload(response)
        )
    except ValidationError as exc:
        raise CommentaryGenerationError(
            f"provider returned invalid commentary JSON: {exc}"
        ) from exc
    generated = generated.model_copy(
        update={
            "text": normalize_exclamation_emphasis(
                generated.text,
                allow_exclamation=commentary_allows_exclamation(scored),
            )
        }
    )

    try:
        validate_commentary(
            generated=generated,
            scored=scored,
            plan=plan,
            analysis=resolved_analysis,
        )
    except CommentaryValidationError as exc:
        raise CommentaryGenerationError(str(exc)) from exc
    return generated

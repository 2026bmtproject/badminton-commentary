from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from badminton_commentary.providers import LLMProvider, ProviderError
from badminton_commentary.schemas import (
    GeneratedStrokeText,
    RallyScore,
    StrokeCommentaryLine,
    StrokeEventAnalysis,
    StrokeEventPlan,
)

from .event_validator import validate_stroke_commentary
from .json_response import extract_json_payload
from .validator import CommentaryValidationError, normalize_exclamation_emphasis


STROKE_PROMPT_VERSION = "stroke-commentator-v1"
SYSTEM_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "stroke_commentator.txt"


class StrokeCommentaryGenerationError(ProviderError):
    """Raised when an event-driven provider response is invalid or unsupported."""


def _stroke_catalog(
    analysis: StrokeEventAnalysis,
    plan: StrokeEventPlan,
    score: RallyScore,
) -> dict[str, object]:
    strokes = [*analysis.previous_strokes, analysis.current_stroke]
    catalog: dict[str, object] = {
        stroke.fact_id: stroke.model_dump() for stroke in strokes
    }
    for local in analysis.local_facts:
        catalog[local.fact_id] = {
            **local.model_dump(),
            "evidence_scope": "adjacent_stroke_sequence_only",
            "does_not_imply": [
                "player_movement",
                "tactical_intent",
                "causality",
                "rally_outcome",
            ],
        }
    if score.a is not None and score.b is not None:
        catalog[f"rally:{analysis.segment_index}:score"] = {
            "a": score.a,
            "b": score.b,
        }
    return {
        fact_id: catalog[fact_id]
        for fact_id in plan.allowed_fact_ids
        if fact_id in catalog
    }


def generate_stroke_commentary(
    *,
    provider: LLMProvider,
    analysis: StrokeEventAnalysis,
    plan: StrokeEventPlan,
    score: RallyScore,
    player_names: dict[str, str] | None = None,
) -> StrokeCommentaryLine:
    if not plan.should_comment:
        raise ValueError("cannot generate commentary for a skipped stroke plan")
    if (
        plan.segment_index != analysis.segment_index
        or plan.stroke_index != analysis.stroke_index
        or plan.frame != analysis.frame
        or plan.time_sec != analysis.time_sec
    ):
        raise ValueError("stroke plan metadata does not match event analysis")

    user_payload = {
        "prompt_version": STROKE_PROMPT_VERSION,
        "players": player_names or {"a": "player a", "b": "player b"},
        "plan": plan.model_dump(),
        "current_stroke": analysis.current_stroke.model_dump(),
        "previous_strokes": [item.model_dump() for item in analysis.previous_strokes],
        "fact_catalog": _stroke_catalog(analysis, plan, score),
    }
    response = provider.generate(
        system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
    )
    try:
        generated = GeneratedStrokeText.model_validate_json(
            extract_json_payload(response)
        )
    except ValidationError as exc:
        raise StrokeCommentaryGenerationError(
            f"provider returned invalid stroke commentary JSON: {exc}"
        ) from exc
    generated = generated.model_copy(
        update={
            "text": normalize_exclamation_emphasis(
                generated.text,
                allow_exclamation=analysis.speaking_score >= 0.9,
            )
        }
    )
    try:
        validate_stroke_commentary(
            generated=generated,
            analysis=analysis,
            plan=plan,
            score=score,
        )
    except CommentaryValidationError as exc:
        raise StrokeCommentaryGenerationError(str(exc)) from exc

    return StrokeCommentaryLine(
        segment_index=analysis.segment_index,
        stroke_index=analysis.stroke_index,
        frame=analysis.frame,
        time_sec=analysis.time_sec,
        text=generated.text,
        source_fact_ids=generated.source_fact_ids,
    )

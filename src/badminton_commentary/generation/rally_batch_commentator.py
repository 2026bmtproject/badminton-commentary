from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from badminton_commentary.analysis import analyze_rally, analyze_stroke_events
from badminton_commentary.providers import LLMProvider, ProviderError
from badminton_commentary.schemas import (
    GeneratedCommentary,
    GeneratedRallyTextBatch,
    GeneratedStrokeText,
    RallyAnalysis,
    RallyCommentaryBundle,
    ScoredRallyFact,
    StrokeCommentaryLine,
    StrokeEventAnalysis,
    StrokeEventPlan,
)

from .commentator import build_fact_catalog
from .event_commentator import build_stroke_catalog
from .event_planner import plan_stroke_commentary
from .event_validator import ensure_uncertainty_wording, validate_stroke_commentary
from .json_response import extract_json_payload
from .planner import plan_commentary, plan_selected_rally_summary
from .validator import (
    CommentaryValidationError,
    CommentaryLanguageSafetyError,
    commentary_allows_exclamation,
    normalize_exclamation_emphasis,
    validate_commentary,
)


BATCH_PROMPT_VERSION = "rally-batch-commentator-v2"
SYSTEM_PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "rally_batch_commentator.txt"
)


class RallyBatchGenerationError(ProviderError):
    """Raised when one batched rally response is invalid or unsupported."""


def _safe_stroke_fallback(
    *,
    analysis: StrokeEventAnalysis,
    plan: StrokeEventPlan,
    player_names: dict[str, str] | None,
) -> GeneratedStrokeText:
    """Build a grounded sentence when provider wording violates language safety."""
    planned_local = next(
        (
            local
            for local in analysis.local_facts
            if local.fact_id in plan.allowed_fact_ids
        ),
        None,
    )
    if planned_local is not None:
        text = planned_local.commentary_hint
        source_fact_ids = [
            planned_local.fact_id,
            *planned_local.supporting_fact_ids,
        ]
    else:
        names = player_names or {"a": "player a", "b": "player b"}
        player = names.get(
            analysis.current_stroke.player,
            f"player {analysis.current_stroke.player}",
        )
        text = f"{player}打出{analysis.current_stroke.stroke_type}"
        source_fact_ids = [analysis.current_stroke.fact_id]
    text = f"{text.rstrip('。！？!?')}。"
    return GeneratedStrokeText(
        text=ensure_uncertainty_wording(
            text,
            confidence_band=analysis.current_stroke.confidence_band,
        ),
        source_fact_ids=source_fact_ids,
    )


def _safe_summary_fallback(
    *,
    scored: ScoredRallyFact,
    analysis: RallyAnalysis,
    allowed_fact_ids: list[str],
) -> GeneratedCommentary:
    """Build a minimal grounded summary after unsafe provider wording."""
    fact = scored.fact
    pattern = next(
        (item for item in analysis.patterns if item.fact_id in allowed_fact_ids),
        None,
    )
    if pattern is not None:
        text = pattern.commentary_hint
        source_fact_ids = [pattern.fact_id]
    elif f"rally:{fact.segment_index}:score" in allowed_fact_ids:
        text = f"目前比分為 {fact.score.a} 比 {fact.score.b}"
        source_fact_ids = [f"rally:{fact.segment_index}:score"]
    elif f"rally:{fact.segment_index}:length" in allowed_fact_ids:
        text = f"這段來回共 {fact.rally_length} 拍"
        source_fact_ids = [f"rally:{fact.segment_index}:length"]
    else:
        text = "這個片段具有精彩片段標記"
        source_fact_ids = [f"rally:{fact.segment_index}:highlight"]
    return GeneratedCommentary(
        segment_index=fact.segment_index,
        text=f"{text.rstrip('。！？!?')}。",
        source_fact_ids=source_fact_ids,
    )


def generate_rally_commentary_batch(
    *,
    provider: LLMProvider,
    scored: ScoredRallyFact,
    player_names: dict[str, str] | None = None,
    require_summary: bool = False,
    user_selected_rally: bool = False,
) -> RallyCommentaryBundle:
    """Generate every identifiable stroke plus an optional rally summary in one call."""
    fact = scored.fact
    event_units = []
    fact_catalog: dict[str, object] = {
        f"rally:{fact.segment_index}:stroke:{event.event_index}": {
            "frame": event.frame,
            "time_sec": event.time_sec,
            "player": event.player,
            "stroke_type": event.stroke_type,
            "confidence": event.stroke_confidence,
            "role": "ordered_stroke_context",
        }
        for event in fact.events
        if event.stroke_type is not None and event.stroke_confidence is not None
    }
    for analysis in analyze_stroke_events(fact, include_all_strokes=True):
        plan = plan_stroke_commentary(
            analysis,
            force_commentary=True,
        )
        event_units.append((analysis, plan))
        fact_catalog.update(build_stroke_catalog(analysis, plan, fact.score))

    rally_analysis = analyze_rally(fact)
    summary_plan = (
        plan_selected_rally_summary(fact, rally_analysis)
        if user_selected_rally
        else plan_commentary(scored, rally_analysis)
    )
    if require_summary and not summary_plan.should_comment:
        allowed_fact_ids = list(summary_plan.allowed_fact_ids)
        length_fact_id = f"rally:{fact.segment_index}:length"
        if fact.events and length_fact_id not in allowed_fact_ids:
            allowed_fact_ids.append(length_fact_id)
        if allowed_fact_ids:
            summary_plan = summary_plan.model_copy(
                update={
                    "should_comment": True,
                    "style": "neutral",
                    "max_sentences": 1,
                    "allowed_fact_ids": allowed_fact_ids,
                }
            )
    if summary_plan.should_comment:
        summary_catalog = build_fact_catalog(scored, rally_analysis)
        fact_catalog.update(
            {
                fact_id: summary_catalog[fact_id]
                for fact_id in summary_plan.allowed_fact_ids
            }
        )

    if not event_units and not summary_plan.should_comment:
        return RallyCommentaryBundle(
            segment_index=fact.segment_index,
            events=[],
            summary=None,
        )

    payload = {
        "prompt_version": BATCH_PROMPT_VERSION,
        "segment_index": fact.segment_index,
        "players": player_names or {"a": "player a", "b": "player b"},
        "all_stroke_fact_ids": [
            f"rally:{fact.segment_index}:stroke:{event.event_index}"
            for event in fact.events
            if event.stroke_type is not None and event.stroke_confidence is not None
        ],
        "event_units": [
            {
                "stroke_index": analysis.stroke_index,
                "frame": analysis.frame,
                "time_sec": analysis.time_sec,
                "current_stroke_fact_id": analysis.current_stroke.fact_id,
                "plan": plan.model_dump(),
            }
            for analysis, plan in event_units
        ],
        "summary_unit": summary_plan.model_dump()
        if summary_plan.should_comment
        else None,
        "fact_catalog": fact_catalog,
    }
    response = provider.generate(
        system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        user_prompt=json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    try:
        generated = GeneratedRallyTextBatch.model_validate_json(
            extract_json_payload(response)
        )
    except ValidationError as exc:
        raise RallyBatchGenerationError(
            f"provider returned invalid rally batch JSON: {exc}"
        ) from exc

    if generated.segment_index != fact.segment_index:
        raise RallyBatchGenerationError("rally batch segment_index does not match")
    expected_indexes = [analysis.stroke_index for analysis, _ in event_units]
    actual_indexes = [item.stroke_index for item in generated.events]
    if actual_indexes != expected_indexes:
        raise RallyBatchGenerationError(
            "rally batch event stroke indexes do not match the planned order: "
            f"expected {expected_indexes}, got {actual_indexes}"
        )

    event_lines = []
    for item, (analysis, plan) in zip(generated.events, event_units):
        event_text = GeneratedStrokeText(
            text=normalize_exclamation_emphasis(
                ensure_uncertainty_wording(
                    item.text,
                    confidence_band=analysis.current_stroke.confidence_band,
                ),
                allow_exclamation=(
                    analysis.current_stroke.confidence_band == "reliable"
                    and analysis.speaking_score >= 0.9
                ),
            ),
            source_fact_ids=item.source_fact_ids,
        )
        try:
            validate_stroke_commentary(
                generated=event_text,
                analysis=analysis,
                plan=plan,
                score=fact.score,
            )
        except CommentaryLanguageSafetyError:
            event_text = _safe_stroke_fallback(
                analysis=analysis,
                plan=plan,
                player_names=player_names,
            )
            try:
                validate_stroke_commentary(
                    generated=event_text,
                    analysis=analysis,
                    plan=plan,
                    score=fact.score,
                )
            except CommentaryValidationError as exc:
                raise RallyBatchGenerationError(
                    f"stroke {analysis.stroke_index} fallback: {exc}"
                ) from exc
        except CommentaryValidationError as exc:
            raise RallyBatchGenerationError(
                f"stroke {analysis.stroke_index}: {exc}"
            ) from exc
        event_lines.append(
            StrokeCommentaryLine(
                segment_index=fact.segment_index,
                stroke_index=analysis.stroke_index,
                frame=analysis.frame,
                time_sec=analysis.time_sec,
                text=event_text.text,
                source_fact_ids=event_text.source_fact_ids,
            )
        )

    summary = generated.summary
    if summary_plan.should_comment and summary is None:
        raise RallyBatchGenerationError("rally batch omitted the planned summary")
    if not summary_plan.should_comment and summary is not None:
        raise RallyBatchGenerationError("rally batch returned an unplanned summary")
    if summary is not None:
        summary = GeneratedCommentary(
            segment_index=summary.segment_index,
            text=normalize_exclamation_emphasis(
                summary.text,
                allow_exclamation=commentary_allows_exclamation(scored),
            ),
            source_fact_ids=summary.source_fact_ids,
        )
        try:
            validate_commentary(
                generated=summary,
                scored=scored,
                plan=summary_plan,
                analysis=rally_analysis,
            )
        except CommentaryLanguageSafetyError:
            summary = _safe_summary_fallback(
                scored=scored,
                analysis=rally_analysis,
                allowed_fact_ids=summary_plan.allowed_fact_ids,
            )
            try:
                validate_commentary(
                    generated=summary,
                    scored=scored,
                    plan=summary_plan,
                    analysis=rally_analysis,
                )
            except CommentaryValidationError as exc:
                raise RallyBatchGenerationError(
                    f"summary fallback: {exc}"
                ) from exc
        except CommentaryValidationError as exc:
            raise RallyBatchGenerationError(f"summary: {exc}") from exc

    return RallyCommentaryBundle(
        segment_index=fact.segment_index,
        events=event_lines,
        summary=summary,
    )

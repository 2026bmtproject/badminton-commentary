"""Generate selected-clip commentary with Fake or Gemini provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.config import load_config
from badminton_commentary.generation.batch import generate_commentaries
from badminton_commentary.generation.event_batch import generate_event_driven_commentary
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.providers import FakeProvider, GeminiProvider, LLMProvider
from badminton_commentary.schemas import (
    CommentaryOutput,
    EventDrivenCommentaryOutput,
    GeneratedCommentary,
    GeneratedStrokeText,
    RallyFactsOutput,
    ScoredRallyFact,
    StrokeEventAnalysis,
    StrokeEventPlan,
)


CLIPS_ROOT = Path("fixtures/development/TTYvsASY/selected_clips")
GROUPS = ("seg0039-0043", "seg0052-0056", "seg0140-0144")
PLAYER_NAMES = {"a": "戴資穎", "b": "安洗瑩"}


def fake_response(scored: ScoredRallyFact) -> str:
    fact = scored.fact
    analysis = analyze_rally(fact)
    plan = plan_commentary(scored, analysis)
    selected_pattern = next(
        (
            pattern
            for pattern in analysis.patterns
            if pattern.fact_id in plan.allowed_fact_ids
        ),
        None,
    )
    if selected_pattern is not None:
        fact_id = selected_pattern.fact_id
        text = f"這段來回中，{selected_pattern.commentary_hint}。"
    else:
        fact_id = f"rally:{fact.segment_index}:score"
        text = f"戴資穎與安洗瑩目前戰成{fact.score.a}比{fact.score.b}。"
    return GeneratedCommentary(
        segment_index=fact.segment_index,
        text=text,
        source_fact_ids=[fact_id],
    ).model_dump_json()


def fake_event_response(
    analysis: StrokeEventAnalysis,
    plan: StrokeEventPlan,
) -> str:
    selected_local = next(
        (
            item
            for item in analysis.local_facts
            if item.fact_id in plan.allowed_fact_ids
        ),
        None,
    )
    if selected_local is not None:
        text = f"{selected_local.commentary_hint}。"
        source_fact_ids = [
            selected_local.fact_id,
            *selected_local.supporting_fact_ids,
        ]
    else:
        player = PLAYER_NAMES[analysis.current_stroke.player]
        stroke_text = {
            "殺球": f"{player}殺球進攻！",
            "撲球": f"{player}在網前撲球！",
            "平快球": f"{player}以平快球回擊。",
            "切球": f"{player}把球放短。",
            "小球": f"{player}把球送到網前。",
            "挑球": f"{player}把球挑高。",
            "高遠球": f"{player}拉出高遠球。",
        }.get(
            analysis.current_stroke.stroke_type,
            f"{player}處理這一拍。",
        )
        text = stroke_text
        source_fact_ids = [analysis.current_stroke.fact_id]
    return GeneratedStrokeText(
        text=text,
        source_fact_ids=source_fact_ids,
    ).model_dump_json()


def generate_group(
    group: str,
    *,
    provider_name: str,
    gemini_provider: LLMProvider | None,
) -> CommentaryOutput:
    facts_path = CLIPS_ROOT / group / "rally_facts.json"
    facts = RallyFactsOutput.model_validate_json(
        facts_path.read_text(encoding="utf-8")
    )
    def provider_factory(scored: ScoredRallyFact) -> LLMProvider:
        provider = (
            FakeProvider(response=fake_response(scored))
            if provider_name == "fake"
            else gemini_provider
        )
        if provider is None:
            raise ValueError("Gemini provider is not configured")
        return provider

    return generate_commentaries(
        scored_rallies=facts.rallies,
        provider_factory=provider_factory,
        player_names=PLAYER_NAMES,
    )


def generate_event_group(
    group: str,
    *,
    provider_name: str,
    gemini_provider: LLMProvider | None,
) -> EventDrivenCommentaryOutput:
    facts_path = CLIPS_ROOT / group / "rally_facts.json"
    facts = RallyFactsOutput.model_validate_json(
        facts_path.read_text(encoding="utf-8")
    )

    def event_provider_factory(scored, analysis, plan):
        if provider_name == "fake":
            return FakeProvider(response=fake_event_response(analysis, plan))
        if gemini_provider is None:
            raise ValueError("Gemini provider is not configured")
        return gemini_provider

    def summary_provider_factory(scored):
        if provider_name == "fake":
            return FakeProvider(response=fake_response(scored))
        if gemini_provider is None:
            raise ValueError("Gemini provider is not configured")
        return gemini_provider

    return generate_event_driven_commentary(
        scored_rallies=facts.rallies,
        event_provider_factory=event_provider_factory,
        summary_provider_factory=summary_provider_factory,
        player_names=PLAYER_NAMES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("fake", "gemini"), default="fake")
    parser.add_argument("--config", default="config.yaml.example")
    parser.add_argument(
        "--mode",
        choices=("event-driven", "summary"),
        default="event-driven",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gemini_provider = None
    if args.provider == "gemini":
        config = load_config(args.config)
        gemini_provider = GeminiProvider.from_config(config.provider.gemini)

    for group in GROUPS:
        if args.mode == "event-driven":
            output = generate_event_group(
                group,
                provider_name=args.provider,
                gemini_provider=gemini_provider,
            )
            filename = f"commentary_{args.provider}_event_driven.json"
        else:
            output = generate_group(
                group,
                provider_name=args.provider,
                gemini_provider=gemini_provider,
            )
            filename = f"commentary_{args.provider}.json"
        output_path = CLIPS_ROOT / group / filename
        output_path.write_text(
            json.dumps(output.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

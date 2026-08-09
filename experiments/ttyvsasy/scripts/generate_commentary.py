"""Generate selected-clip commentary with Fake or Gemini provider."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from badminton_commentary.analysis import analyze_stroke_events
from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.config import load_config
from badminton_commentary.generation.batch import generate_commentaries
from badminton_commentary.generation.event_planner import plan_stroke_commentary
from badminton_commentary.generation.planner import (
    plan_commentary,
    plan_selected_rally_summary,
)
from badminton_commentary.providers import (
    FakeProvider,
    GeminiProvider,
    LLMProvider,
    ProviderTiming,
    TimedProvider,
    TimingStats,
)
from badminton_commentary.services import RallyCommentaryService
from badminton_commentary.schemas import (
    CommentaryOutput,
    EventDrivenCommentaryOutput,
    GeneratedCommentary,
    GeneratedRallyTextBatch,
    GeneratedStrokeBatchItem,
    GeneratedStrokeText,
    RallyFactsOutput,
    ScoredRallyFact,
    StrokeEventAnalysis,
    StrokeEventPlan,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "ttyvsasy"
GROUPS = ("seg0039-0043", "seg0052-0056", "seg0140-0144")
PLAYER_NAMES = {"a": "戴資穎", "b": "安洗瑩"}


def report_provider_timing(timing: ProviderTiming) -> None:
    status = "ok" if timing.succeeded else "failed"
    print(
        f"[timing] {timing.label}: {timing.seconds:.3f}s ({status})",
        file=sys.stderr,
    )


def with_timing(
    provider: LLMProvider,
    *,
    label: str,
    stats: TimingStats | None,
) -> LLMProvider:
    if stats is None:
        return provider
    return TimedProvider(
        provider,
        label=label,
        stats=stats,
        reporter=report_provider_timing,
    )


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
        if analysis.current_stroke.confidence_band != "reliable":
            stroke_text = (
                f"辨識結果顯示，{player}這拍可能是"
                f"{analysis.current_stroke.stroke_type}。"
            )
        else:
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


def fake_rally_batch_response(scored: ScoredRallyFact) -> str:
    event_items = []
    for analysis in analyze_stroke_events(
        scored.fact,
        include_all_strokes=True,
    ):
        plan = plan_stroke_commentary(
            analysis,
            force_commentary=True,
        )
        generated = GeneratedStrokeText.model_validate_json(
            fake_event_response(analysis, plan)
        )
        event_items.append(
            GeneratedStrokeBatchItem(
                stroke_index=analysis.stroke_index,
                text=generated.text,
                source_fact_ids=generated.source_fact_ids,
            )
        )
    summary_plan = plan_selected_rally_summary(
        scored.fact,
        analyze_rally(scored.fact),
    )
    summary = (
        GeneratedCommentary.model_validate_json(fake_response(scored))
        if summary_plan.should_comment
        else None
    )
    return GeneratedRallyTextBatch(
        segment_index=scored.fact.segment_index,
        events=event_items,
        summary=summary,
    ).model_dump_json()


def generate_group(
    group: str,
    *,
    provider_name: str,
    gemini_provider: LLMProvider | None,
    timing_stats: TimingStats | None = None,
) -> CommentaryOutput:
    facts_path = OUTPUTS_ROOT / group / "rally_facts.json"
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
        return with_timing(
            provider,
            label=f"{group}/rally:{scored.fact.segment_index}/{provider_name}",
            stats=timing_stats,
        )

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
    timing_stats: TimingStats | None = None,
) -> EventDrivenCommentaryOutput:
    facts_path = OUTPUTS_ROOT / group / "rally_facts.json"
    facts = RallyFactsOutput.model_validate_json(
        facts_path.read_text(encoding="utf-8")
    )

    bundles = []
    for scored in facts.rallies:
        provider = (
            FakeProvider(response=fake_rally_batch_response(scored))
            if provider_name == "fake"
            else gemini_provider
        )
        if provider is None:
            raise ValueError("Gemini provider is not configured")
        timed_provider = with_timing(
            provider,
            label=f"{group}/rally:{scored.fact.segment_index}/{provider_name}",
            stats=timing_stats,
        )
        bundles.append(
            RallyCommentaryService(
                provider=timed_provider,
                player_names=PLAYER_NAMES,
            ).generate(
                rally_fact=scored.fact,
            )
        )
    return EventDrivenCommentaryOutput(rallies=bundles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("fake", "gemini"), default="fake")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml.example")
    parser.add_argument(
        "--mode",
        choices=("event-driven", "summary"),
        default="event-driven",
    )
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="Disable per-rally, per-group, and total timing output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timing_stats = None if args.no_timing else TimingStats()
    total_started = time.perf_counter()
    try:
        gemini_provider = None
        if args.provider == "gemini":
            config = load_config(args.config)
            gemini_provider = GeminiProvider.from_config(config.provider.gemini)

        for group in GROUPS:
            group_started = time.perf_counter()
            try:
                if args.mode == "event-driven":
                    output = generate_event_group(
                        group,
                        provider_name=args.provider,
                        gemini_provider=gemini_provider,
                        timing_stats=timing_stats,
                    )
                    filename = f"commentary_{args.provider}_event_driven.json"
                else:
                    output = generate_group(
                        group,
                        provider_name=args.provider,
                        gemini_provider=gemini_provider,
                        timing_stats=timing_stats,
                    )
                    filename = f"commentary_{args.provider}.json"
                output_root = OUTPUTS_ROOT / group
                output_root.mkdir(parents=True, exist_ok=True)
                output_path = output_root / filename
                output_path.write_text(
                    json.dumps(output.model_dump(), ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
            finally:
                if timing_stats is not None:
                    print(
                        f"[timing] {group}/total: "
                        f"{time.perf_counter() - group_started:.3f}s",
                        file=sys.stderr,
                    )
    finally:
        if timing_stats is not None:
            print(
                "[timing] all/total: "
                f"{time.perf_counter() - total_started:.3f}s; "
                f"provider_calls={timing_stats.call_count}; "
                f"provider_time={timing_stats.total_seconds:.3f}s",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()

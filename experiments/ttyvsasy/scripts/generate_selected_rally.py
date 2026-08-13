"""Generate one TTYvsASY rally directly from full-match stage artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    build_rally_fact_from_stages,
    read_upstream_stages,
)
from badminton_commentary.analysis import (
    analyze_rally,
    analyze_stroke_events,
    analyze_tactical_facts,
)
from badminton_commentary.config import load_config
from badminton_commentary.facts import (
    CompactRallyFacts,
    GeneratedTacticalAnalysis,
    GeneratedTacticalFact,
    build_compact_rally_facts,
)
from badminton_commentary.generation.planner import plan_selected_rally_summary
from badminton_commentary.providers import FakeProvider, GeminiProvider, LLMProvider
from badminton_commentary.schemas import (
    GeneratedCommentary,
    GeneratedRallyTextBatch,
    GeneratedStrokeBatchItem,
    RallyFact,
)
from badminton_commentary.services import RallyCommentaryService


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = EXPERIMENT_ROOT / "workspace" / "stages"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ttyvsasy" / "from_stages"
PLAYER_NAMES = {"a": "戴資穎", "b": "安洗瑩"}


def _fake_response(fact: RallyFact) -> str:
    event_items = []
    for analysis in analyze_stroke_events(fact, include_all_strokes=True):
        if analysis.local_facts:
            local = analysis.local_facts[0]
            text = f"{local.commentary_hint}。"
            source_fact_ids = [local.fact_id, *local.supporting_fact_ids]
        else:
            player = PLAYER_NAMES[analysis.current_stroke.player]
            if analysis.current_stroke.confidence_band == "reliable":
                text = f"{player}打出{analysis.current_stroke.stroke_type}。"
            else:
                text = (
                    f"辨識結果顯示，{player}這拍可能是"
                    f"{analysis.current_stroke.stroke_type}。"
                )
            source_fact_ids = [analysis.current_stroke.fact_id]
        event_items.append(
            GeneratedStrokeBatchItem(
                stroke_index=analysis.stroke_index,
                text=text,
                source_fact_ids=source_fact_ids,
            )
        )

    analysis = analyze_rally(fact)
    summary_plan = plan_selected_rally_summary(fact, analysis)
    pattern = next(
        (
            item
            for item in analysis.patterns
            if item.fact_id in summary_plan.allowed_fact_ids
        ),
        None,
    )
    if pattern is not None:
        summary_text = f"{pattern.commentary_hint}。"
        summary_fact_id = pattern.fact_id
    elif fact.score.a is not None and fact.score.b is not None:
        summary_text = f"目前比分是{fact.score.a}比{fact.score.b}。"
        summary_fact_id = f"rally:{fact.segment_index}:score"
    elif fact.events:
        summary_text = f"這回合共有{fact.rally_length}拍。"
        summary_fact_id = f"rally:{fact.segment_index}:length"
    else:
        summary_text = "這個片段具有精彩片段標記。"
        summary_fact_id = f"rally:{fact.segment_index}:highlight"

    return GeneratedRallyTextBatch(
        segment_index=fact.segment_index,
        events=event_items,
        summary=GeneratedCommentary(
            segment_index=fact.segment_index,
            text=summary_text,
            source_fact_ids=[summary_fact_id],
        ),
    ).model_dump_json()


def _fake_tactical_response(compact: CompactRallyFacts) -> str:
    if len(compact.events) < 2:
        return GeneratedTacticalAnalysis(
            segment_index=compact.segment_index,
            facts=[],
        ).model_dump_json()
    first, second = compact.events[:2]
    players = list(
        dict.fromkeys(
            player for player in (first.player, second.player) if player is not None
        )
    )
    return GeneratedTacticalAnalysis(
        segment_index=compact.segment_index,
        facts=[
            GeneratedTacticalFact(
                pattern_type="notable_stroke_sequence",
                description="這兩拍形成一段連續球路銜接。",
                confidence=0.8,
                salience=0.6,
                start_event_index=first.event_index,
                end_event_index=second.event_index,
                players=players,
                evidence_fact_ids=[first.fact_id, second.fact_id],
                limitations=["測試用固定戰術分析"],
            )
        ],
    ).model_dump_json()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one TTYvsASY rally directly from full stages.",
    )
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--top-player", choices=("a", "b"), required=True)
    parser.add_argument("--bottom-player", choices=("a", "b"), required=True)
    parser.add_argument("--provider", choices=("fake", "gemini"), default="fake")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fact-output",
        type=Path,
        help="RallyFact JSON path; defaults beside the commentary output.",
    )
    parser.add_argument(
        "--compact-output",
        type=Path,
        help="CompactRallyFacts JSON path; defaults beside commentary output.",
    )
    parser.add_argument(
        "--tactical-output",
        type=Path,
        help="TacticalFact JSON path; defaults beside commentary output.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    output_path = args.output or (
        OUTPUT_ROOT
        / f"seg{args.segment_index:04d}"
        / f"commentary_{args.provider}.json"
    )
    fact_output_path = args.fact_output or output_path.with_name("rally_fact.json")
    compact_output_path = args.compact_output or output_path.with_name(
        "compact_facts.json"
    )
    tactical_output_path = args.tactical_output or output_path.with_name(
        "tactical_facts.json"
    )
    existing_outputs = [
        path
        for path in (
            output_path,
            fact_output_path,
            compact_output_path,
            tactical_output_path,
        )
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    mapping = CourtPositionToPlayer(
        top=args.top_player,
        bottom=args.bottom_player,
    )
    stages = read_upstream_stages(
        StagePaths.from_stage_root(STAGE_ROOT),
        segment_index=args.segment_index,
    )
    fact = build_rally_fact_from_stages(
        stages=stages,
        segment_index=args.segment_index,
        court_position_to_player=mapping,
    )
    fact_output_path.parent.mkdir(parents=True, exist_ok=True)
    fact_output_path.write_text(
        fact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    compact = build_compact_rally_facts(
        stages=stages,
        segment_index=args.segment_index,
        court_position_to_player=mapping,
    )
    compact_output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_output_path.write_text(
        compact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    config = load_config(args.config) if args.provider == "gemini" else None
    provider: LLMProvider
    tactical_provider: LLMProvider
    if args.provider == "fake":
        provider = FakeProvider(response=_fake_response(fact))
        tactical_provider = FakeProvider(response=_fake_tactical_response(compact))
    else:
        assert config is not None
        provider = GeminiProvider.from_config(
            config.provider.gemini
        )
        tactical_provider = GeminiProvider.from_config(
            config.provider.gemini,
            model_override=config.tactical_analyzer.model,
            fallback_models=config.tactical_analyzer.fallback_models,
        )

    tactical = analyze_tactical_facts(
        provider=tactical_provider,
        compact_facts=compact,
        max_facts=config.tactical_analyzer.max_facts if config is not None else 5,
    )
    tactical_output_path.parent.mkdir(parents=True, exist_ok=True)
    tactical_output_path.write_text(
        tactical.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    bundle = RallyCommentaryService(
        provider=provider,
        player_names=PLAYER_NAMES,
    ).generate_from_stages(
        stages=stages,
        segment_index=args.segment_index,
        court_position_to_player=mapping,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output: {output_path.resolve()}")
    print(f"rally_fact: {fact_output_path.resolve()}")
    print(f"compact_facts: {compact_output_path.resolve()}")
    print(f"tactical_facts: {tactical_output_path.resolve()}")
    print(f"segment: {fact.segment_index}; strokes: {fact.rally_length}")
    print(
        "compact: "
        f"pose={sum(item.pose is not None for item in compact.events)}, "
        f"court={sum(item.court_position is not None for item in compact.events)}, "
        f"shuttle={sum(item.shuttle_path is not None for item in compact.events)}"
    )
    if compact.warnings:
        print(f"compact_warnings: {', '.join(compact.warnings)}")
    print(f"tactical_patterns: {len(tactical.facts)}")
    if tactical.provider_model is not None:
        print(f"tactical_model: {tactical.provider_model}")
    if tactical.warnings:
        print(f"tactical_warnings: {', '.join(tactical.warnings)}")
    if isinstance(provider, FakeProvider):
        assert isinstance(tactical_provider, FakeProvider)
        print(f"tactical_provider_calls: {len(tactical_provider.calls)}")
        print(f"commentary_provider_calls: {len(provider.calls)}")
    print(f"elapsed: {time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()

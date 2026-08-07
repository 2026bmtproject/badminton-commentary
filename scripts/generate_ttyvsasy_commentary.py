"""Generate selected-clip commentary with Fake or Gemini provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from badminton_commentary.config import load_config
from badminton_commentary.generation.batch import generate_commentaries
from badminton_commentary.providers import FakeProvider, GeminiProvider, LLMProvider
from badminton_commentary.schemas import (
    CommentaryOutput,
    GeneratedCommentary,
    RallyFactsOutput,
    ScoredRallyFact,
)


CLIPS_ROOT = Path("fixtures/development/TTYvsASY/selected_clips")
GROUPS = ("seg0039-0043", "seg0052-0056", "seg0140-0144")
PLAYER_NAMES = {"a": "戴資穎", "b": "安洗瑩"}


def fake_response(scored: ScoredRallyFact) -> str:
    fact = scored.fact
    fact_id = f"rally:{fact.segment_index}:score"
    text = f"戴資穎與安洗瑩目前戰成{fact.score.a}比{fact.score.b}。"
    return GeneratedCommentary(
        segment_index=fact.segment_index,
        text=text,
        source_fact_ids=[fact_id],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("fake", "gemini"), default="fake")
    parser.add_argument("--config", default="config.yaml.example")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gemini_provider = None
    if args.provider == "gemini":
        config = load_config(args.config)
        gemini_provider = GeminiProvider.from_config(config.provider.gemini)

    for group in GROUPS:
        output = generate_group(
            group,
            provider_name=args.provider,
            gemini_provider=gemini_provider,
        )
        output_path = CLIPS_ROOT / group / f"commentary_{args.provider}.json"
        output_path.write_text(
            json.dumps(output.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

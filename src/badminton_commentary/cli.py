from __future__ import annotations

import argparse
from pathlib import Path

from badminton_commentary.config import load_config
from badminton_commentary.providers import FakeProvider, GeminiProvider, LLMProvider
from badminton_commentary.schemas import ImportanceResult, RallyFact
from badminton_commentary.services import RallyCommentaryService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate structured commentary for one RallyFact JSON.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("gemini", "fake"), default="gemini")
    parser.add_argument("--config", type=Path, default=Path("config.yaml.example"))
    parser.add_argument(
        "--fake-response",
        type=Path,
        help="Validated batch response JSON required when --provider=fake.",
    )
    parser.add_argument("--player-a")
    parser.add_argument("--player-b")
    parser.add_argument("--importance", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _provider(args: argparse.Namespace) -> LLMProvider:
    if args.provider == "fake":
        if args.fake_response is None:
            raise ValueError("--fake-response is required when --provider=fake")
        return FakeProvider(
            response=args.fake_response.read_text(encoding="utf-8"),
        )
    config = load_config(args.config)
    return GeminiProvider.from_config(config.provider.gemini)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite: {args.output}"
        )
    rally_fact = RallyFact.model_validate_json(args.input.read_text(encoding="utf-8"))
    importance = (
        ImportanceResult(score=args.importance, reasons=["cli_override"])
        if args.importance is not None
        else None
    )
    player_names = None
    if args.player_a is not None or args.player_b is not None:
        if args.player_a is None or args.player_b is None:
            raise ValueError("--player-a and --player-b must be provided together")
        player_names = {"a": args.player_a, "b": args.player_b}

    commentary = RallyCommentaryService(
        provider=_provider(args),
        player_names=player_names,
    ).generate(
        rally_fact=rally_fact,
        importance=importance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        commentary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

"""Build deterministic rally facts for the selected TTYvsASY clips."""

from __future__ import annotations

from pathlib import Path

from badminton_commentary.analysis.fact_builder import build_rally_facts
from badminton_commentary.analysis.importance import score_importance
from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.schemas import (
    CommentaryPlansOutput,
    EventsInput,
    HighlightsInput,
    RallyAnalysesOutput,
    RallyFactsOutput,
    ScoredRallyFact,
    ScoresInput,
    SegmentsInput,
    StrokesInput,
)


CLIPS_ROOT = Path("fixtures/development/TTYvsASY/selected_clips")
GROUPS = ("seg0039-0043", "seg0052-0056", "seg0140-0144")


def load_model(path: Path, model_type):
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def build_clip(group: str) -> RallyFactsOutput:
    input_root = CLIPS_ROOT / group / "commentary_input"
    facts = build_rally_facts(
        segments=load_model(input_root / "segments.json", SegmentsInput),
        scores=load_model(input_root / "scores.json", ScoresInput),
        events=load_model(input_root / "events.json", EventsInput),
        strokes=load_model(input_root / "strokes.json", StrokesInput),
        highlights=load_model(input_root / "highlights.json", HighlightsInput),
    )
    return RallyFactsOutput(
        rallies=[
            ScoredRallyFact(fact=fact, importance=score_importance(fact))
            for fact in facts
        ]
    )


def main() -> None:
    for group in GROUPS:
        output = build_clip(group)
        output_path = CLIPS_ROOT / group / "rally_facts.json"
        output_path.write_text(
            output.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        analyses = RallyAnalysesOutput(
            analyses=[analyze_rally(scored.fact) for scored in output.rallies]
        )
        (CLIPS_ROOT / group / "rally_analyses.json").write_text(
            analyses.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        plans = CommentaryPlansOutput(
            plans=[
                plan_commentary(scored, analysis)
                for scored, analysis in zip(output.rallies, analyses.analyses)
            ]
        )
        (CLIPS_ROOT / group / "commentary_plans.json").write_text(
            plans.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

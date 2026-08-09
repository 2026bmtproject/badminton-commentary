"""Build deterministic rally facts for the selected TTYvsASY clips."""

from __future__ import annotations

import json
from pathlib import Path

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    build_rally_fact_from_stages,
    read_upstream_stages,
)
from badminton_commentary.analysis.importance import score_importance
from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.schemas import (
    CommentaryPlansOutput,
    RallyAnalysesOutput,
    RallyFactsOutput,
    ScoredRallyFact,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CLIPS_ROOT = EXPERIMENT_ROOT / "workspace" / "selected_clips"
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "ttyvsasy"
GROUPS = ("seg0039-0043", "seg0052-0056", "seg0140-0144")


def build_clip(group: str) -> RallyFactsOutput:
    clip_root = CLIPS_ROOT / group
    stages = read_upstream_stages(
        StagePaths.from_stage_root(clip_root / "stages")
    )
    mapping_payload = json.loads(
        (clip_root / "player_mapping.json").read_text(encoding="utf-8")
    )
    mapping = CourtPositionToPlayer.model_validate(
        mapping_payload["court_position_to_player"]
    )
    facts = [
        build_rally_fact_from_stages(
            stages=stages,
            segment_index=segment_index,
            court_position_to_player=mapping,
        )
        for segment_index in range(len(stages.match_segmentation.segments))
    ]
    return RallyFactsOutput(
        rallies=[
            ScoredRallyFact(fact=fact, importance=score_importance(fact))
            for fact in facts
        ]
    )


def main() -> None:
    for group in GROUPS:
        output = build_clip(group)
        output_root = OUTPUTS_ROOT / group
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / "rally_facts.json"
        output_path.write_text(
            output.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        analyses = RallyAnalysesOutput(
            analyses=[analyze_rally(scored.fact) for scored in output.rallies]
        )
        (output_root / "rally_analyses.json").write_text(
            analyses.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        plans = CommentaryPlansOutput(
            plans=[
                plan_commentary(scored, analysis)
                for scored, analysis in zip(output.rallies, analyses.analyses)
            ]
        )
        (output_root / "commentary_plans.json").write_text(
            plans.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

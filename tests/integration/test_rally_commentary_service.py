import inspect
import json
from pathlib import Path

import pytest

from badminton_commentary.analysis import analyze_rally, analyze_stroke_events
from badminton_commentary.analysis.importance import score_importance
from badminton_commentary.cli import main as cli_main
from badminton_commentary.generation.event_planner import plan_stroke_commentary
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.generation.rally_batch_commentator import (
    RallyBatchGenerationError,
)
from badminton_commentary.providers import FakeProvider
from badminton_commentary.schemas import (
    GeneratedCommentary,
    GeneratedRallyTextBatch,
    GeneratedStrokeBatchItem,
    RallyFact,
    RallyFactEvent,
    RallyScore,
    ScoredRallyFact,
)
from badminton_commentary.services import RallyCommentaryService
from badminton_commentary.services import rally_commentary as service_module


def rally_fact() -> RallyFact:
    return RallyFact(
        segment_index=37,
        game_index=1,
        start_sec=0,
        end_sec=4,
        duration_sec=4,
        score=RallyScore(a=20, b=20),
        server="a",
        events=[
            RallyFactEvent(
                event_index=index,
                frame=index * 30,
                time_sec=float(index),
                player=player,
                stroke_type=stroke_type,
                stroke_confidence=0.9,
            )
            for index, player, stroke_type in (
                (0, "a", "發球"),
                (1, "b", "小球"),
                (2, "a", "挑球"),
                (3, "b", "殺球"),
            )
        ],
        rally_length=4,
        highlight_score=None,
    )


def valid_response(fact: RallyFact) -> str:
    importance = score_importance(fact)
    scored = ScoredRallyFact(fact=fact, importance=importance)
    event_items = []
    for analysis in analyze_stroke_events(fact):
        plan = plan_stroke_commentary(
            analysis,
            importance_score=importance.score,
        )
        if not plan.should_comment:
            continue
        local = analysis.local_facts[0]
        event_items.append(
            GeneratedStrokeBatchItem(
                stroke_index=analysis.stroke_index,
                text=f"{local.commentary_hint}。",
                source_fact_ids=[local.fact_id, *local.supporting_fact_ids],
            )
        )

    analysis = analyze_rally(fact)
    summary_plan = plan_commentary(scored, analysis)
    pattern = next(
        item
        for item in analysis.patterns
        if item.fact_id in summary_plan.allowed_fact_ids
    )
    return GeneratedRallyTextBatch(
        segment_index=fact.segment_index,
        events=event_items,
        summary=GeneratedCommentary(
            segment_index=fact.segment_index,
            text=f"{pattern.commentary_hint}。",
            source_fact_ids=[pattern.fact_id],
        ),
    ).model_dump_json()


def test_service_generates_one_complete_rally_with_one_provider_call():
    fact = rally_fact()
    provider = FakeProvider(response=valid_response(fact))

    result = RallyCommentaryService(
        provider=provider,
        player_names={"a": "甲", "b": "乙"},
    ).generate(rally_fact=fact)

    assert result.segment_index == 37
    assert result.events
    assert result.summary is not None
    assert len(provider.calls) == 1
    assert all(event.source_fact_ids for event in result.events)
    assert result.summary.source_fact_ids


def test_user_selected_low_importance_rally_still_uses_one_provider_call():
    fact = RallyFact(
        segment_index=38,
        game_index=1,
        start_sec=0,
        end_sec=1,
        duration_sec=1,
        score=RallyScore(a=3, b=1),
        server="a",
        events=[
            RallyFactEvent(
                event_index=0,
                frame=3,
                time_sec=0.1,
                player="a",
                stroke_type="發球",
                stroke_confidence=0.99,
            )
        ],
        rally_length=1,
        highlight_score=None,
    )
    response = GeneratedRallyTextBatch(
        segment_index=38,
        events=[],
        summary=GeneratedCommentary(
            segment_index=38,
            text="目前比分是 3 比 1。",
            source_fact_ids=["rally:38:score"],
        ),
    ).model_dump_json()
    provider = FakeProvider(response=response)

    result = RallyCommentaryService(provider=provider).generate(rally_fact=fact)

    assert result.events == []
    assert result.summary is not None
    assert result.summary.source_fact_ids == ["rally:38:score"]
    assert len(provider.calls) == 1


def test_service_rejects_rally_without_any_grounded_commentary_fact():
    fact = RallyFact(
        segment_index=39,
        game_index=None,
        start_sec=0,
        end_sec=1,
        duration_sec=1,
        score=RallyScore(a=None, b=None),
        server=None,
        events=[],
        rally_length=0,
        highlight_score=None,
    )
    provider = FakeProvider(response="not used")

    with pytest.raises(ValueError, match="no grounded commentary facts"):
        RallyCommentaryService(provider=provider).generate(rally_fact=fact)

    assert provider.calls == []


def test_service_still_runs_provenance_validation():
    fact = rally_fact()
    payload = json.loads(valid_response(fact))
    payload["events"][0]["source_fact_ids"] = ["rally:37:stroke:999"]

    with pytest.raises(RallyBatchGenerationError, match="disallowed fact ids"):
        RallyCommentaryService(
            provider=FakeProvider(response=json.dumps(payload, ensure_ascii=False))
        ).generate(rally_fact=fact)


def test_service_has_no_experiment_or_subtitle_runtime_dependency():
    source = inspect.getsource(service_module)

    assert "TTYvsASY" not in source
    assert "fixtures" not in source
    assert "subtitles" not in source
    assert "ffmpeg" not in source.lower()


def test_generic_cli_generates_one_rally_offline(tmp_path):
    fact = rally_fact()
    input_path = tmp_path / "rally_fact.json"
    response_path = tmp_path / "response.json"
    output_path = tmp_path / "commentary.json"
    input_path.write_text(fact.model_dump_json(), encoding="utf-8")
    response_path.write_text(valid_response(fact), encoding="utf-8")

    cli_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--provider",
            "fake",
            "--fake-response",
            str(response_path),
        ]
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["segment_index"] == 37


def test_ttyvsasy_experiment_consumes_production_service():
    repo_root = Path(__file__).parents[2]
    script = (
        repo_root
        / "experiments"
        / "ttyvsasy"
        / "scripts"
        / "generate_commentary.py"
    )
    source = script.read_text(encoding="utf-8")

    assert "from badminton_commentary.services import RallyCommentaryService" in source
    assert "generate_batched_event_driven_commentary" not in source

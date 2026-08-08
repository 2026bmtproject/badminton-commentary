import json

import pytest

from badminton_commentary.analysis import analyze_rally, analyze_stroke_events
from badminton_commentary.generation.event_planner import plan_stroke_commentary
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.generation.rally_batch_commentator import (
    RallyBatchGenerationError,
    generate_rally_commentary_batch,
)
from badminton_commentary.providers import FakeProvider
from badminton_commentary.schemas import (
    GeneratedCommentary,
    GeneratedRallyTextBatch,
    GeneratedStrokeBatchItem,
    ImportanceResult,
    RallyFact,
    RallyFactEvent,
    RallyScore,
    ScoredRallyFact,
)


def scored_fact() -> ScoredRallyFact:
    events = [
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
    ]
    return ScoredRallyFact(
        fact=RallyFact(
            segment_index=7,
            game_index=None,
            start_sec=0,
            end_sec=4,
            duration_sec=4,
            score=RallyScore(a=20, b=20),
            server="a",
            events=events,
            rally_length=4,
            highlight_score=None,
        ),
        importance=ImportanceResult(score=0.5, reasons=["close_score"]),
    )


def valid_response(scored: ScoredRallyFact) -> str:
    event_items = []
    for analysis in analyze_stroke_events(scored.fact):
        plan = plan_stroke_commentary(analysis)
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

    rally_analysis = analyze_rally(scored.fact)
    summary_plan = plan_commentary(scored, rally_analysis)
    pattern = next(
        item
        for item in rally_analysis.patterns
        if item.fact_id in summary_plan.allowed_fact_ids
    )
    return GeneratedRallyTextBatch(
        segment_index=scored.fact.segment_index,
        events=event_items,
        summary=GeneratedCommentary(
            segment_index=scored.fact.segment_index,
            text=f"{pattern.commentary_hint}。",
            source_fact_ids=[pattern.fact_id],
        ),
    ).model_dump_json()


def test_one_provider_call_generates_all_events_and_summary():
    scored = scored_fact()
    provider = FakeProvider(response=valid_response(scored))

    bundle = generate_rally_commentary_batch(
        provider=provider,
        scored=scored,
        player_names={"a": "甲", "b": "乙"},
    )

    assert len(provider.calls) == 1
    assert [item.stroke_index for item in bundle.events] == [3]
    assert bundle.summary is not None
    payload = json.loads(provider.calls[0].user_prompt)
    assert len(payload["event_units"]) == 1
    assert payload["summary_unit"] is not None
    assert "previous_strokes" not in provider.calls[0].user_prompt


def test_fenced_batch_json_is_accepted():
    scored = scored_fact()
    response = f"```json\n{valid_response(scored)}\n```"

    bundle = generate_rally_commentary_batch(
        provider=FakeProvider(response=response),
        scored=scored,
    )

    assert bundle.segment_index == 7


def test_batch_must_return_exact_planned_stroke_order():
    scored = scored_fact()
    payload = json.loads(valid_response(scored))
    payload["events"][0]["stroke_index"] = 99

    with pytest.raises(RallyBatchGenerationError, match="planned order"):
        generate_rally_commentary_batch(
            provider=FakeProvider(response=json.dumps(payload, ensure_ascii=False)),
            scored=scored,
        )

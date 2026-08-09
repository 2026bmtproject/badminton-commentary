import json

import pytest

from badminton_commentary.analysis import analyze_rally, analyze_stroke_events
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
    for analysis in analyze_stroke_events(
        scored.fact,
        include_all_strokes=True,
    ):
        local = analysis.local_facts[0] if analysis.local_facts else None
        event_items.append(
            GeneratedStrokeBatchItem(
                stroke_index=analysis.stroke_index,
                text=(
                    f"{local.commentary_hint}。"
                    if local is not None
                    else f"這拍是{analysis.current_stroke.stroke_type}。"
                ),
                source_fact_ids=(
                    [local.fact_id, *local.supporting_fact_ids]
                    if local is not None
                    else [analysis.current_stroke.fact_id]
                ),
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
    assert [item.stroke_index for item in bundle.events] == [0, 1, 2, 3]
    assert bundle.summary is not None
    payload = json.loads(provider.calls[0].user_prompt)
    assert len(payload["event_units"]) == 4
    assert payload["all_stroke_fact_ids"][0] == "rally:7:stroke:0"
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


def test_unsafe_summary_wording_uses_grounded_fallback():
    scored = scored_fact()
    payload = json.loads(valid_response(scored))
    payload["summary"]["text"] = "靠著這個球路變化掌握主動。"

    bundle = generate_rally_commentary_batch(
        provider=FakeProvider(response=json.dumps(payload, ensure_ascii=False)),
        scored=scored,
    )

    assert bundle.summary is not None
    assert "掌握主動" not in bundle.summary.text
    assert bundle.summary.source_fact_ids[0].startswith("rally:7:pattern:")


def test_all_identified_strokes_include_serve_and_low_confidence_result():
    fact = RallyFact(
        segment_index=8,
        game_index=None,
        start_sec=0,
        end_sec=2,
        duration_sec=2,
        score=RallyScore(a=None, b=None),
        server="a",
        events=[
            RallyFactEvent(
                event_index=10,
                frame=30,
                time_sec=1,
                player="a",
                stroke_type="發球",
                stroke_confidence=0.99,
            ),
            RallyFactEvent(
                event_index=11,
                frame=45,
                time_sec=1.5,
                player="b",
                stroke_type="殺球",
                stroke_confidence=0.3,
            ),
        ],
        rally_length=2,
        highlight_score=None,
    )
    scored = ScoredRallyFact(
        fact=fact,
        importance=ImportanceResult(score=0, reasons=[]),
    )
    response = GeneratedRallyTextBatch(
        segment_index=8,
        events=[
            GeneratedStrokeBatchItem(
                stroke_index=10,
                text="A 靠著發球掌握主動。",
                source_fact_ids=["rally:8:stroke:10"],
            ),
            GeneratedStrokeBatchItem(
                stroke_index=11,
                text="B 打出殺球。",
                source_fact_ids=["rally:8:stroke:11"],
            ),
        ],
        summary=None,
    ).model_dump_json()
    provider = FakeProvider(response=response)

    bundle = generate_rally_commentary_batch(provider=provider, scored=scored)

    assert [item.stroke_index for item in bundle.events] == [10, 11]
    assert bundle.events[0].text == "player a打出發球。"
    assert bundle.events[1].text == "辨識結果顯示，B 打出殺球。"
    payload = json.loads(provider.calls[0].user_prompt)
    assert payload["all_stroke_fact_ids"] == [
        "rally:8:stroke:10",
        "rally:8:stroke:11",
    ]
    assert payload["prompt_version"] == "rally-batch-commentator-v2"


def test_unmapped_player_stroke_is_context_but_not_a_generated_event():
    fact = RallyFact(
        segment_index=9,
        game_index=None,
        start_sec=0,
        end_sec=2,
        duration_sec=2,
        score=RallyScore(a=None, b=None),
        server=None,
        events=[
            RallyFactEvent(
                event_index=20,
                frame=30,
                time_sec=1,
                player=None,
                stroke_type="發球",
                stroke_confidence=0.9,
            ),
            RallyFactEvent(
                event_index=21,
                frame=45,
                time_sec=1.5,
                player="b",
                stroke_type="平快球",
                stroke_confidence=0.9,
            ),
        ],
        rally_length=2,
        highlight_score=None,
    )
    scored = ScoredRallyFact(
        fact=fact,
        importance=ImportanceResult(score=0, reasons=[]),
    )
    response = GeneratedRallyTextBatch(
        segment_index=9,
        events=[
            GeneratedStrokeBatchItem(
                stroke_index=21,
                text="B 打出平快球。",
                source_fact_ids=["rally:9:stroke:21"],
            )
        ],
        summary=None,
    ).model_dump_json()
    provider = FakeProvider(response=response)

    bundle = generate_rally_commentary_batch(provider=provider, scored=scored)

    assert [item.stroke_index for item in bundle.events] == [21]
    payload = json.loads(provider.calls[0].user_prompt)
    assert payload["all_stroke_fact_ids"] == [
        "rally:9:stroke:20",
        "rally:9:stroke:21",
    ]
    assert payload["fact_catalog"]["rally:9:stroke:20"]["player"] is None

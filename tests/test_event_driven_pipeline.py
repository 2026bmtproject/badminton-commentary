import json

from badminton_commentary.analysis import analyze_rally
from badminton_commentary.generation.event_batch import (
    generate_event_driven_commentary,
)
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.providers import FakeProvider
from badminton_commentary.schemas import (
    GeneratedCommentary,
    ImportanceResult,
    RallyFact,
    RallyFactEvent,
    RallyScore,
    ScoredRallyFact,
)


def scored_fact():
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
            segment_index=0,
            game_index=None,
            start_sec=0,
            end_sec=4,
            duration_sec=4,
            score=RallyScore(a=20, b=20),
            server=None,
            events=events,
            rally_length=4,
            highlight_score=None,
        ),
        importance=ImportanceResult(score=0.5, reasons=["close_score"]),
    )


def event_provider_factory(scored, analysis, plan):
    local_ids = [fact_id for fact_id in plan.allowed_fact_ids if ":local:" in fact_id]
    if local_ids:
        local = next(item for item in analysis.local_facts if item.fact_id == local_ids[0])
        source_ids = [local.fact_id, *local.supporting_fact_ids]
        text = f"{local.commentary_hint}。"
    else:
        source_ids = [analysis.current_stroke.fact_id]
        text = f"{analysis.current_stroke.stroke_type}。"
    return FakeProvider(
        response=json.dumps(
            {"text": text, "source_fact_ids": source_ids},
            ensure_ascii=False,
        )
    )


def summary_provider_factory(scored):
    analysis = analyze_rally(scored.fact)
    plan = plan_commentary(scored, analysis)
    pattern_id = next(
        fact_id for fact_id in plan.allowed_fact_ids if ":pattern:" in fact_id
    )
    response = GeneratedCommentary(
        segment_index=scored.fact.segment_index,
        text="挑球後緊接進攻球。",
        source_fact_ids=[pattern_id],
    )
    return FakeProvider(response=response.model_dump_json())


def test_event_lines_and_rally_summary_coexist_in_time_order():
    output = generate_event_driven_commentary(
        scored_rallies=[scored_fact()],
        event_provider_factory=event_provider_factory,
        summary_provider_factory=summary_provider_factory,
        player_names={"a": "戴資穎", "b": "安洗瑩"},
    )

    bundle = output.rallies[0]
    assert bundle.summary is not None
    assert [line.stroke_index for line in bundle.events] == sorted(
        line.stroke_index for line in bundle.events
    )
    assert [line.time_sec for line in bundle.events] == sorted(
        line.time_sec for line in bundle.events
    )
    assert all(line.source_fact_ids for line in bundle.events)

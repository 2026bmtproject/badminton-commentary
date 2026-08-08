import json

import pytest

from badminton_commentary.analysis.stroke_event_analyzer import analyze_stroke_events
from badminton_commentary.generation.event_commentator import (
    StrokeCommentaryGenerationError,
    generate_stroke_commentary,
)
from badminton_commentary.generation.event_planner import plan_stroke_commentary
from badminton_commentary.providers.fake import FakeProvider
from badminton_commentary.schemas import RallyFact, RallyFactEvent, RallyScore


def make_fact() -> RallyFact:
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
            (0, "a", "小球"),
            (1, "b", "挑球"),
            (2, "a", "殺球"),
        )
    ]
    return RallyFact(
        segment_index=3,
        game_index=None,
        start_sec=0,
        end_sec=3,
        duration_sec=3,
        score=RallyScore(a=20, b=20),
        server=None,
        events=events,
        rally_length=3,
        highlight_score=None,
    )


def make_analysis_and_plan():
    analysis = analyze_stroke_events(make_fact())[-1]
    return analysis, plan_stroke_commentary(analysis)


def provider_response(text: str, source_fact_ids: list[str]) -> FakeProvider:
    return FakeProvider(
        response=json.dumps(
            {"text": text, "source_fact_ids": source_fact_ids},
            ensure_ascii=False,
        )
    )


def test_generated_event_keeps_exact_metadata_and_sequence_provenance():
    fact = make_fact()
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]
    source_ids = [local.fact_id, *local.supporting_fact_ids]

    line = generate_stroke_commentary(
        provider=provider_response("小球、挑球後接上殺球。", source_ids),
        analysis=analysis,
        plan=plan,
        score=fact.score,
        player_names={"a": "甲", "b": "乙"},
    )

    assert (line.segment_index, line.stroke_index, line.frame, line.time_sec) == (
        3,
        2,
        60,
        2.0,
    )
    assert line.source_fact_ids == source_ids


def test_markdown_fenced_event_json_is_accepted():
    fact = make_fact()
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]
    source_ids = [local.fact_id, *local.supporting_fact_ids]
    payload = provider_response("小球、挑球後接上殺球。", source_ids).response

    line = generate_stroke_commentary(
        provider=FakeProvider(response=f"```json\n{payload}\n```"),
        analysis=analysis,
        plan=plan,
        score=fact.score,
    )

    assert line.text == "小球、挑球後接上殺球。"
    assert line.source_fact_ids == source_ids


def test_low_salience_event_normalizes_exclamation_to_period():
    fact = make_fact()
    fact.events = fact.events[:1]
    fact.events[0].stroke_type = "小球"
    fact.rally_length = 1
    analysis = analyze_stroke_events(fact)[0]
    plan = plan_stroke_commentary(analysis)

    line = generate_stroke_commentary(
        provider=provider_response(
            "甲放了一顆小球！",
            [analysis.current_stroke.fact_id],
        ),
        analysis=analysis,
        plan=plan,
        score=fact.score,
    )

    assert line.text == "甲放了一顆小球。"


def test_high_salience_event_normalizes_multiple_exclamations_to_one():
    fact = make_fact()
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]
    source_ids = [local.fact_id, *local.supporting_fact_ids]

    line = generate_stroke_commentary(
        provider=provider_response("小球、挑球後接上殺球！！", source_ids),
        analysis=analysis,
        plan=plan,
        score=fact.score,
    )

    assert line.text == "小球、挑球後接上殺球！"


def test_local_sequence_requires_all_supporting_stroke_facts():
    fact = make_fact()
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]

    with pytest.raises(StrokeCommentaryGenerationError, match="missing support"):
        generate_stroke_commentary(
            provider=provider_response(
                "小球後接上殺球。",
                [local.fact_id, *local.supporting_fact_ids[1:]],
            ),
            analysis=analysis,
            plan=plan,
            score=fact.score,
        )


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("這記殺球拿下這一分。", "outcome claim"),
        ("甲從後場移動到網前完成殺球。", "movement or causal"),
        ("甲靠著殺球掌握主動。", "movement or causal"),
    ],
)
def test_event_commentary_rejects_unsupported_claims(text, error):
    fact = make_fact()
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]
    source_ids = [local.fact_id, *local.supporting_fact_ids]

    with pytest.raises(StrokeCommentaryGenerationError, match=error):
        generate_stroke_commentary(
            provider=provider_response(text, source_ids),
            analysis=analysis,
            plan=plan,
            score=fact.score,
        )


def test_written_score_requires_score_fact_provenance():
    fact = make_fact()
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]
    source_ids = [local.fact_id, *local.supporting_fact_ids]

    with pytest.raises(StrokeCommentaryGenerationError, match="score provenance"):
        generate_stroke_commentary(
            provider=provider_response(
                "20 比 20，小球、挑球後接上殺球。",
                source_ids,
            ),
            analysis=analysis,
            plan=plan,
            score=fact.score,
        )


def test_wrong_written_score_is_rejected_even_with_score_fact():
    fact = make_fact()
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]
    source_ids = [
        local.fact_id,
        *local.supporting_fact_ids,
        "rally:3:score",
    ]

    with pytest.raises(StrokeCommentaryGenerationError, match="score does not match"):
        generate_stroke_commentary(
            provider=provider_response(
                "19 比 20，小球、挑球後接上殺球。",
                source_ids,
            ),
            analysis=analysis,
            plan=plan,
            score=fact.score,
        )


def test_unavailable_score_fact_cannot_be_cited():
    fact = make_fact()
    fact.score = RallyScore(a=None, b=None)
    analysis, plan = make_analysis_and_plan()
    local = analysis.local_facts[0]
    source_ids = [
        local.fact_id,
        *local.supporting_fact_ids,
        "rally:3:score",
    ]

    with pytest.raises(StrokeCommentaryGenerationError, match="unavailable score"):
        generate_stroke_commentary(
            provider=provider_response(
                "小球、挑球後接上殺球。",
                source_ids,
            ),
            analysis=analysis,
            plan=plan,
            score=fact.score,
        )

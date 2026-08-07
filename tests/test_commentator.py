import json

import pytest

from badminton_commentary.generation.commentator import (
    CommentaryGenerationError,
    generate_commentary,
)
from badminton_commentary.generation.planner import plan_commentary
from badminton_commentary.providers.fake import FakeProvider
from badminton_commentary.schemas import (
    ImportanceResult,
    RallyFact,
    RallyFactEvent,
    RallyScore,
    ScoredRallyFact,
)


def make_scored_fact() -> ScoredRallyFact:
    return ScoredRallyFact(
        fact=RallyFact(
            segment_index=2,
            game_index=None,
            start_sec=0,
            end_sec=1,
            duration_sec=1,
            score=RallyScore(a=20, b=20),
            server=None,
            events=[],
            rally_length=0,
            highlight_score=None,
        ),
        importance=ImportanceResult(
            score=0.5,
            reasons=["close_score", "late_game_score"],
        ),
    )


def make_cautious_stroke_fact() -> ScoredRallyFact:
    scored = make_scored_fact()
    scored.fact.events = [
        RallyFactEvent(
            event_index=4,
            frame=15,
            time_sec=0.5,
            player="a",
            stroke_type="殺球",
            stroke_confidence=0.6,
        )
    ]
    scored.fact.rally_length = 1
    return scored


def make_pattern_fact() -> ScoredRallyFact:
    scored = make_scored_fact()
    scored.fact.events = [
        RallyFactEvent(
            event_index=index,
            frame=index * 10,
            time_sec=index / 3,
            player=player,
            stroke_type=stroke_type,
            stroke_confidence=0.9,
        )
        for index, player, stroke_type in (
            (0, "a", "發球"),
            (1, "b", "小球"),
            (2, "a", "小球"),
        )
    ]
    scored.fact.rally_length = 3
    return scored


def make_rear_front_pattern_fact() -> ScoredRallyFact:
    scored = make_scored_fact()
    scored.fact.events = [
        RallyFactEvent(
            event_index=index,
            frame=index * 10,
            time_sec=index / 3,
            player=player,
            stroke_type=stroke_type,
            stroke_confidence=0.9,
        )
        for index, player, stroke_type in (
            (0, "a", "高遠球"),
            (1, "b", "小球"),
            (2, "a", "小球"),
        )
    ]
    scored.fact.rally_length = 3
    return scored


def make_diversity_pattern_fact() -> ScoredRallyFact:
    scored = make_scored_fact()
    scored.fact.events = [
        RallyFactEvent(
            event_index=index,
            frame=index * 10,
            time_sec=index / 3,
            player=player,
            stroke_type=stroke_type,
            stroke_confidence=0.9,
        )
        for index, player, stroke_type in (
            (0, "a", "平快球"),
            (1, "b", "小球"),
            (2, "a", "高遠球"),
        )
    ]
    scored.fact.rally_length = 3
    return scored


def response(**overrides) -> str:
    payload = {
        "segment_index": 2,
        "text": "雙方戰成二十平，局勢相當緊張。",
        "source_fact_ids": ["rally:2:score"],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_fake_provider_generates_valid_grounded_commentary():
    scored = make_scored_fact()
    provider = FakeProvider(response=response())

    generated = generate_commentary(
        provider=provider,
        scored=scored,
        plan=plan_commentary(scored),
        player_names={"a": "戴資穎", "b": "安洗瑩"},
    )

    assert generated.segment_index == 2
    assert generated.source_fact_ids == ["rally:2:score"]
    prompt_payload = json.loads(provider.calls[0].user_prompt)
    assert prompt_payload["players"]["a"] == "戴資穎"
    assert list(prompt_payload["fact_catalog"]) == ["rally:2:score"]


def test_markdown_fenced_json_is_accepted():
    scored = make_scored_fact()
    provider = FakeProvider(response=f"```json\n{response()}\n```")

    generated = generate_commentary(
        provider=provider, scored=scored, plan=plan_commentary(scored)
    )

    assert generated.segment_index == 2


@pytest.mark.parametrize("provider_response", ["not json", "{}"])
def test_invalid_provider_json_is_rejected(provider_response):
    scored = make_scored_fact()

    with pytest.raises(CommentaryGenerationError, match="invalid commentary JSON"):
        generate_commentary(
            provider=FakeProvider(response=provider_response),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_disallowed_fact_id_is_rejected():
    scored = make_scored_fact()

    with pytest.raises(CommentaryGenerationError, match="disallowed fact ids"):
        generate_commentary(
            provider=FakeProvider(
                response=response(source_fact_ids=["rally:2:winner"])
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_sentence_limit_is_enforced():
    scored = make_scored_fact()
    plan = plan_commentary(scored)
    plan.max_sentences = 1

    with pytest.raises(CommentaryGenerationError, match="exceeds 1 sentences"):
        generate_commentary(
            provider=FakeProvider(response=response(text="第一句。第二句。")),
            scored=scored,
            plan=plan,
        )


def test_unsupported_winner_claim_is_rejected():
    scored = make_scored_fact()

    with pytest.raises(CommentaryGenerationError, match="outcome claim"):
        generate_commentary(
            provider=FakeProvider(response=response(text="戴資穎拿下這一分。")),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_cautious_stroke_requires_uncertainty_wording():
    scored = make_cautious_stroke_fact()
    stroke_id = "rally:2:stroke:4"

    with pytest.raises(CommentaryGenerationError, match="uncertainty wording"):
        generate_commentary(
            provider=FakeProvider(
                response=response(text="戴資穎使用殺球。", source_fact_ids=[stroke_id])
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_cautious_stroke_accepts_uncertainty_wording():
    scored = make_cautious_stroke_fact()
    stroke_id = "rally:2:stroke:4"

    generated = generate_commentary(
        provider=FakeProvider(
            response=response(
                text="辨識結果顯示戴資穎可能使用殺球。",
                source_fact_ids=[stroke_id],
            )
        ),
        scored=scored,
        plan=plan_commentary(scored),
    )

    assert generated.source_fact_ids == [stroke_id]


def test_wrong_written_score_is_rejected():
    scored = make_scored_fact()

    with pytest.raises(CommentaryGenerationError, match="score does not match"):
        generate_commentary(
            provider=FakeProvider(response=response(text="雙方目前戰成19比20。")),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_tactical_pattern_catalog_contains_meaning_and_support():
    scored = make_pattern_fact()
    pattern_id = "rally:2:pattern:serve_return_pattern"
    provider = FakeProvider(
        response=response(
            text="雙方由發接發迅速銜接到第三拍。",
            source_fact_ids=[pattern_id],
        )
    )

    generated = generate_commentary(
        provider=provider,
        scored=scored,
        plan=plan_commentary(scored),
    )

    catalog = json.loads(provider.calls[0].user_prompt)["fact_catalog"][pattern_id]
    assert generated.source_fact_ids == [pattern_id]
    assert catalog["commentary_hint"] == "發接發後連續銜接第三拍"
    assert catalog["evidence_scope"] == "stroke_sequence_only"
    assert list(catalog["supporting_strokes"]) == [
        "rally:2:stroke:0",
        "rally:2:stroke:1",
        "rally:2:stroke:2",
    ]


@pytest.mark.parametrize("term", ["觀測球路", "短窗口", "後場類型"])
def test_internal_schema_wording_is_rejected(term):
    scored = make_rear_front_pattern_fact()
    pattern_id = "rally:2:pattern:rear_court_stroke_to_front_court_stroke"

    with pytest.raises(CommentaryGenerationError, match="internal schema wording"):
        generate_commentary(
            provider=FakeProvider(
                response=response(text=f"{term}轉為網前處理。", source_fact_ids=[pattern_id])
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_rear_front_pattern_cannot_claim_player_movement():
    scored = make_rear_front_pattern_fact()
    pattern_id = "rally:2:pattern:rear_court_stroke_to_front_court_stroke"

    with pytest.raises(CommentaryGenerationError, match="movement or causal"):
        generate_commentary(
            provider=FakeProvider(
                response=response(
                    text="戴資穎從後場移動到網前。",
                    source_fact_ids=[pattern_id],
                )
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_stroke_diversity_cannot_claim_tempo_change():
    scored = make_diversity_pattern_fact()
    pattern_id = "rally:2:pattern:stroke_diversity"

    with pytest.raises(CommentaryGenerationError, match="temporal variation"):
        generate_commentary(
            provider=FakeProvider(
                response=response(
                    text="雙方不斷改變攻防節奏。",
                    source_fact_ids=[pattern_id],
                )
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_available_pattern_cannot_be_replaced_by_stroke_listing():
    scored = make_rear_front_pattern_fact()

    with pytest.raises(CommentaryGenerationError, match="omits an available tactical"):
        generate_commentary(
            provider=FakeProvider(
                response=response(
                    text="戴資穎打出小球。",
                    source_fact_ids=["rally:2:stroke:2"],
                )
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_available_pattern_cannot_be_replaced_by_score_only():
    scored = make_rear_front_pattern_fact()

    with pytest.raises(CommentaryGenerationError, match="omits an available tactical"):
        generate_commentary(
            provider=FakeProvider(response=response()),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_low_importance_commentary_cannot_use_exclamation_mark():
    scored = make_pattern_fact()

    with pytest.raises(CommentaryGenerationError, match="exclamation emphasis"):
        generate_commentary(
            provider=FakeProvider(
                response=response(
                    text="雙方由發接發銜接第三拍！",
                    source_fact_ids=["rally:2:pattern:serve_return_pattern"],
                )
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


def test_high_importance_commentary_allows_at_most_one_exclamation_mark():
    scored = make_pattern_fact()
    scored.importance.score = 0.75

    with pytest.raises(CommentaryGenerationError, match="exclamation emphasis"):
        generate_commentary(
            provider=FakeProvider(
                response=response(
                    text="發接發後銜接第三拍！球路變化精彩！",
                    source_fact_ids=["rally:2:pattern:serve_return_pattern"],
                )
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )


@pytest.mark.parametrize("causal_term", ["迫使", "導致", "造成", "靠著"])
def test_pattern_cannot_add_unsupported_causality(causal_term):
    scored = make_rear_front_pattern_fact()
    pattern_id = "rally:2:pattern:rear_court_stroke_to_front_court_stroke"

    with pytest.raises(CommentaryGenerationError, match="movement or causal"):
        generate_commentary(
            provider=FakeProvider(
                response=response(
                    text=f"後場球轉為網前處理，{causal_term}對手失去主動。",
                    source_fact_ids=[pattern_id],
                )
            ),
            scored=scored,
            plan=plan_commentary(scored),
        )

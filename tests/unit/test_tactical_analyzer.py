import json
from pathlib import Path

import pytest

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    read_upstream_stages,
)
from badminton_commentary.analysis import (
    TACTICAL_PROMPT_VERSION,
    TacticalAnalysisError,
    analyze_tactical_facts,
)
from badminton_commentary.facts import (
    GeneratedTacticalAnalysis,
    GeneratedTacticalFact,
    build_compact_rally_facts,
)
from badminton_commentary.providers import FakeProvider


FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "upstream_stages"


def compact_fixture():
    stages = read_upstream_stages(StagePaths.from_stage_root(FIXTURE_ROOT))
    return build_compact_rally_facts(
        stages=stages,
        segment_index=1,
        court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
    )


def tactical_payload(**updates):
    fact = GeneratedTacticalFact(
        pattern_type="rear_to_front_stroke_transition",
        description="這段球路從後場球轉入網前處理。",
        confidence=0.85,
        salience=0.8,
        start_event_index=2,
        end_event_index=4,
        players=["b"],
        evidence_fact_ids=["rally:1:stroke:2", "rally:1:stroke:4"],
        limitations=["只描述球種轉換，不代表球員實際移動"],
    ).model_copy(update=updates)
    return GeneratedTacticalAnalysis(
        segment_index=1,
        facts=[fact],
    ).model_dump_json()


def test_analyzer_uses_one_call_and_assigns_traceable_fact_id():
    provider = FakeProvider(response=tactical_payload())

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact_fixture(),
    )

    assert len(provider.calls) == 1
    assert result.prompt_version == TACTICAL_PROMPT_VERSION
    assert result.provider_model is None
    assert result.warnings == []
    assert result.facts[0].fact_id == (
        "rally:1:tactical:0:rear_to_front_stroke_transition"
    )
    assert result.facts[0].evidence_fact_ids == [
        "rally:1:stroke:2",
        "rally:1:stroke:4",
    ]
    user_payload = json.loads(provider.calls[0].user_prompt)
    assert user_payload["compact_rally_facts"]["segment_index"] == 1
    assert "keypoints" not in provider.calls[0].user_prompt


def test_analyzer_accepts_optional_markdown_json_fence():
    provider = FakeProvider(response=f"```json\n{tactical_payload()}\n```")

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact_fixture(),
    )

    assert len(result.facts) == 1


def test_analyzer_discards_unknown_or_cross_rally_evidence():
    provider = FakeProvider(
        response=tactical_payload(
            evidence_fact_ids=["rally:0:stroke:2", "rally:1:stroke:4"]
        )
    )

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact_fixture(),
    )

    assert result.facts == []
    assert result.warnings == [
        "rejected_tactical_fact:0:rear_to_front_stroke_transition:"
        "unknown_evidence_fact",
        "no_supported_tactical_patterns",
    ]


def test_analyzer_discards_pattern_without_required_stroke_evidence():
    provider = FakeProvider(
        response=tactical_payload(pattern_type="sustained_attack")
    )

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact_fixture(),
    )

    assert result.facts == []
    assert result.warnings[0].endswith("sustained_attack_missing_evidence")


@pytest.mark.parametrize("unsupported", ["正手", "最後一拍", "球速", "迫使"])
def test_analyzer_discards_unsupported_semantic_claims(unsupported):
    provider = FakeProvider(
        response=tactical_payload(description=f"這是{unsupported}形成的變化。")
    )

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact_fixture(),
    )

    assert result.facts == []
    assert result.warnings[0].endswith("unsupported_description_claim")


def test_analyzer_skips_provider_when_fewer_than_two_events():
    compact = compact_fixture().model_copy(
        update={"events": compact_fixture().events[:1]}
    )
    provider = FakeProvider(response="not used")

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact,
    )

    assert result.facts == []
    assert result.warnings == ["insufficient_events_for_tactical_analysis"]
    assert provider.calls == []


def test_analyzer_rejects_more_than_configured_max_facts():
    payload = json.loads(tactical_payload())
    payload["facts"].append(
        {
            **payload["facts"][0],
            "pattern_type": "notable_stroke_sequence",
        }
    )
    provider = FakeProvider(response=json.dumps(payload, ensure_ascii=False))

    with pytest.raises(TacticalAnalysisError, match="more than max_facts=1"):
        analyze_tactical_facts(
            provider=provider,
            compact_facts=compact_fixture(),
            max_facts=1,
        )


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"players": ["top"]}, "players.0"),
        ({"confidence": 1.1}, "confidence"),
        ({"start_event_index": -1}, "start_event_index"),
    ],
)
def test_generated_tactical_schema_rejects_invalid_core_values(updates, field):
    payload = json.loads(tactical_payload())
    payload["facts"][0].update(updates)
    provider = FakeProvider(response=json.dumps(payload, ensure_ascii=False))

    with pytest.raises(TacticalAnalysisError, match=field):
        analyze_tactical_facts(
            provider=provider,
            compact_facts=compact_fixture(),
        )


def test_candidate_pattern_without_limitation_is_discarded():
    provider = FakeProvider(
        response=tactical_payload(
            pattern_type="attacking_initiative_candidate",
            description="球員a掌握主動進攻。",
            evidence_fact_ids=["rally:1:stroke:2", "rally:1:stroke:3"],
            players=["a"],
            limitations=[],
        )
    )

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact_fixture(),
    )

    assert result.facts == []
    assert result.warnings[0].endswith("candidate_limitation_missing")


def test_invalid_front_back_candidate_does_not_remove_other_valid_fact():
    invalid = json.loads(tactical_payload())["facts"][0]
    invalid.update(
        {
            "pattern_type": "front_back_court_displacement",
            "description": "兩個擊球位置呈現前後場差異。",
        }
    )
    valid = json.loads(tactical_payload())["facts"][0]
    provider = FakeProvider(
        response=json.dumps(
            {"segment_index": 1, "facts": [invalid, valid]},
            ensure_ascii=False,
        )
    )

    result = analyze_tactical_facts(
        provider=provider,
        compact_facts=compact_fixture(),
    )

    assert len(result.facts) == 1
    assert result.facts[0].fact_id == (
        "rally:1:tactical:0:rear_to_front_stroke_transition"
    )
    assert result.warnings == [
        "rejected_tactical_fact:0:front_back_court_displacement:"
        "same_player_depth_change_missing"
    ]


def test_analyzer_records_provider_model_fallback():
    class ModelAwareFakeProvider:
        model = "pro-preview"
        last_model_used = "stable-flash"

        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return tactical_payload()

    result = analyze_tactical_facts(
        provider=ModelAwareFakeProvider(),
        compact_facts=compact_fixture(),
    )

    assert result.provider_model == "stable-flash"
    assert result.warnings == [
        "provider_model_fallback:pro-preview->stable-flash"
    ]

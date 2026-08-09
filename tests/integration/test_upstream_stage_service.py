import json
from pathlib import Path

import pytest

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    read_upstream_stages,
)
from badminton_commentary.analysis import analyze_rally, analyze_stroke_events
from badminton_commentary.generation.planner import plan_selected_rally_summary
from badminton_commentary.generation.rally_batch_commentator import (
    RallyBatchGenerationError,
)
from badminton_commentary.providers import FakeProvider
from badminton_commentary.schemas import (
    GeneratedCommentary,
    GeneratedRallyTextBatch,
    GeneratedStrokeBatchItem,
)
from badminton_commentary.services import RallyCommentaryService


FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "upstream_stages"


def _fake_response(fact):
    event_items = []
    for analysis in analyze_stroke_events(fact, include_all_strokes=True):
        if analysis.local_facts:
            local = analysis.local_facts[0]
            text = f"{local.commentary_hint}。"
            source_fact_ids = [local.fact_id, *local.supporting_fact_ids]
        else:
            text = f"這拍辨識為{analysis.current_stroke.stroke_type}。"
            source_fact_ids = [analysis.current_stroke.fact_id]
        event_items.append(
            GeneratedStrokeBatchItem(
                stroke_index=analysis.stroke_index,
                text=text,
                source_fact_ids=source_fact_ids,
            )
        )

    rally_analysis = analyze_rally(fact)
    summary_plan = plan_selected_rally_summary(fact, rally_analysis)
    pattern = next(
        pattern
        for pattern in rally_analysis.patterns
        if pattern.fact_id in summary_plan.allowed_fact_ids
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


def test_parsed_stages_to_commentary_is_offline_and_one_provider_call():
    stages = read_upstream_stages(StagePaths.from_stage_root(FIXTURE_ROOT))
    mapping = CourtPositionToPlayer(top="b", bottom="a")
    preparation_service = RallyCommentaryService(
        provider=FakeProvider(response="not used")
    )
    fact = preparation_service.prepare_rally_fact(
        stages=stages,
        segment_index=1,
        court_position_to_player=mapping,
    )
    provider = FakeProvider(response=_fake_response(fact))

    bundle = RallyCommentaryService(provider=provider).generate_from_stages(
        stages=stages,
        segment_index=1,
        court_position_to_player=mapping,
    )

    assert bundle.segment_index == 1
    assert bundle.events
    assert bundle.summary is not None
    assert len(provider.calls) == 1
    assert all(line.segment_index == 1 for line in bundle.events)
    assert all(
        fact_id.startswith("rally:1:")
        for line in bundle.events
        for fact_id in line.source_fact_ids
    )


def test_stages_entrypoint_keeps_provenance_validation():
    stages = read_upstream_stages(StagePaths.from_stage_root(FIXTURE_ROOT))
    mapping = CourtPositionToPlayer(top="b", bottom="a")
    service = RallyCommentaryService(provider=FakeProvider(response="not used"))
    fact = service.prepare_rally_fact(
        stages=stages,
        segment_index=1,
        court_position_to_player=mapping,
    )
    payload = json.loads(_fake_response(fact))
    payload["events"][0]["source_fact_ids"] = ["rally:0:stroke:0"]

    with pytest.raises(RallyBatchGenerationError, match="disallowed fact ids"):
        RallyCommentaryService(
            provider=FakeProvider(response=json.dumps(payload, ensure_ascii=False))
        ).generate_from_stages(
            stages=stages,
            segment_index=1,
            court_position_to_player=mapping,
        )


def test_service_prepares_compact_facts_without_calling_provider():
    stages = read_upstream_stages(StagePaths.from_stage_root(FIXTURE_ROOT))
    provider = FakeProvider(response="not used")

    compact = RallyCommentaryService(provider=provider).prepare_compact_facts(
        stages=stages,
        segment_index=1,
        court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
    )

    assert [item.event_index for item in compact.events] == [2, 3, 4]
    assert compact.schema_version == "compact-rally-facts-v1"
    assert provider.calls == []

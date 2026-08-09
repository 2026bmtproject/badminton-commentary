import inspect
from pathlib import Path

import pytest

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    build_rally_fact_from_stages,
    read_upstream_stages,
)
from badminton_commentary.adapters import upstream as adapter_module


FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "upstream_stages"


def fixture_stages():
    return read_upstream_stages(StagePaths.from_stage_root(FIXTURE_ROOT))


def test_reads_actual_main_system_stage_envelopes_without_optional_stages():
    paths = StagePaths.from_stage_root(FIXTURE_ROOT)
    stages = read_upstream_stages(paths)

    assert len(stages.match_segmentation.segments) == 3
    assert len(stages.event_detection.events) == 6
    assert stages.highlights is None
    assert paths.court_detection is None
    assert paths.shuttle_tracking is None
    assert paths.pose is None


def test_builds_only_requested_segment_and_preserves_source_event_indexes():
    fact = build_rally_fact_from_stages(
        stages=fixture_stages(),
        segment_index=1,
        court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
    )

    assert fact.segment_index == 1
    assert (fact.score.a, fact.score.b) == (8, 7)
    assert fact.server == "a"
    assert fact.highlight_score is None
    assert [event.event_index for event in fact.events] == [2, 3, 4]
    assert [event.frame for event in fact.events] == [110, 130, 160]
    assert [event.player for event in fact.events] == ["b", "a", "b"]
    assert [event.stroke_type for event in fact.events] == [
        "高遠球",
        "殺球",
        "小球",
    ]
    assert all(100 <= event.frame <= 199 for event in fact.events)


def test_position_mapping_is_required_instead_of_guessing_player_identity():
    with pytest.raises(ValueError, match="court_position_to_player is required"):
        build_rally_fact_from_stages(
            stages=fixture_stages(),
            segment_index=1,
            court_position_to_player=None,
        )


def test_unknown_upstream_hitter_degrades_without_a_position_mapping():
    fact = build_rally_fact_from_stages(
        stages=fixture_stages(),
        segment_index=2,
        court_position_to_player=None,
    )

    assert fact.rally_length == 1
    assert fact.events[0].event_index == 5
    assert fact.events[0].player is None
    assert fact.events[0].stroke_type == "未知球種"


def test_adapter_has_no_ttyvsasy_or_experiment_dependency():
    source = inspect.getsource(adapter_module)

    assert "TTYvsASY" not in source
    assert "experiments" not in source
    assert "workspace" not in source

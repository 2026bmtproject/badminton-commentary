from __future__ import annotations

from collections.abc import Mapping

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    UpstreamStageData,
    build_rally_fact_from_stages,
)
from badminton_commentary.generation.rally_batch_commentator import (
    generate_rally_commentary_batch,
)
from badminton_commentary.providers import LLMProvider
from badminton_commentary.schemas import (
    ImportanceResult,
    Player,
    RallyCommentaryBundle,
    RallyFact,
    ScoredRallyFact,
)


class RallyCommentaryService:
    """Production boundary for one user-selected rally and one provider call."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        player_names: Mapping[Player, str] | None = None,
    ) -> None:
        self._provider = provider
        self._player_names = dict(player_names) if player_names is not None else None

    def generate(
        self,
        *,
        rally_fact: RallyFact,
        importance: ImportanceResult | None = None,
    ) -> RallyCommentaryBundle:
        """Generate every identifiable stroke and a summary for a selected rally."""
        has_score = (
            rally_fact.score.a is not None and rally_fact.score.b is not None
        )
        if not has_score and not rally_fact.events and rally_fact.highlight_score is None:
            raise ValueError("rally_fact has no grounded commentary facts")
        scored = ScoredRallyFact(
            fact=rally_fact,
            importance=importance
            or ImportanceResult(score=0, reasons=["user_selected_rally"]),
        )
        return generate_rally_commentary_batch(
            provider=self._provider,
            scored=scored,
            player_names=self._player_names,
            require_summary=True,
            user_selected_rally=True,
        )

    def prepare_rally_fact(
        self,
        *,
        stages: UpstreamStageData,
        segment_index: int,
        court_position_to_player: CourtPositionToPlayer | None,
    ) -> RallyFact:
        """Build the canonical commentary fact for one upstream segment."""
        return build_rally_fact_from_stages(
            stages=stages,
            segment_index=segment_index,
            court_position_to_player=court_position_to_player,
        )

    def generate_from_stages(
        self,
        *,
        stages: UpstreamStageData,
        segment_index: int,
        court_position_to_player: CourtPositionToPlayer | None,
        importance: ImportanceResult | None = None,
    ) -> RallyCommentaryBundle:
        """Adapt one upstream segment and generate it with one provider call."""
        return self.generate(
            rally_fact=self.prepare_rally_fact(
                stages=stages,
                segment_index=segment_index,
                court_position_to_player=court_position_to_player,
            ),
            importance=importance,
        )


def generate_rally_commentary(
    *,
    rally_fact: RallyFact,
    provider: LLMProvider,
    player_names: Mapping[Player, str] | None = None,
    importance: ImportanceResult | None = None,
) -> RallyCommentaryBundle:
    """Functional convenience API for a single user-selected rally."""
    return RallyCommentaryService(
        provider=provider,
        player_names=player_names,
    ).generate(
        rally_fact=rally_fact,
        importance=importance,
    )


def generate_rally_commentary_from_stages(
    *,
    stages: UpstreamStageData,
    segment_index: int,
    court_position_to_player: CourtPositionToPlayer | None,
    provider: LLMProvider,
    player_names: Mapping[Player, str] | None = None,
    importance: ImportanceResult | None = None,
) -> RallyCommentaryBundle:
    """Functional high-level API for parsed main-system stage data."""
    return RallyCommentaryService(
        provider=provider,
        player_names=player_names,
    ).generate_from_stages(
        stages=stages,
        segment_index=segment_index,
        court_position_to_player=court_position_to_player,
        importance=importance,
    )

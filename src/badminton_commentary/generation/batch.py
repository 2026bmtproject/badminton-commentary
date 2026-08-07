from collections.abc import Callable, Iterable

from badminton_commentary.providers import LLMProvider
from badminton_commentary.analysis.rally_analyzer import analyze_rally
from badminton_commentary.schemas import (
    CommentaryOutput,
    ScoredRallyFact,
)

from .commentator import generate_commentary
from .planner import plan_commentary


def generate_commentaries(
    *,
    scored_rallies: Iterable[ScoredRallyFact],
    provider_factory: Callable[[ScoredRallyFact], LLMProvider],
    player_names: dict[str, str] | None = None,
) -> CommentaryOutput:
    """Plan and generate commentary for every eligible rally."""
    lines = []
    for scored in scored_rallies:
        analysis = analyze_rally(scored.fact)
        plan = plan_commentary(scored, analysis)
        if not plan.should_comment:
            continue
        lines.append(
            generate_commentary(
                provider=provider_factory(scored),
                scored=scored,
                plan=plan,
                analysis=analysis,
                player_names=player_names,
            )
        )
    return CommentaryOutput(lines=lines)

from __future__ import annotations

from collections.abc import Callable, Iterable

from badminton_commentary.analysis import analyze_rally, analyze_stroke_events
from badminton_commentary.providers import LLMProvider
from badminton_commentary.schemas import (
    EventDrivenCommentaryOutput,
    RallyCommentaryBundle,
    ScoredRallyFact,
    StrokeEventAnalysis,
    StrokeEventPlan,
)

from .commentator import generate_commentary
from .event_commentator import generate_stroke_commentary
from .event_planner import plan_stroke_commentary
from .planner import plan_commentary


EventProviderFactory = Callable[
    [ScoredRallyFact, StrokeEventAnalysis, StrokeEventPlan],
    LLMProvider,
]
SummaryProviderFactory = Callable[[ScoredRallyFact], LLMProvider]


def generate_event_driven_commentary(
    *,
    scored_rallies: Iterable[ScoredRallyFact],
    event_provider_factory: EventProviderFactory,
    summary_provider_factory: SummaryProviderFactory | None = None,
    player_names: dict[str, str] | None = None,
) -> EventDrivenCommentaryOutput:
    """Generate chronological per-stroke lines plus optional rally summaries."""
    bundles = []
    for scored in sorted(scored_rallies, key=lambda item: item.fact.segment_index):
        event_lines = []
        for event_analysis in analyze_stroke_events(scored.fact):
            event_plan = plan_stroke_commentary(event_analysis)
            if not event_plan.should_comment:
                continue
            event_lines.append(
                generate_stroke_commentary(
                    provider=event_provider_factory(scored, event_analysis, event_plan),
                    analysis=event_analysis,
                    plan=event_plan,
                    score=scored.fact.score,
                    player_names=player_names,
                )
            )
        if any(
            later.stroke_index <= earlier.stroke_index
            or later.time_sec < earlier.time_sec
            for earlier, later in zip(event_lines, event_lines[1:])
        ):
            raise ValueError("generated stroke commentary is not chronological")

        summary = None
        rally_analysis = analyze_rally(scored.fact)
        summary_plan = plan_commentary(scored, rally_analysis)
        if summary_provider_factory is not None and summary_plan.should_comment:
            summary = generate_commentary(
                provider=summary_provider_factory(scored),
                scored=scored,
                plan=summary_plan,
                analysis=rally_analysis,
                player_names=player_names,
            )
        bundles.append(
            RallyCommentaryBundle(
                segment_index=scored.fact.segment_index,
                events=event_lines,
                summary=summary,
            )
        )
    return EventDrivenCommentaryOutput(rallies=bundles)

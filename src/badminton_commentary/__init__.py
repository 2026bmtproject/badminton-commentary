from badminton_commentary.facts import (
    CompactFactConfig,
    CompactRallyFacts,
    TacticalAnalysisResult,
    TacticalFact,
    build_compact_rally_facts,
)
from badminton_commentary.analysis import analyze_tactical_facts
from badminton_commentary.services import (
    RallyCommentaryService,
    generate_rally_commentary,
    generate_rally_commentary_from_stages,
)

__all__ = [
    "CompactFactConfig",
    "CompactRallyFacts",
    "TacticalAnalysisResult",
    "TacticalFact",
    "RallyCommentaryService",
    "analyze_tactical_facts",
    "build_compact_rally_facts",
    "generate_rally_commentary",
    "generate_rally_commentary_from_stages",
]


def main() -> None:
    from badminton_commentary.cli import main as cli_main

    cli_main()

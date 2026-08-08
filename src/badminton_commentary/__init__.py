from badminton_commentary.services import (
    RallyCommentaryService,
    generate_rally_commentary,
)

__all__ = ["RallyCommentaryService", "generate_rally_commentary"]


def main() -> None:
    from badminton_commentary.cli import main as cli_main

    cli_main()

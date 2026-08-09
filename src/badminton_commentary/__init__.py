from badminton_commentary.services import (
    RallyCommentaryService,
    generate_rally_commentary,
    generate_rally_commentary_from_stages,
)

__all__ = [
    "RallyCommentaryService",
    "generate_rally_commentary",
    "generate_rally_commentary_from_stages",
]


def main() -> None:
    from badminton_commentary.cli import main as cli_main

    cli_main()

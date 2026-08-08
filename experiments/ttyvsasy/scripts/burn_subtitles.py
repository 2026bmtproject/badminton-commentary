from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from badminton_commentary.schemas import (
    EventDrivenCommentaryOutput,
    SegmentsInput,
)
from badminton_commentary.subtitles import build_subtitle_cues, render_ass


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_ROOT = EXPERIMENT_ROOT / "workspace" / "selected_clips"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ttyvsasy"


def _source_video(group_dir: Path) -> Path:
    candidates = [
        path
        for path in (group_dir / "video").glob("*.mp4")
        if "_commentary_" not in path.stem
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one source MP4 in {group_dir / 'video'}, found {len(candidates)}"
        )
    return candidates[0]


def _ffmpeg_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.replace(":", r"\:").replace("'", r"\'")


def process_group(
    workspace_group_dir: Path,
    output_group_dir: Path,
    *,
    provider: str,
    event_duration: float,
    summary_duration: float,
    font_name: str,
    event_font_size: int,
    summary_font_size: int,
    crf: int,
    preset: str,
    subtitles_only: bool,
    overwrite: bool,
) -> tuple[Path, Path | None]:
    commentary_path = output_group_dir / f"commentary_{provider}_event_driven.json"
    segments_path = workspace_group_dir / "commentary_input" / "segments.json"
    if not commentary_path.is_file():
        raise FileNotFoundError(f"commentary file not found: {commentary_path}")
    if not segments_path.is_file():
        raise FileNotFoundError(f"segments file not found: {segments_path}")

    commentary = EventDrivenCommentaryOutput.model_validate_json(
        commentary_path.read_text(encoding="utf-8")
    )
    segments = SegmentsInput.model_validate_json(
        segments_path.read_text(encoding="utf-8")
    )
    cues = build_subtitle_cues(
        commentary,
        segments,
        event_duration=event_duration,
        summary_duration=summary_duration,
    )

    subtitle_dir = output_group_dir / "subtitles"
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    subtitle_path = subtitle_dir / f"commentary_{provider}_event_driven.ass"
    if subtitle_path.exists() and not overwrite:
        raise FileExistsError(f"subtitle already exists: {subtitle_path}")
    subtitle_path.write_text(
        render_ass(
            cues,
            font_name=font_name,
            event_font_size=event_font_size,
            summary_font_size=summary_font_size,
        ),
        encoding="utf-8-sig",
    )
    if subtitles_only:
        return subtitle_path, None

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    source_video = _source_video(workspace_group_dir)
    output_video_dir = output_group_dir / "video"
    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_video_dir / (
        f"{source_video.stem}_commentary_{provider}.mp4"
    )
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"output video already exists: {output_video}")

    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(source_video.resolve()),
        "-vf",
        f"ass=filename='{_ffmpeg_filter_path(subtitle_path)}'",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_video.resolve()),
    ]
    subprocess.run(command, check=True)
    return subtitle_path, output_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ASS subtitles and burn event-driven commentary into clips."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--provider", choices=("fake", "gemini"), default="gemini")
    parser.add_argument(
        "--group",
        action="append",
        help="Process only this group directory name; may be passed more than once.",
    )
    parser.add_argument("--event-duration", type=float, default=3.0)
    parser.add_argument("--summary-duration", type=float, default=4.5)
    parser.add_argument("--font-name", default="Microsoft JhengHei")
    parser.add_argument("--event-font-size", type=int, default=54)
    parser.add_argument("--summary-font-size", type=int, default=42)
    parser.add_argument("--crf", type=int, choices=range(0, 52), default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--subtitles-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    output_root = args.output_root.resolve()
    groups = args.group or sorted(
        path.name for path in workspace_root.iterdir() if path.is_dir()
    )
    if not groups:
        raise ValueError(f"no clip groups found under {workspace_root}")

    for group in groups:
        workspace_group_dir = workspace_root / group
        if not workspace_group_dir.is_dir():
            raise FileNotFoundError(f"clip group not found: {workspace_group_dir}")
        output_group_dir = output_root / group
        output_group_dir.mkdir(parents=True, exist_ok=True)
        subtitle_path, output_video = process_group(
            workspace_group_dir,
            output_group_dir,
            provider=args.provider,
            event_duration=args.event_duration,
            summary_duration=args.summary_duration,
            font_name=args.font_name,
            event_font_size=args.event_font_size,
            summary_font_size=args.summary_font_size,
            crf=args.crf,
            preset=args.preset,
            subtitles_only=args.subtitles_only,
            overwrite=args.overwrite,
        )
        print(f"subtitle: {subtitle_path}")
        if output_video is not None:
            print(f"video: {output_video}")


if __name__ == "__main__":
    main()

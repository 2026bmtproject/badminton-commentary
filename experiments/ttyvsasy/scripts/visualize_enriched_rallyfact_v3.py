"""Burn validated Gemini v3 observations and pose evidence into a rally clip."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = runpy.run_path(str(Path(__file__).with_name("run_direct_rallyfact_v3.py")))
DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT / "outputs" / "ttyvsasy" / "direct_rallyfact" / "seg0144"
)
DEFAULT_FACT = DEFAULT_EXPERIMENT_DIR / "gemini_enriched_rally_fact_v3.json"
DEFAULT_PACKAGE = DEFAULT_EXPERIMENT_DIR / "rally_stage_input.json"
DEFAULT_DEBUG_PACKAGE = DEFAULT_EXPERIMENT_DIR / "rally_stage_input_debug.json"
DEFAULT_VIDEO = (
    REPO_ROOT
    / "outputs"
    / "ttyvsasy"
    / "from_stages"
    / "seg0144"
    / "TTYvsASY_seg0144_corrected.mp4"
)
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "visualization"

SKELETON_EDGES = [
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

DIRECTION_LABELS = {
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
    "up_left": "↖",
    "up_right": "↗",
    "down_left": "↙",
    "down_right": "↘",
    "stable": "stable",
    "unknown": "?",
}


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _ass_escape(value: object) -> str:
    return str(value).replace("{", "｛").replace("}", "｝").replace("\\", "／")


def _dialogue(
    *,
    layer: int,
    start: float,
    end: float,
    style: str,
    text: str,
) -> str:
    return (
        f"Dialogue: {layer},{_ass_timestamp(start)},{_ass_timestamp(end)},"
        f"{style},,0,0,0,,{text}"
    )


def _line_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    thickness: float = 5.0,
) -> str | None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return None
    px = -(y2 - y1) * thickness / (2 * length)
    py = (x2 - x1) * thickness / (2 * length)
    points = [
        (x1 + px, y1 + py),
        (x2 + px, y2 + py),
        (x2 - px, y2 - py),
        (x1 - px, y1 - py),
    ]
    return "m " + " l ".join(f"{round(x)} {round(y)}" for x, y in points)


def _pose_path(keypoints: dict[str, list[float]]) -> str:
    paths = []
    for start_name, end_name in SKELETON_EDGES:
        start = keypoints[start_name]
        end = keypoints[end_name]
        if min(start[2], end[2]) < 0.3:
            continue
        polygon = _line_polygon((start[0], start[1]), (end[0], end[1]))
        if polygon is not None:
            paths.append(polygon)
    for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        x, y, confidence = keypoints[name]
        if confidence < 0.3:
            continue
        radius = 6
        paths.append(
            f"m {round(x-radius)} {round(y)} "
            f"l {round(x)} {round(y-radius)} "
            f"l {round(x+radius)} {round(y)} "
            f"l {round(x)} {round(y+radius)}"
        )
    return " ".join(paths)


def _event_text(event) -> str:
    pose = event.pose_observation
    court = event.court_observation
    shuttle = event.shuttle_observation
    stroke_confidence = (
        f"{event.stroke_confidence:.2f}"
        if event.stroke_confidence is not None
        else "—"
    )
    lines = [
        (
            f"EVENT #{event.event_index}  player {event.player or 'unknown'}  "
            f"frame {event.frame}"
        ),
        f"Stroke  {event.stroke_type or 'unknown'}  conf {stroke_confidence}",
    ]
    if pose is None:
        lines.append("Pose  unavailable")
    else:
        cues = ", ".join(pose.secondary_cues) or "none"
        lines.append(
            f"Pose  {pose.posture_candidate}  conf {pose.posture_confidence:.2f}  "
            f"cues [{cues}]  window {pose.source_start_frame}–{pose.source_end_frame}"
        )
    if court is None:
        lines.append("Court  unavailable")
    else:
        behind = (
            " · behind baseline"
            if "projected_point_behind_own_baseline" in court.limitations
            else ""
        )
        lines.append(
            f"Court  {court.depth_zone} / "
            f"{court.position_change_from_previous_same_player_hit}  "
            f"conf {court.confidence:.2f}{behind}"
        )
    if shuttle is None:
        lines.append("Shuttle  unavailable")
    else:
        incoming = DIRECTION_LABELS[shuttle.incoming_image_direction]
        outgoing = DIRECTION_LABELS[shuttle.outgoing_image_direction]
        lines.append(
            f"Shuttle  {incoming} → {outgoing}  "
            f"change {shuttle.trajectory_change_candidate}  "
            f"conf {shuttle.confidence:.2f}"
        )
    return r"\N".join(_ass_escape(line) for line in lines)


def _tactical_text(candidate) -> str:
    stages = ", ".join(sorted({item.stage for item in candidate.evidence}))
    return r"\N".join(
        _ass_escape(line)
        for line in (
            (
                f"TACTICAL  {candidate.pattern_type}  "
                f"E{candidate.start_event_index}–E{candidate.end_event_index}"
            ),
            candidate.description,
            (
                f"confidence {candidate.confidence:.2f} · "
                f"salience {candidate.salience:.2f} · evidence {stages}"
            ),
        )
    )


def render_ass(
    enriched,
    package: dict,
    *,
    debug_package: dict | None = None,
    model_label: str = "Gemini 3.1 Pro",
) -> str:
    rally = package["rally"]
    fps = float(rally["fps"])
    start_frame = rally["start_frame"]
    duration = (rally["end_frame"] - start_frame + 1) / fps

    def local_time(frame: int) -> float:
        return (frame - start_frame) / fps

    pose_events = {
        event["event_index"]: event
        for event in (debug_package or package)["events"]
    }
    event_times = {
        event.event_index: local_time(event.frame) for event in enriched.events
    }

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Event,Microsoft JhengHei,30,&H00FFFFFF,&H00FFFFFF,&H00101010,&H98000000,-1,0,0,0,100,100,0,0,3,2,0,1,34,34,28,1
Style: Tactical,Microsoft JhengHei,29,&H0015CCFA,&H0015CCFA,&H00101010,&HA0000000,-1,0,0,0,100,100,0,0,3,2,0,8,120,120,120,1
Style: Hit,Microsoft JhengHei,36,&H0015CCFA,&H0015CCFA,&H00101010,&H70000000,-1,0,0,0,100,100,0,0,3,2,0,8,50,50,44,1
Style: PoseB,Arial,10,&H0080DE4A,&H0080DE4A,&H00052E16,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: PoseA,Arial,10,&H0015CCFA,&H0015CCFA,&H00303030,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Title,Microsoft JhengHei,22,&H00FFFFFF,&H00FFFFFF,&H00101010,&H70000000,-1,0,0,0,100,100,0,0,3,1,0,9,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogue = []
    dialogue.append(
        _dialogue(
            layer=1,
            start=0,
            end=duration,
            style="Title",
            text=_ass_escape(f"{model_label} · Enriched RallyFact v3 · SEG144"),
        )
    )

    for position, event in enumerate(enriched.events):
        start = max(0.0, event_times[event.event_index] - 0.10)
        if position + 1 < len(enriched.events):
            next_time = event_times[enriched.events[position + 1].event_index]
            end = min(duration, next_time - 0.04)
        else:
            end = min(duration, event_times[event.event_index] + 1.25)
        end = max(start + 0.15, end)
        dialogue.append(
            _dialogue(
                layer=4,
                start=start,
                end=end,
                style="Event",
                text=_event_text(event),
            )
        )
        hit_time = event_times[event.event_index]
        dialogue.append(
            _dialogue(
                layer=5,
                start=max(0.0, hit_time - 0.06),
                end=min(duration, hit_time + 0.24),
                style="Hit",
                text=_ass_escape(f"● HIT #{event.event_index}"),
            )
        )

    pose_records: dict[tuple[int, str], dict] = {}
    for event in enriched.events:
        source = pose_events[event.event_index]
        for pose in source["pose_window"]:
            pose_records[(pose["frame"], pose["player"])] = pose
    for (frame, player), pose in sorted(pose_records.items()):
        start = max(0.0, local_time(frame))
        end = min(duration, start + 1 / fps)
        path = _pose_path(pose["keypoints"])
        if not path or end <= start:
            continue
        dialogue.append(
            _dialogue(
                layer=2,
                start=start,
                end=end,
                style="PoseB" if player == "b" else "PoseA",
                text=r"{\p1}" + path,
            )
        )

    for candidate in enriched.tactical_candidates:
        start = max(0.0, event_times[candidate.start_event_index] - 0.10)
        end_position = next(
            index
            for index, event in enumerate(enriched.events)
            if event.event_index == candidate.end_event_index
        )
        if end_position + 1 < len(enriched.events):
            end = event_times[enriched.events[end_position + 1].event_index] - 0.05
        else:
            end = event_times[candidate.end_event_index] + 1.0
        dialogue.append(
            _dialogue(
                layer=6,
                start=start,
                end=min(duration, end),
                style="Tactical",
                text=_tactical_text(candidate),
            )
        )

    return header + "\n".join(dialogue) + "\n"


def _ffmpeg_filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def _video_info(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)["streams"][0]


def visualize(
    *,
    fact_path: Path,
    package_path: Path,
    debug_package_path: Path,
    video_path: Path,
    output_dir: Path,
    model_label: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    debug_package = json.loads(debug_package_path.read_text(encoding="utf-8"))
    enriched = RUNNER["EnrichedRallyFactV3"].model_validate_json(
        fact_path.read_text(encoding="utf-8")
    )
    RUNNER["validate_against_package"](enriched, package)

    info = _video_info(video_path)
    rally = package["rally"]
    expected_frames = rally["end_frame"] - rally["start_frame"] + 1
    actual_fps = float(Fraction(info["avg_frame_rate"]))
    if (int(info["width"]), int(info["height"])) != (1920, 1080):
        raise ValueError("visualization currently requires a 1920x1080 source video")
    if not math.isclose(actual_fps, rally["fps"], abs_tol=1e-6):
        raise ValueError("source video fps does not match the rally package")
    if int(info["nb_frames"]) != expected_frames:
        raise ValueError(
            f"source video has {info['nb_frames']} frames; expected {expected_frames}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character.lower() if character.isalnum() else "_"
        for character in model_label
    ).strip("_")
    subtitle_path = output_dir / f"seg0144_{safe_label}_v3_visualization.ass"
    video_output = output_dir / f"seg0144_{safe_label}_v3_visualization.mp4"
    if not overwrite and (subtitle_path.exists() or video_output.exists()):
        raise FileExistsError("visualization already exists; pass --overwrite")
    subtitle_path.write_text(
        render_ass(
            enriched,
            package,
            debug_package=debug_package,
            model_label=model_label,
        ),
        encoding="utf-8-sig",
    )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y" if overwrite else "-n",
            "-i",
            str(video_path.resolve()),
            "-vf",
            f"ass=filename='{_ffmpeg_filter_path(subtitle_path)}'",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(video_output.resolve()),
        ],
        check=True,
    )
    return subtitle_path, video_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize validated Gemini v3 observations on one rally clip."
    )
    parser.add_argument("--fact", type=Path, default=DEFAULT_FACT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--debug-package",
        type=Path,
        default=DEFAULT_DEBUG_PACKAGE,
        help="Full pose-window artifact used only for skeleton rendering.",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-label", default="gemini-3.1-pro")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    subtitle, video = visualize(
        fact_path=args.fact,
        package_path=args.package,
        debug_package_path=args.debug_package,
        video_path=args.video,
        output_dir=args.output,
        model_label=args.model_label,
        overwrite=args.overwrite,
    )
    print(f"subtitle: {subtitle.resolve()}")
    print(f"video: {video.resolve()}")


if __name__ == "__main__":
    main()

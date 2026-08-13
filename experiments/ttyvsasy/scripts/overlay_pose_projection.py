"""Overlay event pose and court projection evidence on source video frames."""

from __future__ import annotations

import argparse
import base64
import html
import runpy
import shutil
import struct
import subprocess
from pathlib import Path

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    read_upstream_stages,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_ROOT = EXPERIMENT_ROOT / "workspace" / "stages"
DEFAULT_VIDEO = EXPERIMENT_ROOT / "workspace" / "video" / "TTYvsASY.mp4"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "ttyvsasy"
    / "direct_rallyfact"
    / "seg0144"
    / "geometry"
    / "frame_overlays"
)
PACKAGE = runpy.run_path(str(Path(__file__).with_name("package_direct_rallyfact.py")))
VISUALIZER = runpy.run_path(
    str(Path(__file__).with_name("visualize_rally_geometry.py"))
)

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


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG image: {path}")
    return struct.unpack(">II", signature[16:24])


def _court_to_image(
    homography: list[list[float]],
    point: tuple[float, float],
) -> tuple[float, float] | None:
    x, y = point
    denominator = homography[2][0] * x + homography[2][1] * y + homography[2][2]
    if abs(denominator) < 1e-12:
        return None
    return (
        (homography[0][0] * x + homography[0][1] * y + homography[0][2])
        / denominator,
        (homography[1][0] * x + homography[1][1] * y + homography[1][2])
        / denominator,
    )


def _extract_frame(
    *,
    video: Path,
    absolute_frame: int,
    fps: float,
    output: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{absolute_frame / fps:.9f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )


def render_overlay_svg(
    *,
    background_data_url: str,
    width: int,
    height: int,
    event,
    diagnostic,
    court_corners: list[tuple[float, float]],
    homography: list[list[float]],
    fps: float,
) -> str:
    pose = min(event.pose_window, key=lambda item: abs(item.frame_delta))
    keypoints = pose.keypoints
    ankle_midpoint = diagnostic.image_point
    projected = diagnostic.court_point_m
    corner_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in court_corners)
    net_left = _court_to_image(homography, (0, PACKAGE["COURT_HALF_LENGTH_M"]))
    net_right = _court_to_image(
        homography,
        (PACKAGE["COURT_WIDTH_M"], PACKAGE["COURT_HALF_LENGTH_M"]),
    )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Segoe UI,Arial,sans-serif}.label{font-size:24px;font-weight:700}.small{font-size:19px}</style>",
        f'<image href="{background_data_url}" width="{width}" height="{height}"/>',
        f'<polygon points="{corner_text}" fill="none" stroke="#22d3ee" stroke-width="5" opacity="0.9"/>',
    ]
    if net_left is not None and net_right is not None:
        elements.append(
            f'<line x1="{net_left[0]}" y1="{net_left[1]}" x2="{net_right[0]}" y2="{net_right[1]}" stroke="#fde047" stroke-width="5" opacity="0.9"/>'
        )

    for start, end in SKELETON_EDGES:
        x1, y1, c1 = keypoints[start]
        x2, y2, c2 = keypoints[end]
        opacity = max(0.25, min(1.0, (c1 + c2) / 2))
        elements.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4ade80" stroke-width="5" opacity="{opacity:.3f}"/>'
        )
    for name, (x, y, confidence) in keypoints.items():
        color = "#facc15" if "ankle" in name else "#4ade80"
        elements.append(
            f'<circle cx="{x}" cy="{y}" r="{7 if "ankle" in name else 5}" fill="{color}" stroke="#052e16" stroke-width="2" opacity="{max(0.3, min(1.0, confidence)):.3f}"/>'
        )

    left, top, right, bottom = pose.bbox
    elements.append(
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#4ade80" stroke-width="3" stroke-dasharray="10 7"/>'
    )
    if ankle_midpoint is not None:
        ax, ay = ankle_midpoint
        elements.extend(
            [
                f'<circle cx="{ax}" cy="{ay}" r="13" fill="#ef4444" stroke="white" stroke-width="4"/>',
                f'<line x1="{ax-18}" y1="{ay}" x2="{ax+18}" y2="{ay}" stroke="white" stroke-width="3"/>',
                f'<line x1="{ax}" y1="{ay-18}" x2="{ax}" y2="{ay+18}" stroke="white" stroke-width="3"/>',
            ]
        )

    panel_x, panel_y, panel_w, panel_h = 35, 175, 620, 205
    point_text = (
        f"court=({projected[0]:.3f}, {projected[1]:.3f}) m"
        if projected is not None
        else "court=unavailable"
    )
    elements.extend(
        [
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="12" fill="#020617" opacity="0.82"/>',
            f'<text x="{panel_x+20}" y="{panel_y+38}" class="label" fill="white">event {event.event_index} · frame {event.frame}</text>',
            f'<text x="{panel_x+20}" y="{panel_y+74}" class="small" fill="#cbd5e1">video time={event.frame/fps:.3f}s · player={html.escape(str(event.player))} ({html.escape(str(event.stage_player))})</text>',
            f'<text x="{panel_x+20}" y="{panel_y+108}" class="small" fill="#cbd5e1">pose source={pose.frame} · {html.escape(str(diagnostic.position_source))}</text>',
            f'<text x="{panel_x+20}" y="{panel_y+142}" class="small" fill="#fecaca">image=({ankle_midpoint[0]:.2f}, {ankle_midpoint[1]:.2f}) · {point_text}</text>',
            f'<text x="{panel_x+20}" y="{panel_y+176}" class="small" fill="#fca5a5">status={html.escape(diagnostic.status)}</text>',
        ]
    )

    mini_margin = 1.5
    mini_scale = 24
    mini_x = width - 285
    mini_y = 80
    court_x = mini_x + mini_margin * mini_scale
    court_y = mini_y + mini_margin * mini_scale
    court_w = PACKAGE["COURT_WIDTH_M"] * mini_scale
    court_h = PACKAGE["COURT_LENGTH_M"] * mini_scale
    elements.extend(
        [
            f'<rect x="{mini_x}" y="{mini_y}" width="{(PACKAGE["COURT_WIDTH_M"]+2*mini_margin)*mini_scale}" height="{(PACKAGE["COURT_LENGTH_M"]+2*mini_margin)*mini_scale}" rx="10" fill="#020617" opacity="0.82"/>',
            f'<rect x="{court_x}" y="{court_y}" width="{court_w}" height="{court_h}" fill="#166534" stroke="white" stroke-width="3"/>',
            f'<line x1="{court_x}" y1="{court_y+PACKAGE["COURT_HALF_LENGTH_M"]*mini_scale}" x2="{court_x+court_w}" y2="{court_y+PACKAGE["COURT_HALF_LENGTH_M"]*mini_scale}" stroke="#fde047" stroke-width="3"/>',
        ]
    )
    if projected is not None:
        px = court_x + projected[0] * mini_scale
        py = court_y + projected[1] * mini_scale
        elements.extend(
            [
                f'<circle cx="{px}" cy="{py}" r="8" fill="#ef4444" stroke="white" stroke-width="3"/>',
                f'<text x="{px+12}" y="{py+5}" class="small" fill="white">{event.event_index}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{mini_x+12}" y="{mini_y+24}" class="small" fill="white">court plane (metres)</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def build_overlays(
    *,
    video: Path,
    stage_root: Path,
    segment_index: int,
    event_indexes: list[int],
    mapping: CourtPositionToPlayer,
    output_dir: Path,
) -> list[Path]:
    stages = read_upstream_stages(
        StagePaths.from_stage_root(stage_root),
        segment_index=segment_index,
    )
    stage_input = PACKAGE["_build_event_centric_input"](
        stages=stages,
        segment_index=segment_index,
        mapping=mapping,
    )
    geometry = VISUALIZER["build_geometry_report"](
        stages=stages,
        segment_index=segment_index,
        mapping=mapping,
    )
    events = {item.event_index: item for item in stage_input.events}
    diagnostics = {item.event_index: item for item in geometry.diagnostics}
    missing = sorted(set(event_indexes) - set(events))
    if missing:
        raise ValueError(f"event indexes are outside selected rally: {missing}")

    court_stage = stages.vision.court_detection if stages.vision else None
    if court_stage is None or court_stage.detection_failed:
        raise ValueError("court calibration is unavailable")
    exact = [
        item for item in court_stage.courts if item.segment_index == segment_index
    ]
    candidates = exact or [item for item in court_stage.courts if item.segment_index is None]
    if len(candidates) != 1:
        raise ValueError("court calibration is missing or ambiguous")
    calibration = candidates[0]
    fps = stages.match_segmentation.fps
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for event_index in event_indexes:
        event = events[event_index]
        diagnostic = diagnostics[event_index]
        source_png = output_dir / f"event_{event_index}_source.png"
        _extract_frame(
            video=video,
            absolute_frame=event.frame,
            fps=fps,
            output=source_png,
        )
        width, height = _png_dimensions(source_png)
        background = "data:image/png;base64," + base64.b64encode(
            source_png.read_bytes()
        ).decode("ascii")
        svg = render_overlay_svg(
            background_data_url=background,
            width=width,
            height=height,
            event=event,
            diagnostic=diagnostic,
            court_corners=calibration.corners,
            homography=calibration.homography,
            fps=fps,
        )
        output = output_dir / f"event_{event_index}_projection_overlay.svg"
        output.write_text(svg + "\n", encoding="utf-8")
        outputs.append(output)

    cards = "".join(
        f'<section><h2>Event {index}</h2><img src="event_{index}_projection_overlay.svg" alt="Event {index} projection overlay"></section>'
        for index in event_indexes
    )
    (output_dir / "index.html").write_text(
        "<!doctype html><html lang=\"zh-Hant\"><head><meta charset=\"utf-8\">"
        "<title>Pose projection overlays</title><style>body{font-family:Segoe UI,Arial,sans-serif;background:#0f172a;color:white;margin:24px}img{width:100%;height:auto;border:1px solid #475569}section{margin-bottom:32px}</style></head>"
        f"<body><h1>SEG{segment_index} pose/court projection overlays</h1>{cards}</body></html>\n",
        encoding="utf-8",
    )
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay pose and court projection on source video frames."
    )
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--event-index", type=int, action="append", required=True)
    parser.add_argument("--top-player", choices=("a", "b"), required=True)
    parser.add_argument("--bottom-player", choices=("a", "b"), required=True)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = build_overlays(
        video=args.video,
        stage_root=args.stage_root,
        segment_index=args.segment_index,
        event_indexes=args.event_index,
        mapping=CourtPositionToPlayer(
            top=args.top_player,
            bottom=args.bottom_player,
        ),
        output_dir=args.output,
    )
    print(f"index: {(args.output / 'index.html').resolve()}")
    for output in outputs:
        print(f"overlay: {output.resolve()}")


if __name__ == "__main__":
    main()

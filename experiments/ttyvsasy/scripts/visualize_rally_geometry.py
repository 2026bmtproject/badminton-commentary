"""Render deterministic court-projection diagnostics for one selected rally."""

from __future__ import annotations

import argparse
import html
import json
import math
import runpy
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    UpstreamStageData,
    read_upstream_stages,
)
from badminton_commentary.schemas import Player, Probability


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_ROOT = EXPERIMENT_ROOT / "workspace" / "stages"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ttyvsasy" / "direct_rallyfact"
PACKAGE_SCRIPT = Path(__file__).with_name("package_direct_rallyfact.py")

_PACKAGE = runpy.run_path(str(PACKAGE_SCRIPT))
COURT_WIDTH_M = _PACKAGE["COURT_WIDTH_M"]
COURT_LENGTH_M = _PACKAGE["COURT_LENGTH_M"]
COURT_HALF_LENGTH_M = _PACKAGE["COURT_HALF_LENGTH_M"]
COURT_BASELINE_EXTENSION_M = _PACKAGE["COURT_BASELINE_EXTENSION_M"]
POSE_PRE_FRAMES = _PACKAGE["POSE_PRE_FRAMES"]
POSE_POST_FRAMES = _PACKAGE["POSE_POST_FRAMES"]


class DiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeometryDiagnostic(DiagnosticModel):
    event_index: int
    frame: int
    stage_player: Literal["top", "bottom"] | None
    player: Player | None
    source_frame: int | None
    frame_delta: int | None
    position_source: str | None
    image_point: tuple[float, float] | None
    court_point_m: tuple[float, float] | None
    projection_confidence: Probability | None
    within_court_bounds: bool | None
    within_player_half: bool | None
    accepted_by_package: bool
    depth_zone: str | None
    position_change_from_previous_same_player_hit: str | None
    status: Literal[
        "accepted",
        "accepted_behind_baseline",
        "outside_player_half",
        "outside_court_bounds",
        "projection_unavailable",
        "pose_unavailable",
        "player_unavailable",
        "calibration_unavailable",
    ]
    limitations: list[str]


class GeometryReport(DiagnosticModel):
    report_version: Literal["court-geometry-diagnostic-v1"]
    segment_index: int
    court_width_m: float
    court_length_m: float
    homography_direction: Literal["court_to_image"]
    calibration_confirmed: bool | None
    detection_failed: bool | None
    inverse_homography_available: bool
    player_mapping: dict[str, Player]
    accepted_count: int
    event_count: int
    diagnostics: list[GeometryDiagnostic]


def _selected_source_pose(stages, *, event_frame: int, stage_player: str):
    vision = stages.vision
    if vision is None or vision.pose is None:
        return None
    candidates = [
        pose
        for pose in vision.pose.frames
        if pose.player == stage_player
        and event_frame - POSE_PRE_FRAMES
        <= pose.frame
        <= event_frame + POSE_POST_FRAMES
    ]
    return (
        min(candidates, key=lambda pose: (abs(pose.frame - event_frame), pose.frame))
        if candidates
        else None
    )


def build_geometry_report(
    *,
    stages: UpstreamStageData,
    segment_index: int,
    mapping: CourtPositionToPlayer,
) -> GeometryReport:
    stage_input = _PACKAGE["_build_event_centric_input"](
        stages=stages,
        segment_index=segment_index,
        mapping=mapping,
    )
    homography = _PACKAGE["_select_court_calibration"](
        stages,
        segment_index=segment_index,
    )
    inverse = _PACKAGE["_inverse_3x3"](homography) if homography else None
    court_stage = (
        stages.vision.court_detection
        if stages.vision is not None
        else None
    )
    diagnostics: list[GeometryDiagnostic] = []

    for event in stage_input.events:
        accepted = event.court_position
        base = {
            "event_index": event.event_index,
            "frame": event.frame,
            "stage_player": event.stage_player,
            "player": event.player,
            "accepted_by_package": accepted is not None,
            "depth_zone": accepted.depth_zone if accepted else None,
            "position_change_from_previous_same_player_hit": (
                accepted.position_change_from_previous_same_player_hit
                if accepted
                else None
            ),
        }
        if event.stage_player is None:
            diagnostics.append(
                GeometryDiagnostic(
                    **base,
                    source_frame=None,
                    frame_delta=None,
                    position_source=None,
                    image_point=None,
                    court_point_m=None,
                    projection_confidence=None,
                    within_court_bounds=None,
                    within_player_half=None,
                    status="player_unavailable",
                    limitations=["event_player_unavailable"],
                )
            )
            continue

        pose = _selected_source_pose(
            stages,
            event_frame=event.frame,
            stage_player=event.stage_player,
        )
        if pose is None:
            diagnostics.append(
                GeometryDiagnostic(
                    **base,
                    source_frame=None,
                    frame_delta=None,
                    position_source=None,
                    image_point=None,
                    court_point_m=None,
                    projection_confidence=None,
                    within_court_bounds=None,
                    within_player_half=None,
                    status="pose_unavailable",
                    limitations=["pose_window_has_no_hitting_player_record"],
                )
            )
            continue

        image_point, source, confidence, limitations = _PACKAGE[
            "_court_source_from_pose"
        ](pose)
        common = {
            **base,
            "source_frame": pose.frame,
            "frame_delta": pose.frame - event.frame,
            "position_source": source,
            "image_point": image_point,
            "projection_confidence": confidence,
        }
        if inverse is None:
            diagnostics.append(
                GeometryDiagnostic(
                    **common,
                    court_point_m=None,
                    within_court_bounds=None,
                    within_player_half=None,
                    status="calibration_unavailable",
                    limitations=[*limitations, "inverse_homography_unavailable"],
                )
            )
            continue

        projected = _PACKAGE["_project"](inverse, image_point)
        if projected is None or not all(math.isfinite(value) for value in projected):
            diagnostics.append(
                GeometryDiagnostic(
                    **common,
                    court_point_m=None,
                    within_court_bounds=None,
                    within_player_half=None,
                    status="projection_unavailable",
                    limitations=[*limitations, "projection_denominator_invalid"],
                )
            )
            continue

        court_x, court_y = projected
        within_court = (
            0 <= court_x <= COURT_WIDTH_M and 0 <= court_y <= COURT_LENGTH_M
        )
        relative_depth = _PACKAGE["_player_relative_depth"](
            court_y,
            event.stage_player,
        )
        within_half = relative_depth is not None
        if accepted is not None and not within_court:
            status = "accepted_behind_baseline"
            limitations = list(
                dict.fromkeys(
                    [
                        *accepted.limitations,
                        "outside_official_court_but_within_baseline_extension",
                    ]
                )
            )
        elif not within_court:
            status = "outside_court_bounds"
            limitations = [*limitations, "projected_point_outside_court_bounds"]
        elif not within_half:
            status = "outside_player_half"
            limitations = [*limitations, "projected_point_outside_player_half"]
        else:
            status = "accepted"
        diagnostics.append(
            GeometryDiagnostic(
                **common,
                court_point_m=projected,
                within_court_bounds=within_court,
                within_player_half=within_half,
                status=status,
                limitations=limitations,
            )
        )

    return GeometryReport(
        report_version="court-geometry-diagnostic-v1",
        segment_index=segment_index,
        court_width_m=COURT_WIDTH_M,
        court_length_m=COURT_LENGTH_M,
        homography_direction="court_to_image",
        calibration_confirmed=(court_stage.confirmed if court_stage else None),
        detection_failed=(court_stage.detection_failed if court_stage else None),
        inverse_homography_available=inverse is not None,
        player_mapping=mapping.model_dump(mode="json"),
        accepted_count=sum(item.accepted_by_package for item in diagnostics),
        event_count=len(diagnostics),
        diagnostics=diagnostics,
    )


def _svg(report: GeometryReport) -> str:
    scale = 45.0
    margin_m = 2.0
    plot_left = 85.0
    plot_top = 65.0
    min_x = -margin_m
    min_y = -margin_m
    plot_height = (COURT_LENGTH_M + 2 * margin_m) * scale
    canvas_width = 980
    canvas_height = int(plot_top + plot_height + 90)

    def sx(x: float) -> float:
        return plot_left + (x - min_x) * scale

    def sy(y: float) -> float:
        return plot_top + (y - min_y) * scale

    def clipped(point: tuple[float, float]) -> tuple[float, float]:
        return (
            min(COURT_WIDTH_M + margin_m, max(-margin_m, point[0])),
            min(COURT_LENGTH_M + margin_m, max(-margin_m, point[1])),
        )

    zone_boundaries = [
        COURT_HALF_LENGTH_M / 3,
        2 * COURT_HALF_LENGTH_M / 3,
        COURT_HALF_LENGTH_M,
        COURT_HALF_LENGTH_M + COURT_HALF_LENGTH_M / 3,
        COURT_HALF_LENGTH_M + 2 * COURT_HALF_LENGTH_M / 3,
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" '
        f'height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        "<style>text{font-family:Segoe UI,Arial,sans-serif}"
        ".label{font-size:11px;font-weight:600}.small{font-size:12px}"
        ".title{font-size:22px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="35" y="32" class="title">SEG{report.segment_index} court geometry validation</text>',
        f'<rect x="{sx(0)}" y="{sy(0)}" width="{COURT_WIDTH_M * scale}" '
        f'height="{COURT_LENGTH_M * scale}" fill="#dcfce7" stroke="#0f172a" stroke-width="2"/>',
    ]
    for boundary in zone_boundaries:
        is_net = math.isclose(boundary, COURT_HALF_LENGTH_M)
        parts.append(
            f'<line x1="{sx(0)}" y1="{sy(boundary)}" x2="{sx(COURT_WIDTH_M)}" '
            f'y2="{sy(boundary)}" stroke="#{"0f172a" if is_net else "64748b"}" '
            f'stroke-width="{3 if is_net else 1}" '
            f'stroke-dasharray="{"none" if is_net else "5 5"}"/>'
        )
    parts.extend(
        [
            f'<text x="{sx(COURT_WIDTH_M) + 18}" y="{sy(1.1)}" class="small">TOP rear</text>',
            f'<text x="{sx(COURT_WIDTH_M) + 18}" y="{sy(3.35)}" class="small">TOP mid</text>',
            f'<text x="{sx(COURT_WIDTH_M) + 18}" y="{sy(5.6)}" class="small">TOP front</text>',
            f'<text x="{sx(COURT_WIDTH_M) + 18}" y="{sy(7.8)}" class="small">BOTTOM front</text>',
            f'<text x="{sx(COURT_WIDTH_M) + 18}" y="{sy(10.05)}" class="small">BOTTOM mid</text>',
            f'<text x="{sx(COURT_WIDTH_M) + 18}" y="{sy(12.3)}" class="small">BOTTOM rear</text>',
        ]
    )

    for stage_player, color in (("top", "#2563eb"), ("bottom", "#d97706")):
        points = [
            clipped(item.court_point_m)
            for item in report.diagnostics
            if item.stage_player == stage_player
            and item.court_point_m is not None
            and item.accepted_by_package
        ]
        if len(points) > 1:
            coordinate_text = " ".join(f"{sx(x)},{sy(y)}" for x, y in points)
            parts.append(
                f'<polyline points="{coordinate_text}" fill="none" stroke="{color}" '
                'stroke-width="1.5" stroke-dasharray="4 4" opacity="0.55"/>'
            )

    for item in report.diagnostics:
        if item.court_point_m is None:
            continue
        x, y = clipped(item.court_point_m)
        if item.status in {"accepted", "accepted_behind_baseline"}:
            color = "#2563eb" if item.stage_player == "top" else "#d97706"
            shape = f'<circle cx="{sx(x)}" cy="{sy(y)}" r="7" fill="{color}"/>'
        elif item.status == "outside_player_half":
            shape = f'<circle cx="{sx(x)}" cy="{sy(y)}" r="8" fill="#7c3aed"/>'
        else:
            shape = (
                f'<path d="M {sx(x)-7} {sy(y)-7} L {sx(x)+7} {sy(y)+7} '
                f'M {sx(x)+7} {sy(y)-7} L {sx(x)-7} {sy(y)+7}" '
                'stroke="#dc2626" stroke-width="4"/>'
            )
        parts.extend(
            [
                shape,
                f'<text x="{sx(x)+9}" y="{sy(y)-8}" class="label" fill="#0f172a">{item.event_index}</text>',
            ]
        )

    legend_x = sx(COURT_WIDTH_M) + 18
    legend_y = sy(COURT_LENGTH_M) - 75
    parts.extend(
        [
            f'<circle cx="{legend_x}" cy="{legend_y}" r="6" fill="#2563eb"/><text x="{legend_x+12}" y="{legend_y+4}" class="small">top accepted</text>',
            f'<circle cx="{legend_x}" cy="{legend_y+24}" r="6" fill="#d97706"/><text x="{legend_x+12}" y="{legend_y+28}" class="small">bottom accepted</text>',
            f'<circle cx="{legend_x}" cy="{legend_y+48}" r="6" fill="#7c3aed"/><text x="{legend_x+12}" y="{legend_y+52}" class="small">wrong player half</text>',
            f'<path d="M {legend_x-6} {legend_y+66} L {legend_x+6} {legend_y+78} M {legend_x+6} {legend_y+66} L {legend_x-6} {legend_y+78}" stroke="#dc2626" stroke-width="3"/><text x="{legend_x+12}" y="{legend_y+76}" class="small">outside court</text>',
            f'<text x="35" y="{canvas_height-35}" class="small">Court range: x=0..{COURT_WIDTH_M}m, y=0..{COURT_LENGTH_M}m · accepted {report.accepted_count}/{report.event_count}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def _html(report: GeometryReport, svg: str) -> str:
    rows = []
    for item in report.diagnostics:
        point = (
            f"({item.court_point_m[0]:.3f}, {item.court_point_m[1]:.3f})"
            if item.court_point_m
            else "—"
        )
        image_point = (
            f"({item.image_point[0]:.1f}, {item.image_point[1]:.1f})"
            if item.image_point
            else "—"
        )
        rows.append(
            "<tr>"
            f"<td>{item.event_index}</td><td>{html.escape(str(item.stage_player))}</td>"
            f"<td>{item.source_frame if item.source_frame is not None else '—'}</td>"
            f"<td>{html.escape(str(item.position_source or '—'))}</td>"
            f"<td>{image_point}</td><td>{point}</td>"
            f"<td>{html.escape(item.status)}</td>"
            f"<td>{html.escape(str(item.depth_zone or '—'))}</td>"
            f"<td>{html.escape(', '.join(item.limitations) or '—')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>SEG{report.segment_index} geometry validation</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#0f172a}}svg{{max-width:100%;height:auto;border:1px solid #cbd5e1}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #cbd5e1;padding:6px;text-align:left}}th{{background:#e2e8f0}}tr:nth-child(even){{background:#f8fafc}}code{{background:#f1f5f9;padding:2px 4px}}</style></head>
<body><h1>SEG{report.segment_index} deterministic court geometry</h1>
<p><code>within_court_bounds</code> 檢查 6.1×13.4m；<code>within_player_half</code> 再檢查 top/bottom 所屬半場。紅色叉號是球場外投影。</p>
<p>Homography: <code>{report.homography_direction}</code>；confirmed=<code>{report.calibration_confirmed}</code>；detection_failed=<code>{report.detection_failed}</code>；inverse available=<code>{report.inverse_homography_available}</code>。</p>
{svg}
<h2>逐 event 診斷</h2><table><thead><tr><th>event</th><th>side</th><th>source frame</th><th>source</th><th>image point</th><th>court point (m)</th><th>status</th><th>zone</th><th>limitations</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def write_geometry_report(report: GeometryReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    (output_dir / "court_geometry_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    svg = _svg(report)
    (output_dir / "court_geometry.svg").write_text(svg + "\n", encoding="utf-8")
    (output_dir / "court_geometry.html").write_text(
        _html(report, svg) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize deterministic court geometry for one rally."
    )
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--top-player", choices=("a", "b"), required=True)
    parser.add_argument("--bottom-player", choices=("a", "b"), required=True)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Accepted for CLI consistency; no provider is called.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mapping = CourtPositionToPlayer(
        top=args.top_player,
        bottom=args.bottom_player,
    )
    stages = read_upstream_stages(
        StagePaths.from_stage_root(args.stage_root),
        segment_index=args.segment_index,
    )
    report = build_geometry_report(
        stages=stages,
        segment_index=args.segment_index,
        mapping=mapping,
    )
    output_dir = args.output or (
        DEFAULT_OUTPUT_ROOT / f"seg{args.segment_index:04d}" / "geometry"
    )
    write_geometry_report(report, output_dir)
    status_counts: dict[str, int] = {}
    for item in report.diagnostics:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
    print(f"segment: {report.segment_index}")
    print(f"events: {report.event_count}")
    print(f"accepted: {report.accepted_count}/{report.event_count}")
    print(f"statuses: {json.dumps(status_counts, ensure_ascii=False)}")
    print(f"html: {(output_dir / 'court_geometry.html').resolve()}")
    print(f"svg: {(output_dir / 'court_geometry.svg').resolve()}")
    print(f"json: {(output_dir / 'court_geometry_report.json').resolve()}")


if __name__ == "__main__":
    main()

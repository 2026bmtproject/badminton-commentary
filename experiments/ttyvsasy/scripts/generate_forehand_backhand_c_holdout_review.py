"""Generate court-prior-only hold-out videos and human-reference viewers."""

from __future__ import annotations

import argparse
import html
import json
import runpy
from dataclasses import asdict
from pathlib import Path

from badminton_commentary.adapters import CourtPositionToPlayer


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT = runpy.run_path(str(SCRIPT_DIR / "experiment_forehand_backhand.py"))
Config = EXPERIMENT["Config"]
analyze_segment = EXPERIMENT["analyze_segment"]
render_ass = EXPERIMENT["render_ass"]
render_video = EXPERIMENT["render_video"]
render_viewer = EXPERIMENT["render_viewer"]

DEFAULT_STAGE_ROOT = (
    REPO_ROOT / "experiments" / "ttyvsasy" / "workspace" / "stages"
)
DEFAULT_VIDEO = (
    REPO_ROOT
    / "experiments"
    / "ttyvsasy"
    / "workspace"
    / "video"
    / "TTYvsASY.mp4"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "ttyvsasy"
    / "forehand_backhand"
    / "holdout_c_review"
)
DEFAULT_SEGMENTS = [130, 132, 136, 139, 144, 146]


def render_index(manifest: dict[str, object]) -> str:
    cards = []
    for item in manifest["rallies"]:
        segment = int(item["segment_index"])
        folder = f"seg{segment:04d}_C_court_prior"
        cards.append(
            "<article><h2>SEG{segment}</h2>"
            "<p>{hits} hits · {duration:.2f} sec · {unknown} unknown candidates</p>"
            "<p><a href='{folder}/frame_review.html'>開始／繼續人工審核</a> · "
            "<a href='{folder}/{video}'>開啟疊圖影片</a></p></article>".format(
                segment=segment,
                hits=item["hits"],
                duration=item["duration_sec"],
                unknown=item["unknown_candidates"],
                folder=folder,
                video=html.escape(str(item["video_name"])),
            )
        )
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>C 版本正反手 Hold-out 人工審核</title>
<style>
body{{font-family:Segoe UI,Microsoft JhengHei,sans-serif;max-width:960px;margin:32px auto;background:#0f172a;color:#e2e8f0}}
article{{background:#1e293b;border-radius:12px;padding:16px 22px;margin:14px 0}}
a{{color:#7dd3fc}} code{{background:#334155;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>C 版本正反手 Hold-out 人工審核</h1>
<p>固定策略：<code>orientation_policy=court_prior</code>。請在各頁使用
<code>F</code> 正手、<code>B</code> 反手、<code>U</code> 無法判定，最後按
<code>E</code> 匯出 JSON。</p>
<p>SEG144 是指定的 C 版複審；其餘 rally 未參與 A/B/C/D 選型，可作為本輪 hold-out。</p>
{''.join(cards)}
</body></html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=DEFAULT_SEGMENTS,
    )
    parser.add_argument("--top-player", choices=("a", "b"), default="b")
    parser.add_argument("--bottom-player", choices=("a", "b"), default="a")
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config(orientation_policy="court_prior")
    mapping = CourtPositionToPlayer(
        top=args.top_player,
        bottom=args.bottom_player,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rallies = []
    for segment_index in args.segments:
        print(f"analyzing SEG{segment_index} with C/court_prior")
        result, pose_lookup = analyze_segment(
            stage_root=args.stage_root,
            segment_index=segment_index,
            mapping=mapping,
            left_handed_players=set(),
            config=config,
        )
        folder = args.output / f"seg{segment_index:04d}_C_court_prior"
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "forehand_backhand_results.json"
        subtitle_path = folder / "forehand_backhand_overlay.ass"
        video_name = f"seg{segment_index:04d}_C_forehand_backhand_overlay.mp4"
        video_path = folder / video_name
        viewer_path = folder / "frame_review.html"
        result_path.write_text(
            result.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        subtitle_path.write_text(
            render_ass(result, pose_lookup),
            encoding="utf-8-sig",
        )
        if not args.skip_video:
            render_video(
                video=args.video,
                subtitles=subtitle_path,
                output=video_path,
                source_start_frame=result.source_start_frame,
                source_end_frame=result.source_end_frame,
                fps=result.fps,
            )
        viewer_path.write_text(
            render_viewer(result, video_name),
            encoding="utf-8",
        )
        rallies.append(
            {
                "segment_index": segment_index,
                "hits": len(result.shots),
                "duration_sec": (
                    (result.source_end_frame - result.source_start_frame + 1)
                    / result.fps
                ),
                "unknown_candidates": sum(
                    shot.side is None for shot in result.shots
                ),
                "video_name": video_name,
                "result_path": str(result_path),
                "viewer_path": str(viewer_path),
            }
        )
    manifest = {
        "schema_version": "forehand-backhand-holdout-review-set-v1",
        "variant": "C",
        "config": asdict(config),
        "segments": args.segments,
        "selection": {
            "requested_recheck": [144],
            "new_holdout": [item for item in args.segments if item != 144],
            "basis": (
                "Selected before viewing C outcomes to cover short, medium, and long "
                "rallies; no candidate prediction or human verdict was used."
            ),
        },
        "rallies": rallies,
    }
    manifest_path = args.output / "review_manifest.json"
    index_path = args.output / "index.html"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index_path.write_text(render_index(manifest), encoding="utf-8")
    print(f"manifest: {manifest_path.resolve()}")
    print(f"review index: {index_path.resolve()}")


if __name__ == "__main__":
    main()

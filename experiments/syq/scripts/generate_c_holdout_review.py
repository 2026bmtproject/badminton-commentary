"""Generate the frozen C review set for the SYQ cross-match hold-out."""

from __future__ import annotations

import argparse
import html
import json
import runpy
from dataclasses import asdict
from pathlib import Path

from badminton_commentary.adapters import CourtPositionToPlayer


REPO_ROOT = Path(__file__).resolve().parents[3]
SYQ_ROOT = REPO_ROOT / "experiments" / "syq"
TTY_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "ttyvsasy"
    / "scripts"
    / "experiment_forehand_backhand.py"
)
EXPERIMENT = runpy.run_path(str(TTY_SCRIPT))
Config = EXPERIMENT["Config"]
analyze_segment = EXPERIMENT["analyze_segment"]
render_ass = EXPERIMENT["render_ass"]
render_video = EXPERIMENT["render_video"]
render_viewer = EXPERIMENT["render_viewer"]

DEFAULT_SELECTION = SYQ_ROOT / "holdout_selection.json"
DEFAULT_STAGE_ROOT = SYQ_ROOT / "workspace" / "stages"
DEFAULT_VIDEO = SYQ_ROOT / "workspace" / "video" / "syq.mp4"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "syq" / "forehand_backhand" / "holdout_c_review"


def render_index(manifest: dict[str, object]) -> str:
    cards = []
    for item in manifest["rallies"]:
        segment = int(item["segment_index"])
        folder = f"seg{segment:04d}_C_court_prior"
        mapping = item["player_mapping"]
        cards.append(
            "<article><h2>SEG{segment}</h2>"
            "<p>{hits} hits · {duration:.2f} sec · {unknown} unknown candidates</p>"
            "<p>top={top} · bottom={bottom}</p>"
            "<p><a href='{folder}/frame_review.html'>開始／繼續人工審核</a> · "
            "<a href='{folder}/{video}'>開啟疊圖影片</a></p></article>".format(
                segment=segment,
                hits=item["hits"],
                duration=item["duration_sec"],
                unknown=item["unknown_candidates"],
                top=html.escape(str(mapping["top_name"])),
                bottom=html.escape(str(mapping["bottom_name"])),
                folder=folder,
                video=html.escape(str(item["video_name"])),
            )
        )
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>SYQ C 版本跨場次 Hold-out 審核</title>
<style>
body{{font-family:Segoe UI,Microsoft JhengHei,sans-serif;max-width:960px;margin:32px auto;background:#0f172a;color:#e2e8f0}}
article{{background:#1e293b;border-radius:12px;padding:16px 22px;margin:14px 0}}
a{{color:#7dd3fc}} code{{background:#334155;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>SYQ C 版本跨場次 Hold-out 審核</h1>
<p>策略與 rally 已在 classifier 執行前凍結。請按
<code>F</code> 正手、<code>B</code> 反手、<code>U</code> 無法判定，最後按
<code>E</code> 匯出每個 segment 的 v3 reference JSON。</p>
{''.join(cards)}
</body></html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if not selection.get("selected_before_classifier_run"):
        raise ValueError("SYQ hold-out selection must be frozen before generation")
    config = Config(orientation_policy="court_prior")
    args.output.mkdir(parents=True, exist_ok=True)
    names = selection["players"]
    rallies = []
    for selected in selection["segments"]:
        segment_index = int(selected["segment_index"])
        mapping = CourtPositionToPlayer(
            top=selected["top_player"],
            bottom=selected["bottom_player"],
        )
        print(f"analyzing SYQ SEG{segment_index} with frozen C")
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
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        subtitle_path.write_text(
            render_ass(result, pose_lookup), encoding="utf-8-sig"
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
            render_viewer(result, video_name), encoding="utf-8"
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
                "player_mapping": {
                    "top": mapping.top,
                    "bottom": mapping.bottom,
                    "top_name": names[mapping.top],
                    "bottom_name": names[mapping.bottom],
                },
                "video_name": video_name,
            }
        )
    manifest = {
        "schema_version": "syq-forehand-backhand-holdout-review-set-v1",
        "selection": selection,
        "config": asdict(config),
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

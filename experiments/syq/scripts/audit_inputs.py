"""Audit SYQ video and upstream stage alignment without running the classifier."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from badminton_commentary.adapters import StagePaths, read_upstream_stages
from badminton_commentary.adapters.vision import (
    PoseFrame,
    ShuttlePoint,
    _iter_top_level_array,
    read_court_detection_stage,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_ROOT = REPO_ROOT / "experiments" / "syq" / "workspace" / "stages"
DEFAULT_VIDEO = REPO_ROOT / "experiments" / "syq" / "workspace" / "video" / "syq.mp4"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "syq" / "input_audit.json"


def _video_metadata(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
        "duration_sec": float(stream["duration"]),
    }


def audit_inputs(*, stage_root: Path, video: Path) -> dict[str, object]:
    paths = StagePaths.from_stage_root(stage_root)
    stages = read_upstream_stages(paths)
    segments = stages.match_segmentation.segments
    events = stages.event_detection.events
    strokes = stages.stroke_classification.strokes
    video_info = _video_metadata(video)

    join_errors = []
    strokes_by_segment: Counter[int] = Counter()
    for stroke in strokes:
        if stroke.event_index >= len(events):
            join_errors.append(
                {"event_index": stroke.event_index, "reason": "out_of_range"}
            )
            continue
        event = events[stroke.event_index]
        if stroke.frame != event.frame:
            join_errors.append(
                {"event_index": stroke.event_index, "reason": "frame_mismatch"}
            )
            continue
        if not 0 <= stroke.segment_index < len(segments):
            join_errors.append(
                {"event_index": stroke.event_index, "reason": "segment_out_of_range"}
            )
            continue
        segment = segments[stroke.segment_index]
        if not segment.start_frame <= stroke.frame <= segment.end_frame:
            join_errors.append(
                {"event_index": stroke.event_index, "reason": "outside_segment"}
            )
            continue
        if stroke.player is not None:
            strokes_by_segment[stroke.segment_index] += 1

    pose_total = 0
    pose_valid = 0
    pose_missing = 0
    pose_errors = []
    pose_valid_by_segment: Counter[int] = Counter()
    pose_missing_by_segment: Counter[int] = Counter()
    if paths.pose is None:
        raise FileNotFoundError("SYQ audit requires pose/pose.json")
    for raw in _iter_top_level_array(paths.pose, "frames"):
        pose_total += 1
        if not isinstance(raw, dict):
            pose_errors.append({"record": pose_total, "reason": "not_object"})
            continue
        segment_index = raw.get("segment_index")
        if raw.get("keypoints") is None or raw.get("bbox") is None:
            pose_missing += 1
            if isinstance(segment_index, int):
                pose_missing_by_segment[segment_index] += 1
            continue
        try:
            pose = PoseFrame.model_validate(raw)
        except ValueError as exc:
            pose_errors.append({"record": pose_total, "reason": str(exc)})
            continue
        pose_valid += 1
        pose_valid_by_segment[pose.segment_index] += 1
        if not 0 <= pose.segment_index < len(segments):
            pose_errors.append(
                {"record": pose_total, "reason": "segment_out_of_range"}
            )
        else:
            segment = segments[pose.segment_index]
            if not segment.start_frame <= pose.frame <= segment.end_frame:
                pose_errors.append(
                    {"record": pose_total, "reason": "outside_segment"}
                )

    shuttle_total = 0
    shuttle_errors = []
    shuttle_by_segment: Counter[int] = Counter()
    if paths.shuttle_tracking is not None:
        for raw in _iter_top_level_array(paths.shuttle_tracking, "points"):
            shuttle_total += 1
            try:
                point = ShuttlePoint.model_validate(raw)
            except ValueError as exc:
                shuttle_errors.append(
                    {"record": shuttle_total, "reason": str(exc)}
                )
                continue
            shuttle_by_segment[point.segment_index] += 1
            if not 0 <= point.segment_index < len(segments):
                shuttle_errors.append(
                    {"record": shuttle_total, "reason": "segment_out_of_range"}
                )
            else:
                segment = segments[point.segment_index]
                if not segment.start_frame <= point.frame <= segment.end_frame:
                    shuttle_errors.append(
                        {"record": shuttle_total, "reason": "outside_segment"}
                    )

    court = (
        read_court_detection_stage(paths.court_detection)
        if paths.court_detection is not None
        else None
    )
    segment_rows = [
        {
            "segment_index": index,
            "start_frame": segment.start_frame,
            "end_frame": segment.end_frame,
            "duration_sec": segment.duration_sec,
            "player_strokes": strokes_by_segment[index],
            "valid_pose_records": pose_valid_by_segment[index],
            "missing_pose_records": pose_missing_by_segment[index],
            "shuttle_points": shuttle_by_segment[index],
        }
        for index, segment in enumerate(segments)
    ]
    return {
        "schema_version": "syq-input-audit-v1",
        "video": video_info,
        "stages": {
            "fps": stages.match_segmentation.fps,
            "segments": len(segments),
            "events": len(events),
            "strokes": len(strokes),
            "score_rallies": len(stages.score_recognition.rallies),
            "max_segment_end_frame": max(item.end_frame for item in segments),
        },
        "alignment": {
            "fps_matches_video": video_info["fps"] == stages.match_segmentation.fps,
            "segments_fit_video": (
                max(item.end_frame for item in segments) < video_info["frames"]
            ),
            "stroke_event_join_errors": join_errors,
        },
        "pose": {
            "total_records": pose_total,
            "valid_records": pose_valid,
            "missing_keypoints_or_bbox": pose_missing,
            "errors": pose_errors,
        },
        "shuttle": {
            "total_records": shuttle_total,
            "errors": shuttle_errors,
        },
        "court": {
            "present": court is not None,
            "confirmed": court.confirmed if court is not None else None,
            "detection_failed": (
                court.detection_failed if court is not None else None
            ),
            "calibrations": len(court.courts) if court is not None else 0,
        },
        "segments": segment_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    audit = audit_inputs(stage_root=args.stage_root, video=args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "video": audit["video"],
                "stages": audit["stages"],
                "alignment": audit["alignment"],
                "pose": audit["pose"],
                "shuttle": audit["shuttle"],
                "court": audit["court"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"audit: {args.output.resolve()}")


if __name__ == "__main__":
    main()

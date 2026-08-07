"""Split full-match TTYvsASY stage outputs into the three selected clips."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path("fixtures/development/TTYvsASY")
CLIPS_ROOT = ROOT / "selected_clips"
GROUPS = ("seg0039-0043", "seg0052-0056", "seg0140-0144")
PLAYER_MAPPINGS = {
    "seg0039-0043": {"top": "b", "bottom": "a"},
    "seg0052-0056": {"top": "a", "bottom": "b"},
    "seg0140-0144": {"top": "a", "bottom": "b"},
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def localize_item(item: dict, mappings: list[dict]) -> dict | None:
    frame = item.get("frame")
    for mapping in mappings:
        if mapping["source_start_frame"] <= frame <= mapping["source_end_frame"]:
            localized = dict(item)
            localized["frame"] = (
                mapping["local_start_frame"]
                + frame
                - mapping["source_start_frame"]
            )
            if "segment_index" in localized:
                localized["segment_index"] = mapping["local_segment_index"]
            return localized
    return None


def split_group(
    group: str,
    *,
    source_scores: dict,
    source_events: dict,
    source_strokes: dict,
    source_pose: dict,
    source_shuttle: dict,
) -> None:
    clip_root = CLIPS_ROOT / group
    output_root = clip_root / "stages"
    mapping_payload = load_json(clip_root / "source_mapping.json")
    mappings = mapping_payload["segments"]
    source_indexes = {item["source_segment_index"] for item in mappings}

    local_segments = load_json(clip_root / "commentary_input" / "segments.json")
    write_json(output_root / "match_segmentation" / "segments.json", local_segments)

    scores = dict(source_scores)
    scores["rallies"] = [
        {
            **rally,
            "segment_index": next(
                item["local_segment_index"]
                for item in mappings
                if item["source_segment_index"] == rally["segment_index"]
            ),
        }
        for rally in source_scores["rallies"]
        if rally["segment_index"] in source_indexes
    ]
    write_json(output_root / "score_recognition" / "scores.json", scores)

    localized_events: list[dict] = []
    event_index_map: dict[int, int] = {}
    for source_event_index, event in enumerate(source_events["events"]):
        localized = localize_item(event, mappings)
        if localized is not None:
            event_index_map[source_event_index] = len(localized_events)
            localized_events.append(localized)
    event_payload = {
        key: value
        for key, value in source_events.items()
        if key not in {"events", "offsets"}
    }
    event_payload["events"] = localized_events
    write_json(output_root / "event_detection" / "events.json", event_payload)

    localized_strokes: list[dict] = []
    for stroke in source_strokes["strokes"]:
        if stroke["event_index"] not in event_index_map:
            continue
        localized = localize_item(stroke, mappings)
        if localized is None:
            raise ValueError(
                f"{group}: selected stroke {stroke['event_index']} is outside mappings"
            )
        localized["event_index"] = event_index_map[stroke["event_index"]]
        localized_strokes.append(localized)
    stroke_payload = dict(source_strokes)
    stroke_payload["strokes"] = localized_strokes
    write_json(
        output_root / "stroke_classification" / "strokes.json", stroke_payload
    )

    player_mapping = PLAYER_MAPPINGS[group]
    write_json(
        clip_root / "player_mapping.json",
        {
            "players": {"a": "TAI T.Y.", "b": "AN S.Y."},
            "court_position_to_player": player_mapping,
            "evidence": "identity_frame.jpg",
        },
    )
    commentary_events: list[dict] = []
    commentary_strokes: list[dict] = []
    for stroke in localized_strokes:
        player = stroke["player"]
        if player not in player_mapping:
            continue
        commentary_event_index = len(commentary_events)
        commentary_events.append(
            {
                "frame": stroke["frame"],
                "player": player_mapping[player],
                "segment_index": stroke["segment_index"],
            }
        )
        commentary_strokes.append(
            {
                "event_index": commentary_event_index,
                "stroke_type": stroke["stroke_type"],
                "confidence": stroke["confidence"],
            }
        )
    write_json(
        clip_root / "commentary_input" / "events.json",
        {"events": commentary_events},
    )
    write_json(
        clip_root / "commentary_input" / "strokes.json",
        {"strokes": commentary_strokes},
    )

    pose_payload = dict(source_pose)
    pose_payload["frames"] = [
        localized
        for item in source_pose["frames"]
        if (localized := localize_item(item, mappings)) is not None
    ]
    write_json(output_root / "pose" / "pose.json", pose_payload)

    shuttle_payload = dict(source_shuttle)
    shuttle_payload["points"] = [
        localized
        for item in source_shuttle["points"]
        if (localized := localize_item(item, mappings)) is not None
    ]
    write_json(output_root / "shuttle_tracking" / "shuttle.json", shuttle_payload)

    court_output = output_root / "court_detection" / "court.json"
    court_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "stages" / "court_detection" / "court.json", court_output)


def main() -> None:
    stages = ROOT / "stages"
    source_payloads = {
        "source_scores": load_json(stages / "score_recognition" / "scores.json"),
        "source_events": load_json(stages / "event_detection" / "events.json"),
        "source_strokes": load_json(
            stages / "stroke_classification" / "strokes.json"
        ),
        "source_pose": load_json(stages / "pose" / "pose.json"),
        "source_shuttle": load_json(stages / "shuttle_tracking" / "shuttle.json"),
    }
    for group in GROUPS:
        split_group(group, **source_payloads)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SYQ_ROOT = REPO_ROOT / "experiments" / "syq"
MODULE = runpy.run_path(str(SYQ_ROOT / "scripts" / "generate_c_holdout_review.py"))
render_index = MODULE["render_index"]


def test_syq_selection_is_frozen_unique_and_covers_end_swaps():
    selection = json.loads(
        (SYQ_ROOT / "holdout_selection.json").read_text(encoding="utf-8")
    )
    segments = selection["segments"]
    indexes = [item["segment_index"] for item in segments]
    mappings = {
        (item["top_player"], item["bottom_player"]) for item in segments
    }

    assert selection["status"] == "abandoned_before_human_review"
    assert selection["excluded_from_metrics"] is True
    assert selection["selected_before_classifier_run"] is True
    assert selection["variant"] == "C_court_prior"
    assert len(indexes) == len(set(indexes)) == 6
    assert mappings == {("a", "b"), ("b", "a")}


def test_syq_review_index_includes_identity_mapping_and_review_link():
    rendered = render_index(
        {
            "rallies": [
                {
                    "segment_index": 12,
                    "hits": 11,
                    "duration_sec": 16.6,
                    "unknown_candidates": 1,
                    "player_mapping": {
                        "top_name": "FAIHAN",
                        "bottom_name": "SHI Y.Q.",
                    },
                    "video_name": "seg0012_C_overlay.mp4",
                }
            ]
        }
    )

    assert "SYQ C 版本跨場次 Hold-out" in rendered
    assert "top=FAIHAN" in rendered
    assert "bottom=SHI Y.Q." in rendered
    assert "seg0012_C_court_prior/frame_review.html" in rendered

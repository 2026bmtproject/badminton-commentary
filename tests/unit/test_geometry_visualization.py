import json
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "ttyvsasy"
    / "scripts"
    / "visualize_rally_geometry.py"
)
SCRIPT = runpy.run_path(str(SCRIPT_PATH))


def _diagnostic(**overrides):
    payload = {
        "event_index": 1280,
        "frame": 156182,
        "stage_player": "top",
        "player": "b",
        "source_frame": 156182,
        "frame_delta": 0,
        "position_source": "ankle_midpoint",
        "image_point": (930.01, 560.82),
        "court_point_m": (2.77, -1.17),
        "projection_confidence": 0.84,
        "within_court_bounds": False,
        "within_player_half": False,
        "accepted_by_package": False,
        "depth_zone": None,
        "position_change_from_previous_same_player_hit": None,
        "status": "outside_court_bounds",
        "limitations": ["projected_point_outside_court_bounds"],
    }
    payload.update(overrides)
    return SCRIPT["GeometryDiagnostic"](**payload)


def test_geometry_visualization_writes_svg_html_and_json(tmp_path):
    report = SCRIPT["GeometryReport"](
        report_version="court-geometry-diagnostic-v1",
        segment_index=144,
        court_width_m=6.1,
        court_length_m=13.4,
        homography_direction="court_to_image",
        calibration_confirmed=False,
        detection_failed=False,
        inverse_homography_available=True,
        player_mapping={"top": "b", "bottom": "a"},
        accepted_count=1,
        event_count=2,
        diagnostics=[
            _diagnostic(),
            _diagnostic(
                event_index=1281,
                stage_player="bottom",
                player="a",
                court_point_m=(3.1, 10.0),
                within_court_bounds=True,
                within_player_half=True,
                accepted_by_package=True,
                depth_zone="mid",
                position_change_from_previous_same_player_hit="unknown",
                status="accepted",
                limitations=[],
            ),
        ],
    )

    SCRIPT["write_geometry_report"](report, tmp_path)

    payload = json.loads(
        (tmp_path / "court_geometry_report.json").read_text(encoding="utf-8")
    )
    svg = (tmp_path / "court_geometry.svg").read_text(encoding="utf-8")
    html = (tmp_path / "court_geometry.html").read_text(encoding="utf-8")
    assert payload["diagnostics"][0]["court_point_m"] == [2.77, -1.17]
    assert payload["calibration_confirmed"] is False
    assert payload["inverse_homography_available"] is True
    assert payload["diagnostics"][0]["status"] == "outside_court_bounds"
    assert "1280" in svg and "1281" in svg
    assert "#dc2626" in svg
    assert "TOP front" in svg and "BOTTOM rear" in svg
    assert "逐 event 診斷" in html
    assert "projected_point_outside_court_bounds" in html

import runpy
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "ttyvsasy"
    / "scripts"
    / "overlay_pose_projection.py"
)
SCRIPT = runpy.run_path(str(SCRIPT_PATH))


def test_render_overlay_svg_contains_frame_pose_and_projection_evidence():
    keypoint_names = {
        name for edge in SCRIPT["SKELETON_EDGES"] for name in edge
    }
    keypoints = {
        name: (900.0 + index, 400.0 + index, 0.9)
        for index, name in enumerate(sorted(keypoint_names))
    }
    event = SimpleNamespace(
        event_index=1280,
        frame=156182,
        player="b",
        stage_player="top",
        pose_window=[
            SimpleNamespace(
                frame=156182,
                frame_delta=0,
                bbox=(880.0, 370.0, 970.0, 590.0),
                keypoints=keypoints,
            )
        ],
    )
    diagnostic = SimpleNamespace(
        image_point=(930.01, 560.82),
        court_point_m=(2.775, -1.171),
        position_source="ankle_midpoint",
        status="outside_court_bounds",
    )

    svg = SCRIPT["render_overlay_svg"](
        background_data_url="data:image/png;base64,AA==",
        width=1920,
        height=1080,
        event=event,
        diagnostic=diagnostic,
        court_corners=[
            (620.0, 580.0),
            (1300.0, 580.0),
            (1500.0, 1020.0),
            (410.0, 1020.0),
        ],
        homography=[
            [111.0, -15.0, 620.0],
            [0.0, 32.8, 580.0],
            [0.0, 0.0, 1.0],
        ],
        fps=30.0,
    )

    assert "data:image/png;base64,AA==" in svg
    assert "event 1280" in svg
    assert "frame 156182" in svg
    assert "court=(2.775, -1.171) m" in svg
    assert "outside_court_bounds" in svg
    assert 'stroke="#22d3ee"' in svg
    assert 'stroke="#4ade80"' in svg
    assert 'fill="#ef4444"' in svg
    assert "court plane (metres)" in svg


def test_court_to_image_rejects_point_at_projective_infinity():
    result = SCRIPT["_court_to_image"](
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]],
        (1.0, 2.0),
    )

    assert result is None

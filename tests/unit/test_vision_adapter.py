import json

import pytest
from pydantic import ValidationError

from badminton_commentary.adapters.vision import (
    ShuttlePoint,
    read_court_detection_stage,
    read_selected_pose_stage,
    read_selected_shuttle_stage,
)


def keypoints(confidence=0.9):
    return [[float(index), float(index + 1), confidence] for index in range(17)]


def test_streaming_readers_keep_only_requested_segment(tmp_path):
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame": 10,
                        "segment_index": 0,
                        "player": "top",
                        "keypoints": keypoints(),
                        "bbox": [0, 0, 10, 20],
                    },
                    {
                        "frame": 110,
                        "segment_index": 1,
                        "player": "bottom",
                        "keypoints": keypoints(1.0055),
                        "bbox": [1, 2, 11, 22],
                    },
                    {
                        "frame": 210,
                        "segment_index": 2,
                        "player": "top",
                        "keypoints": keypoints(),
                        "bbox": [0, 0, 10, 20],
                    },
                ],
                "pose_mode": "balanced",
            }
        ),
        encoding="utf-8",
    )
    shuttle_path = tmp_path / "shuttle.json"
    shuttle_path.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "frame": frame,
                        "segment_index": segment,
                        "method": "inpaint",
                        "x": 10.0,
                        "y": 20.0,
                        "visible": True,
                        "confidence": 0.9,
                    }
                    for frame, segment in ((10, 0), (110, 1), (210, 2))
                ],
                "fps": 30,
            }
        ),
        encoding="utf-8",
    )

    pose = read_selected_pose_stage(pose_path, segment_index=1)
    shuttle = read_selected_shuttle_stage(
        shuttle_path,
        segment_index=1,
        fps=30,
    )

    assert [item.frame for item in pose.frames] == [110]
    assert pose.frames[0].keypoints[0][2] == 1.0055
    assert [item.frame for item in shuttle.points] == [110]


def test_court_reader_accepts_global_calibration(tmp_path):
    path = tmp_path / "court.json"
    path.write_text(
        json.dumps(
            {
                "courts": [
                    {
                        "corners": [[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]],
                        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "segment_index": None,
                    }
                ],
                "detection_failed": False,
                "confirmed": True,
            }
        ),
        encoding="utf-8",
    )

    court = read_court_detection_stage(path)

    assert court.confirmed is True
    assert court.courts[0].segment_index is None


def test_visible_shuttle_point_requires_coordinates():
    with pytest.raises(ValidationError, match="requires x and y"):
        ShuttlePoint.model_validate(
            {
                "frame": 1,
                "segment_index": 0,
                "method": "inpaint",
                "x": None,
                "y": None,
                "visible": True,
                "confidence": 0.8,
            }
        )

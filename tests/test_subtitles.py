import pytest

from badminton_commentary.schemas import (
    EventDrivenCommentaryOutput,
    GeneratedCommentary,
    RallyCommentaryBundle,
    Segment,
    SegmentsInput,
    StrokeCommentaryLine,
)
from badminton_commentary.subtitles import build_subtitle_cues, render_ass


def segments() -> SegmentsInput:
    return SegmentsInput(
        fps=30,
        segments=[
            Segment(
                start_frame=0,
                end_frame=299,
                start_sec=0,
                end_sec=10,
                duration_sec=10,
            )
        ],
    )


def commentary(*, event_time: float = 2.0) -> EventDrivenCommentaryOutput:
    return EventDrivenCommentaryOutput(
        rallies=[
            RallyCommentaryBundle(
                segment_index=0,
                events=[
                    StrokeCommentaryLine(
                        segment_index=0,
                        stroke_index=2,
                        frame=120,
                        time_sec=4,
                        text="第二條賽評。",
                        source_fact_ids=["rally:0:stroke:2"],
                    ),
                    StrokeCommentaryLine(
                        segment_index=0,
                        stroke_index=1,
                        frame=60,
                        time_sec=event_time,
                        text="第一條賽評。",
                        source_fact_ids=["rally:0:stroke:1"],
                    ),
                ],
                summary=GeneratedCommentary(
                    segment_index=0,
                    text="這是回合總結。",
                    source_fact_ids=["rally:0:score"],
                ),
            )
        ]
    )


def test_builds_chronological_event_and_summary_cues():
    cues = build_subtitle_cues(commentary(), segments())

    assert [(cue.kind, cue.start_sec, cue.end_sec) for cue in cues] == [
        ("event", 2.0, 3.95),
        ("event", 4.0, 7.0),
        ("summary", 5.5, 10.0),
    ]


def test_rejects_event_time_outside_its_segment():
    with pytest.raises(ValueError, match="outside segment"):
        build_subtitle_cues(commentary(event_time=11), segments())


def test_ass_has_separate_styles_and_escapes_override_braces():
    ass = render_ass(build_subtitle_cues(commentary(), segments()))

    assert "Style: Event,Microsoft JhengHei" in ass
    assert "Style: Summary,Microsoft JhengHei" in ass
    assert ",0,8,100,100,170,1" in ass
    assert "Dialogue: 0,0:00:02.00,0:00:03.95,Event" in ass
    assert "Dialogue: 0,0:00:05.50,0:00:10.00,Summary" in ass


def test_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="event_duration"):
        build_subtitle_cues(commentary(), segments(), event_duration=0)

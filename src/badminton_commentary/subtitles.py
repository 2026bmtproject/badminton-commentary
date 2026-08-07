from __future__ import annotations

import re
import textwrap

from badminton_commentary.schemas import (
    EventDrivenCommentaryOutput,
    SegmentsInput,
    SubtitleCue,
)


def build_subtitle_cues(
    commentary: EventDrivenCommentaryOutput,
    segments: SegmentsInput,
    *,
    event_duration: float = 3.0,
    summary_duration: float = 4.5,
) -> list[SubtitleCue]:
    """Build validated ASS cues on the concatenated clip timeline."""
    if event_duration <= 0:
        raise ValueError("event_duration must be greater than zero")
    if summary_duration <= 0:
        raise ValueError("summary_duration must be greater than zero")

    cues: list[SubtitleCue] = []
    seen_segments: set[int] = set()
    for rally in sorted(commentary.rallies, key=lambda item: item.segment_index):
        segment_index = rally.segment_index
        if segment_index in seen_segments:
            raise ValueError(f"duplicate commentary segment_index {segment_index}")
        seen_segments.add(segment_index)
        if segment_index >= len(segments.segments):
            raise ValueError(
                f"commentary segment_index {segment_index} has no matching segment"
            )
        segment = segments.segments[segment_index]
        events = sorted(rally.events, key=lambda item: (item.time_sec, item.stroke_index))

        for position, event in enumerate(events):
            if event.segment_index != segment_index:
                raise ValueError(
                    f"stroke {event.stroke_index} segment_index does not match its rally"
                )
            if not segment.start_sec <= event.time_sec <= segment.end_sec:
                raise ValueError(
                    f"stroke {event.stroke_index} time_sec is outside segment {segment_index}"
                )
            next_time = (
                events[position + 1].time_sec
                if position + 1 < len(events)
                else segment.end_sec
            )
            natural_end = min(event.time_sec + event_duration, segment.end_sec)
            if next_time > event.time_sec:
                natural_end = min(natural_end, max(event.time_sec + 0.1, next_time - 0.05))
            if natural_end > event.time_sec:
                cues.append(
                    SubtitleCue(
                        segment_index=segment_index,
                        kind="event",
                        start_sec=event.time_sec,
                        end_sec=natural_end,
                        text=event.text,
                    )
                )

        if rally.summary is not None:
            if rally.summary.segment_index != segment_index:
                raise ValueError("summary segment_index does not match its rally")
            summary_start = max(
                segment.start_sec,
                segment.end_sec - summary_duration,
                events[-1].time_sec + 0.15 if events else segment.start_sec,
            )
            if segment.end_sec - summary_start >= 0.1:
                cues.append(
                    SubtitleCue(
                        segment_index=segment_index,
                        kind="summary",
                        start_sec=summary_start,
                        end_sec=segment.end_sec,
                        text=rally.summary.text,
                    )
                )

    return sorted(cues, key=lambda cue: (cue.start_sec, cue.kind, cue.segment_index))


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _ass_text(text: str, *, width: int) -> str:
    safe = text.replace("{", "｛").replace("}", "｝")
    safe = re.sub(r"\s+", " ", safe).strip()
    return r"\N".join(
        textwrap.wrap(
            safe,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
    )


def render_ass(
    cues: list[SubtitleCue],
    *,
    font_name: str = "Microsoft JhengHei",
    event_font_size: int = 54,
    summary_font_size: int = 42,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
) -> str:
    if event_font_size <= 0 or summary_font_size <= 0:
        raise ValueError("subtitle font sizes must be greater than zero")
    if play_res_x <= 0 or play_res_y <= 0:
        raise ValueError("ASS resolution must be greater than zero")

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Event,{font_name},{event_font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H78000000,-1,0,0,0,100,100,0,0,3,3,0,2,80,80,70,1
Style: Summary,{font_name},{summary_font_size},&H0000FFFF,&H0000FFFF,&H00101010,&H78000000,-1,0,0,0,100,100,0,0,3,3,0,8,100,100,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogue = []
    for cue in cues:
        style = "Event" if cue.kind == "event" else "Summary"
        width = 30 if cue.kind == "event" else 38
        dialogue.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(cue.start_sec)},{_ass_timestamp(cue.end_sec)},"
            f"{style},,0,0,0,,{_ass_text(cue.text, width=width)}"
        )
    return header + "\n".join(dialogue) + ("\n" if dialogue else "")

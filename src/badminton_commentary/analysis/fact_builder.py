from badminton_commentary.schemas import (
    EventsInput,
    HighlightsInput,
    RallyFact,
    RallyFactEvent,
    RallyScore,
    ScoresInput,
    SegmentsInput,
    Stroke,
    StrokesInput,
)


def _index_by_segment(items, *, item_name: str):
    indexed = {}
    for item in items:
        if item.segment_index in indexed:
            raise ValueError(
                f"duplicate {item_name} for segment_index {item.segment_index}"
            )
        indexed[item.segment_index] = item
    return indexed


def _index_strokes(strokes: StrokesInput, event_count: int) -> dict[int, Stroke]:
    indexed: dict[int, Stroke] = {}
    for stroke in strokes.strokes:
        if stroke.event_index >= event_count:
            raise ValueError(
                f"stroke event_index {stroke.event_index} is out of range for "
                f"{event_count} events"
            )
        if stroke.event_index in indexed:
            raise ValueError(f"duplicate stroke event_index {stroke.event_index}")
        indexed[stroke.event_index] = stroke
    return indexed


def build_rally_facts(
    *,
    segments: SegmentsInput,
    scores: ScoresInput,
    events: EventsInput,
    strokes: StrokesInput,
    highlights: HighlightsInput,
) -> list[RallyFact]:
    """Build deterministic rally facts without mutating the input models."""
    segment_count = len(segments.segments)
    score_by_segment = _index_by_segment(scores.rallies, item_name="score")
    highlight_by_segment = _index_by_segment(
        highlights.highlights, item_name="highlight"
    )
    stroke_by_event = _index_strokes(strokes, len(events.events))

    fact_events: dict[int, list[RallyFactEvent]] = {
        segment_index: [] for segment_index in range(segment_count)
    }
    for event_index, event in enumerate(events.events):
        if event.segment_index >= segment_count:
            raise ValueError(
                f"event {event_index} references unknown segment_index "
                f"{event.segment_index}"
            )
        segment = segments.segments[event.segment_index]
        if not segment.start_frame <= event.frame <= segment.end_frame:
            raise ValueError(
                f"event {event_index} frame {event.frame} is outside segment "
                f"{event.segment_index} frame range "
                f"{segment.start_frame}..{segment.end_frame}"
            )

        stroke = stroke_by_event.get(event_index)
        fact_events[event.segment_index].append(
            RallyFactEvent(
                event_index=event_index,
                frame=event.frame,
                time_sec=event.frame / segments.fps,
                player=event.player,
                stroke_type=stroke.stroke_type if stroke else None,
                stroke_confidence=stroke.confidence if stroke else None,
            )
        )

    facts = []
    for segment_index, segment in enumerate(segments.segments):
        score = score_by_segment.get(segment_index)
        highlight = highlight_by_segment.get(segment_index)
        segment_events = sorted(
            fact_events[segment_index], key=lambda item: (item.frame, item.event_index)
        )
        facts.append(
            RallyFact(
                segment_index=segment_index,
                game_index=score.game_index if score else None,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                duration_sec=segment.duration_sec,
                score=RallyScore(
                    a=score.score_a if score else None,
                    b=score.score_b if score else None,
                ),
                server=score.server if score else None,
                events=segment_events,
                rally_length=len(segment_events),
                highlight_score=highlight.score if highlight else None,
            )
        )
    return facts

import json

from badminton_commentary.generation.batch import generate_commentaries
from badminton_commentary.providers.fake import FakeProvider
from badminton_commentary.schemas import (
    GeneratedCommentary,
    ImportanceResult,
    RallyFact,
    RallyScore,
    ScoredRallyFact,
)


def test_batch_generates_stable_fake_commentary():
    scored = ScoredRallyFact(
        fact=RallyFact(
            segment_index=0,
            game_index=None,
            start_sec=0,
            end_sec=1,
            duration_sec=1,
            score=RallyScore(a=0, b=0),
            server=None,
            events=[],
            rally_length=0,
            highlight_score=None,
        ),
        importance=ImportanceResult(score=0.25, reasons=["close_score"]),
    )
    response = GeneratedCommentary(
        segment_index=0,
        text="戴資穎與安洗瑩目前戰成0比0。",
        source_fact_ids=["rally:0:score"],
    )

    output = generate_commentaries(
        scored_rallies=[scored],
        provider_factory=lambda _: FakeProvider(
            response=json.dumps(response.model_dump(), ensure_ascii=False)
        ),
        player_names={"a": "戴資穎", "b": "安洗瑩"},
    )

    assert len(output.lines) == 1
    assert output.lines[0].segment_index == 0
    assert output.lines[0].source_fact_ids == ["rally:0:score"]
    assert output.lines[0].text == "戴資穎與安洗瑩目前戰成0比0。"

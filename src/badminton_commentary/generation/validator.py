from __future__ import annotations

import re

from badminton_commentary.schemas import (
    CommentaryPlan,
    GeneratedCommentary,
    RallyAnalysis,
    ScoredRallyFact,
)


class CommentaryValidationError(ValueError):
    """Raised when generated text is not supported by its structured facts."""


def normalize_exclamation_emphasis(text: str, *, allow_exclamation: bool) -> str:
    """Deterministically enforce the emphasis policy without changing claims."""
    normalized = []
    used_exclamation = False
    for character in text:
        if character not in "！!":
            normalized.append(character)
            continue
        if allow_exclamation and not used_exclamation:
            normalized.append("！")
            used_exclamation = True
        else:
            normalized.append("。")
    return re.sub(r"([。！？?])(?:[。！？!?])+", r"\1", "".join(normalized))


def commentary_allows_exclamation(scored: ScoredRallyFact) -> bool:
    return scored.importance.score >= 0.70 or (
        scored.fact.highlight_score is not None
        and scored.fact.highlight_score >= 0.75
    )


def _sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[。！？!?]+", text) if part.strip()]
    return max(len(parts), 1)


def validate_language_safety(*, text: str, allow_exclamation: bool) -> None:
    exclamation_count = text.count("!") + text.count("！")
    if exclamation_count > 1 or (exclamation_count and not allow_exclamation):
        raise CommentaryValidationError(
            "generated commentary uses unsupported exclamation emphasis"
        )

    internal_terms = ("觀測球路", "短窗口", "後場類型", "網前類型", "類型")
    if any(term in text for term in internal_terms):
        raise CommentaryValidationError(
            "generated commentary exposes internal schema wording"
        )

    unsupported_inferences = (
        "移動到網前",
        "跑到網前",
        "從後場移動",
        "前移到網前",
        "逼迫",
        "迫使",
        "掌握主動",
        "取得主動",
        "戰術奏效",
        "導致",
        "造成",
        "因而",
        "靠著",
        "抓到機會",
        "被迫",
    )
    if any(term in text for term in unsupported_inferences):
        raise CommentaryValidationError(
            "generated commentary makes an unsupported movement or causal inference"
        )

    unsupported_outcome_claims = (
        "致勝",
        "拿下這一分",
        "拿下分數",
        "得分",
        "最後一拍",
        "最後以",
    )
    if any(claim in text for claim in unsupported_outcome_claims):
        raise CommentaryValidationError(
            "generated commentary makes an unsupported rally outcome claim"
        )


def validate_commentary(
    *,
    generated: GeneratedCommentary,
    scored: ScoredRallyFact,
    plan: CommentaryPlan,
    analysis: RallyAnalysis,
) -> None:
    if generated.segment_index != plan.segment_index:
        raise CommentaryValidationError("generated segment_index does not match plan")
    disallowed = set(generated.source_fact_ids) - set(plan.allowed_fact_ids)
    if disallowed:
        raise CommentaryValidationError(
            f"generated commentary cites disallowed fact ids: {sorted(disallowed)}"
        )
    if _sentence_count(generated.text) > plan.max_sentences:
        raise CommentaryValidationError(
            f"generated commentary exceeds {plan.max_sentences} sentences"
        )

    validate_language_safety(
        text=generated.text,
        allow_exclamation=commentary_allows_exclamation(scored),
    )

    cited_patterns = {
        fact_id
        for fact_id in generated.source_fact_ids
        if ":pattern:" in fact_id
    }
    cited_strokes = {
        fact_id for fact_id in generated.source_fact_ids if ":stroke:" in fact_id
    }
    planned_patterns = {
        fact_id for fact_id in plan.allowed_fact_ids if ":pattern:" in fact_id
    }
    if planned_patterns and not cited_patterns:
        raise CommentaryValidationError(
            "generated commentary omits an available tactical pattern"
        )
    if cited_patterns and len(cited_strokes) > 1:
        raise CommentaryValidationError(
            "generated commentary lists too many strokes beside a pattern"
        )
    if any(fact_id.endswith(":pattern:stroke_diversity") for fact_id in cited_patterns):
        if "節奏" in generated.text or "快慢" in generated.text:
            raise CommentaryValidationError(
                "stroke diversity commentary overclaims temporal variation"
            )

    expected_score = (scored.fact.score.a, scored.fact.score.b)
    written_scores = [
        (int(a), int(b))
        for a, b in re.findall(r"(\d+)\s*比\s*(\d+)", generated.text)
    ]
    if (
        written_scores
        and None not in expected_score
        and any(score != expected_score for score in written_scores)
    ):
        raise CommentaryValidationError(
            f"generated score does not match rally score {expected_score}"
        )

    cautious_ids = {
        stroke.fact_id
        for stroke in analysis.notable_strokes
        if stroke.confidence_band == "cautious"
    }
    if cautious_ids.intersection(generated.source_fact_ids) and not re.search(
        r"可能|似乎|看來|辨識結果|研判", generated.text
    ):
        raise CommentaryValidationError(
            "cautious stroke commentary requires uncertainty wording"
        )

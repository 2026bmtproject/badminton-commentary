"""Run one validated Gemini call for the enriched RallyFact v3 experiment."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import time
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    ValidationError,
    model_validator,
)

from badminton_commentary.adapters import CourtPositionToPlayer
from badminton_commentary.config import load_config
from badminton_commentary.generation.json_response import extract_json_payload
from badminton_commentary.providers import GeminiProvider
from badminton_commentary.providers.base import ProviderError


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SCRIPT = Path(__file__).with_name("package_direct_rallyfact.py")
PACKAGE = runpy.run_path(str(PACKAGE_SCRIPT))
DEFAULT_STAGE_ROOT = PACKAGE["DEFAULT_STAGE_ROOT"]
DEFAULT_OUTPUT_ROOT = PACKAGE["DEFAULT_OUTPUT_ROOT"]


class V3Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


Player = Literal["a", "b"]
Direction = Literal[
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
    "stable",
    "unknown",
]


class V3Score(V3Model):
    a: NonNegativeInt | None
    b: NonNegativeInt | None


class V3PoseObservation(V3Model):
    source_start_frame: NonNegativeInt
    source_end_frame: NonNegativeInt
    confidence: float = Field(ge=0, le=1)
    posture_candidate: Literal[
        "neutral",
        "lunge",
        "deep_lunge",
        "low_reach",
        "stretched_reach",
        "jump",
        "dive_or_fall",
        "recovery",
        "unknown",
    ]
    posture_confidence: float = Field(ge=0, le=1)
    secondary_cues: list[
        Literal[
            "low_body_center",
            "large_step",
            "extended_reach",
            "torso_lean",
            "deep_knee_flexion",
            "airborne_candidate",
            "unstable_support",
            "rising_recovery",
            "unknown",
        ]
    ]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_frame_range(self) -> V3PoseObservation:
        if self.source_end_frame < self.source_start_frame:
            raise ValueError("pose source_end_frame must not precede start")
        return self


class V3CourtObservation(V3Model):
    source_frame: NonNegativeInt
    confidence: float = Field(ge=0, le=1)
    depth_zone: Literal["rear", "mid", "front", "unknown"]
    position_change_from_previous_same_player_hit: Literal[
        "forward",
        "backward",
        "stable",
        "unknown",
    ]
    limitations: list[str]


class V3ShuttleObservation(V3Model):
    start_frame: NonNegativeInt
    end_frame: NonNegativeInt
    confidence: float = Field(ge=0, le=1)
    incoming_image_direction: Direction
    outgoing_image_direction: Direction
    trajectory_change_candidate: Literal[
        "none",
        "direction_reversal",
        "horizontal_redirection",
        "vertical_redirection",
        "sharp_redirection",
        "unknown",
    ]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_frame_range(self) -> V3ShuttleObservation:
        if self.end_frame < self.start_frame:
            raise ValueError("shuttle end_frame must not precede start_frame")
        return self


class V3Event(V3Model):
    event_index: NonNegativeInt
    frame: NonNegativeInt
    time_sec: NonNegativeFloat
    player: Player | None
    stroke_type: str | None
    stroke_confidence: float | None = Field(default=None, ge=0, le=1)
    pose_observation: V3PoseObservation | None
    court_observation: V3CourtObservation | None
    shuttle_observation: V3ShuttleObservation | None
    warnings: list[str]


class V3Evidence(V3Model):
    stage: Literal[
        "event_detection",
        "stroke_classification",
        "pose",
        "court_detection",
        "shuttle_tracking",
    ]
    event_indexes: list[NonNegativeInt]
    frames: list[NonNegativeInt]


class V3TacticalCandidate(V3Model):
    pattern_type: Literal[
        "sustained_attack",
        "front_court_exchange",
        "rear_court_exchange",
        "rear_to_front_stroke_transition",
        "front_to_rear_stroke_transition",
        "attack_transition",
        "defense_to_counterattack_candidate",
        "front_back_court_displacement",
        "attacking_initiative_candidate",
        "notable_stroke_sequence",
        "defensive_recovery_sequence",
        "notable_posture_sequence",
        "repeated_posture_pattern",
        "rally_tactical_theme",
    ]
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    salience: float = Field(ge=0, le=1)
    start_event_index: NonNegativeInt
    end_event_index: NonNegativeInt
    players: list[Player]
    evidence: list[V3Evidence]
    limitations: list[str]


class EnrichedRallyFactV3(V3Model):
    schema_version: Literal["experimental-enriched-rally-fact-v3"]
    segment_index: NonNegativeInt
    game_index: NonNegativeInt | None
    start_sec: NonNegativeFloat
    end_sec: NonNegativeFloat
    duration_sec: NonNegativeFloat
    score: V3Score
    server: Player | None
    events: list[V3Event]
    rally_length: NonNegativeInt
    highlight_score: None
    tactical_candidates: list[V3TacticalCandidate]
    warnings: list[str]

    @model_validator(mode="after")
    def validate_rally(self) -> EnrichedRallyFactV3:
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must not precede start_sec")
        if self.rally_length != len(self.events):
            raise ValueError("rally_length must equal events length")
        indexes = [event.event_index for event in self.events]
        if len(indexes) != len(set(indexes)):
            raise ValueError("event_index values must be unique")
        return self


class V3ExperimentError(RuntimeError):
    pass


def validate_against_package(
    generated: EnrichedRallyFactV3,
    package_payload: dict,
) -> None:
    rally = package_payload["rally"]
    if generated.segment_index != rally["segment_index"]:
        raise V3ExperimentError("generated segment_index does not match package")
    for name in ("start_sec", "end_sec", "duration_sec"):
        if not math.isclose(getattr(generated, name), rally[name], abs_tol=1e-3):
            raise V3ExperimentError(f"generated {name} does not match package")
    if generated.score.model_dump() != rally["score"]:
        raise V3ExperimentError("generated score does not match package")
    if generated.server != rally["server"]:
        raise V3ExperimentError("generated server does not match package")

    source_events = package_payload["events"]
    source_indexes = [event["event_index"] for event in source_events]
    generated_indexes = [event.event_index for event in generated.events]
    if generated_indexes != source_indexes:
        raise V3ExperimentError("generated events do not preserve package order")
    source_by_index = {event["event_index"]: event for event in source_events}

    allowed_frames: dict[tuple[int, str], set[int]] = {}
    for event in generated.events:
        source = source_by_index[event.event_index]
        for name in ("frame", "player"):
            if getattr(event, name) != source[name]:
                raise V3ExperimentError(
                    f"event {event.event_index} {name} does not match package"
                )
        if not math.isclose(event.time_sec, source["time_sec"], abs_tol=1e-3):
            raise V3ExperimentError(
                f"event {event.event_index} time_sec does not match package"
            )
        stroke = source["stroke"]
        if event.stroke_type != (stroke["stroke_type"] if stroke else None):
            raise V3ExperimentError(
                f"event {event.event_index} stroke_type does not match package"
            )
        expected_confidence = stroke["confidence"] if stroke else None
        if (
            event.stroke_confidence is None
            or expected_confidence is None
        ) and event.stroke_confidence != expected_confidence:
            raise V3ExperimentError(
                f"event {event.event_index} stroke confidence availability changed"
            )
        if event.stroke_confidence is not None and not math.isclose(
            event.stroke_confidence,
            expected_confidence,
            abs_tol=1e-6,
        ):
            raise V3ExperimentError(
                f"event {event.event_index} stroke confidence does not match"
            )

        pose_features = source["pose_features"]
        pose_frames = {
            keyframe["frame"] for keyframe in source.get("pose_keyframes", [])
        }
        if pose_features is not None:
            hit_frame = pose_features["hit_frame"]
            pose_frames.add(hit_frame)
            for feature_name, delta_name in (
                ("step_width", "max_frame_delta"),
                ("step_width", "hit_feature_frame_delta"),
                ("knee_flexion", "min_frame_delta"),
                ("torso_lean", "max_frame_delta"),
                ("wrist_reach", "max_frame_delta"),
            ):
                feature = pose_features.get(feature_name) or {}
                delta = feature.get(delta_name)
                if delta is not None:
                    pose_frames.add(hit_frame + delta)
        allowed_frames[(event.event_index, "pose")] = pose_frames
        if event.pose_observation is not None:
            if pose_features is None or (
                event.pose_observation.source_start_frame
                != pose_features["source_start_frame"]
                or event.pose_observation.source_end_frame
                != pose_features["source_end_frame"]
            ):
                raise V3ExperimentError(
                    f"event {event.event_index} pose range does not match package"
                )

        court = source["court_position"]
        allowed_frames[(event.event_index, "court_detection")] = (
            {court["source_frame"]} if court else set()
        )
        if (event.court_observation is None) != (court is None):
            raise V3ExperimentError(
                f"event {event.event_index} court availability changed"
            )
        if event.court_observation is not None:
            observed = event.court_observation
            expected = {
                "source_frame": court["source_frame"],
                "confidence": court["projection_confidence"],
                "depth_zone": court["depth_zone"],
                "position_change_from_previous_same_player_hit": court[
                    "position_change_from_previous_same_player_hit"
                ],
                "limitations": court["limitations"],
            }
            actual = observed.model_dump()
            if actual.keys() != expected.keys() or any(
                not math.isclose(actual[key], value, abs_tol=1e-6)
                if key == "confidence"
                else actual[key] != value
                for key, value in expected.items()
            ):
                raise V3ExperimentError(
                    f"event {event.event_index} court observation changed"
                )

        shuttle = source["shuttle_window"]
        shuttle_frames = {point["frame"] for point in shuttle["points"]} if shuttle else set()
        allowed_frames[(event.event_index, "shuttle_tracking")] = shuttle_frames
        if event.shuttle_observation is not None:
            if shuttle is None or (
                event.shuttle_observation.start_frame != shuttle["start_frame"]
                or event.shuttle_observation.end_frame != shuttle["end_frame"]
            ):
                raise V3ExperimentError(
                    f"event {event.event_index} shuttle range does not match package"
                )
        allowed_frames[(event.event_index, "event_detection")] = {source["frame"]}
        allowed_frames[(event.event_index, "stroke_classification")] = (
            {stroke["source_frame"]} if stroke else set()
        )

    order = {event_index: index for index, event_index in enumerate(source_indexes)}
    for candidate in generated.tactical_candidates:
        if (
            candidate.start_event_index not in order
            or candidate.end_event_index not in order
            or order[candidate.start_event_index] >= order[candidate.end_event_index]
        ):
            raise V3ExperimentError(
                f"{candidate.pattern_type} must span at least two ordered events"
            )
        span = set(
            source_indexes[
                order[candidate.start_event_index] : order[candidate.end_event_index] + 1
            ]
        )
        if not candidate.evidence:
            raise V3ExperimentError(f"{candidate.pattern_type} has no evidence")
        for evidence in candidate.evidence:
            if not evidence.event_indexes or not set(evidence.event_indexes) <= span:
                raise V3ExperimentError(
                    f"{candidate.pattern_type} evidence is outside its event span"
                )
            permitted = set().union(
                *(
                    allowed_frames.get((event_index, evidence.stage), set())
                    for event_index in evidence.event_indexes
                )
            )
            if not evidence.frames or not set(evidence.frames) <= permitted:
                raise V3ExperimentError(
                    f"{candidate.pattern_type} evidence frames are not traceable"
                )


def run_experiment(
    *,
    stage_root: Path,
    output_dir: Path,
    segment_index: int,
    mapping: CourtPositionToPlayer,
    config_path: Path,
    model: str | None,
    overwrite: bool,
    rebuild_package: bool,
) -> Path:
    validated_path = output_dir / "gemini_enriched_rally_fact_v3.json"
    raw_path = output_dir / "gemini_response_v3_raw.txt"
    if (validated_path.exists() or raw_path.exists()) and not overwrite:
        raise FileExistsError(
            f"v3 output already exists; pass --overwrite: {validated_path}"
        )

    if overwrite:
        validated_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        (output_dir / "gemini_v3_run_metadata.json").unlink(missing_ok=True)
    if rebuild_package:
        PACKAGE["build_package"](
            stage_root=stage_root,
            output_dir=output_dir,
            segment_index=segment_index,
            mapping=mapping,
            overwrite=True,
        )
    elif not (
        (output_dir / "rally_stage_input.json").is_file()
        and (output_dir / "prompt_with_rally_stage_input.txt").is_file()
    ):
        raise FileNotFoundError(
            "--reuse-package requires rally_stage_input.json and combined prompt"
        )
    package_payload = json.loads(
        (output_dir / "rally_stage_input.json").read_text(encoding="utf-8")
    )
    expected_prompt = PACKAGE["_render_prompt"](
        segment_index=segment_index,
        mapping=mapping,
    )
    packaged_prompt = (output_dir / "prompt.txt").read_text(encoding="utf-8")
    if packaged_prompt.rstrip() != expected_prompt.rstrip():
        raise V3ExperimentError(
            "existing package uses a stale prompt; rerun without --reuse-package"
        )
    combined_prompt = (output_dir / "prompt_with_rally_stage_input.txt").read_text(
        encoding="utf-8"
    )
    config = load_config(config_path)
    provider = GeminiProvider.from_config(
        config.provider.gemini,
        model_override=model,
    )
    started = time.perf_counter()
    try:
        response = provider.generate(
            system_prompt=(
                "Follow the supplied Experimental Enriched RallyFact v3 "
                "instructions. Return exactly one JSON object and no Markdown."
            ),
            user_prompt=combined_prompt,
        )
    except ProviderError as exc:
        elapsed = time.perf_counter() - started
        failure_metadata = {
            "schema_version": "experimental-enriched-rally-fact-v3",
            "segment_index": segment_index,
            "requested_model": model or config.provider.gemini.model,
            "elapsed_seconds": round(elapsed, 3),
            "logical_provider_calls": 1,
            "status": "provider_error",
            "error": str(exc),
        }
        (output_dir / "gemini_v3_run_metadata.json").write_text(
            json.dumps(failure_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(response.rstrip() + "\n", encoding="utf-8")
    try:
        generated = EnrichedRallyFactV3.model_validate_json(
            extract_json_payload(response)
        )
    except ValidationError as exc:
        raise V3ExperimentError(
            f"Gemini returned invalid v3 JSON; raw response saved to {raw_path}: {exc}"
        ) from exc
    validate_against_package(generated, package_payload)
    validated_path.write_text(
        json.dumps(generated.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": generated.schema_version,
        "segment_index": generated.segment_index,
        "model": provider.last_model_used,
        "elapsed_seconds": round(elapsed, 3),
        "logical_provider_calls": 1,
        "event_count": len(generated.events),
        "tactical_candidate_count": len(generated.tactical_candidates),
        "validated_output": validated_path.name,
        "raw_response": raw_path.name,
    }
    (output_dir / "gemini_v3_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"model: {provider.last_model_used}")
    print("provider_calls: 1")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"events: {len(generated.events)}")
    print(f"tactical_candidates: {len(generated.tactical_candidates)}")
    print(f"output: {validated_path.resolve()}")
    return validated_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Gemini Experimental Enriched RallyFact v3 call."
    )
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--top-player", choices=("a", "b"), required=True)
    parser.add_argument("--bottom-player", choices=("a", "b"), required=True)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument(
        "--reuse-package",
        action="store_true",
        help="Reuse an existing v3 package instead of reading stages again.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output or (
        DEFAULT_OUTPUT_ROOT / f"seg{args.segment_index:04d}"
    )
    run_experiment(
        stage_root=args.stage_root,
        output_dir=output_dir,
        segment_index=args.segment_index,
        mapping=CourtPositionToPlayer(
            top=args.top_player,
            bottom=args.bottom_player,
        ),
        config_path=args.config,
        model=args.model,
        overwrite=args.overwrite,
        rebuild_package=not args.reuse_package,
    )


if __name__ == "__main__":
    main()

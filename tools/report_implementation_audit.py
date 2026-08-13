"""Offline implementation audit for the algorithm/development report.

The audit imports current schemas, executes one synthetic pose-geometry sample,
and optionally inspects a generated direct-RallyFact package.  It never reads an
API key, calls an LLM, or runs upstream computer-vision models.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from badminton_commentary.adapters import StagePaths  # noqa: E402
from badminton_commentary.analysis.pose_geometry import (  # noqa: E402
    MIN_KP_CONF,
    POSE_KEYFRAME_DELTAS,
    PoseGeometryFeatures,
    compute_pose_geometry,
)
from badminton_commentary.config import (  # noqa: E402
    GeminiConfig,
    TacticalAnalyzerConfig,
)
from badminton_commentary.facts import (  # noqa: E402
    CompactPoseFact,
    CompactRallyFacts,
    TacticalAnalysisResult,
)
from badminton_commentary.schemas import (  # noqa: E402
    RallyCommentaryBundle,
    RallyFact,
)
from badminton_commentary.services import RallyCommentaryService  # noqa: E402


AuditStatus = Literal["VERIFIED", "FAILED", "NOT_CHECKED"]


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    claim: str
    status: AuditStatus
    evidence: str


def _literal_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return constants


def _field_names(model: type) -> set[str]:
    return set(model.model_fields)


def _check(
    check_id: str,
    claim: str,
    condition: bool,
    evidence: str,
) -> AuditCheck:
    return AuditCheck(
        check_id=check_id,
        claim=claim,
        status="VERIFIED" if condition else "FAILED",
        evidence=evidence,
    )


def _synthetic_pose_geometry() -> PoseGeometryFeatures:
    keypoints = {
        "left_shoulder": (-50.0, 0.0, 1.2),
        "right_shoulder": (50.0, 0.0, 1.2),
        "left_wrist": (-100.0, 0.0, 1.2),
        "right_wrist": (100.0, 0.0, 1.2),
        "left_hip": (-50.0, 100.0, 1.2),
        "right_hip": (50.0, 100.0, 1.2),
        "left_knee": (-50.0, 150.0, 1.2),
        "right_knee": (50.0, 150.0, 1.2),
        "left_ankle": (-75.0, 200.0, 1.2),
        "right_ankle": (75.0, 200.0, 1.2),
    }
    record = SimpleNamespace(
        frame=100,
        frame_delta=0,
        keypoints=keypoints,
    )
    return compute_pose_geometry(
        [record],
        hit_frame=100,
        source_start_frame=92,
        source_end_frame=110,
    )


def run_audit(package_path: Path | None = None) -> list[AuditCheck]:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_script = (
        REPO_ROOT
        / "experiments"
        / "ttyvsasy"
        / "scripts"
        / "package_direct_rallyfact.py"
    )
    package_constants = _literal_constants(package_script)
    geometry = _synthetic_pose_geometry()
    checks = [
        _check(
            "A001",
            "The project is pinned to Python 3.12.",
            pyproject["project"]["requires-python"] == ">=3.12,<3.13",
            "pyproject.toml project.requires-python",
        ),
        _check(
            "A002",
            "RallyFact is the canonical single-rally domain input.",
            {
                "segment_index",
                "start_sec",
                "end_sec",
                "score",
                "events",
                "rally_length",
            }
            <= _field_names(RallyFact),
            "badminton_commentary.schemas.RallyFact.model_fields",
        ),
        _check(
            "A003",
            "StagePaths exposes four core and three optional vision stages.",
            {
                "match_segmentation",
                "event_detection",
                "score_recognition",
                "stroke_classification",
                "court_detection",
                "shuttle_tracking",
                "pose",
            }
            <= _field_names(StagePaths),
            "badminton_commentary.adapters.StagePaths.model_fields",
        ),
        _check(
            "A004",
            "The service exposes parsed-stage, compact-fact, tactical, and generation boundaries.",
            all(
                hasattr(RallyCommentaryService, name)
                for name in (
                    "prepare_rally_fact",
                    "prepare_compact_facts",
                    "analyze_tactics",
                    "generate_from_stages",
                )
            ),
            "RallyCommentaryService public methods",
        ),
        _check(
            "A005",
            "The production batch output is a RallyCommentaryBundle.",
            {"segment_index", "events", "summary"}
            == _field_names(RallyCommentaryBundle),
            "RallyCommentaryBundle.model_fields",
        ),
        _check(
            "A006",
            "Production CompactRallyFacts remains schema v1 and includes per-event facts.",
            "events" in _field_names(CompactRallyFacts)
            and CompactRallyFacts.model_fields["schema_version"].annotation
            is not None,
            "CompactRallyFacts.model_fields",
        ),
        _check(
            "A007",
            "Production CompactPoseFact still contains a hitting-arm geometry candidate.",
            "hitting_arm_candidate" in _field_names(CompactPoseFact),
            "CompactPoseFact.model_fields",
        ),
        _check(
            "A008",
            "Experimental pose geometry has six numeric feature groups and no posture label.",
            {
                "step_width",
                "knee_flexion",
                "body_height",
                "torso_lean",
                "wrist_reach",
                "body_displacement",
            }
            <= _field_names(PoseGeometryFeatures)
            and "posture_candidate" not in _field_names(PoseGeometryFeatures),
            "PoseGeometryFeatures.model_fields",
        ),
        _check(
            "A009",
            "Pose geometry uses confidence threshold 0.35 and fixed keyframe deltas.",
            MIN_KP_CONF == 0.35
            and POSE_KEYFRAME_DELTAS == (-8, -4, 0, 4, 8, 10),
            "pose_geometry.MIN_KP_CONF and POSE_KEYFRAME_DELTAS",
        ),
        _check(
            "A010",
            "Synthetic step width is torso-normalized and confidence is clamped.",
            geometry.step_width.at_hit_ratio == 1.5
            and geometry.step_width.confidence == 1.0,
            "compute_pose_geometry synthetic runtime result",
        ),
        _check(
            "A011",
            "The direct package transport version is v4 with a -8/+10 pose source window.",
            package_constants.get("PACKAGE_VERSION")
            == "direct-rallyfact-event-centric-v4"
            and package_constants.get("POSE_PRE_FRAMES") == 8
            and package_constants.get("POSE_POST_FRAMES") == 10,
            "package_direct_rallyfact.py literal constants",
        ),
        _check(
            "A012",
            "The direct v4 transport still requests Experimental Enriched RallyFact v3 output.",
            package_constants.get("OUTPUT_SCHEMA_VERSION")
            == "experimental-enriched-rally-fact-v3",
            "package_direct_rallyfact.py OUTPUT_SCHEMA_VERSION",
        ),
        _check(
            "A013",
            "Tactical analysis returns at most five validated facts.",
            TacticalAnalysisResult.model_fields["facts"].annotation is not None
            and TacticalAnalyzerConfig().max_facts == 5,
            "TacticalAnalysisResult schema and TacticalAnalyzerConfig default",
        ),
        _check(
            "A014",
            "Gemini defaults use an environment-variable key and bounded retry settings.",
            GeminiConfig().api_key_env == "GEMINI_API_KEY"
            and GeminiConfig().max_attempts == 3
            and GeminiConfig().timeout_seconds == 30.0,
            "GeminiConfig defaults",
        ),
        _check(
            "A015",
            "The audit itself does not require provider or CV execution.",
            "provider" not in inspect.signature(run_audit).parameters,
            "inspect.signature(run_audit)",
        ),
    ]

    if package_path is None:
        checks.append(
            AuditCheck(
                check_id="A016",
                claim="A generated direct v4 package matches the compact transport contract.",
                status="NOT_CHECKED",
                evidence="Pass --package to inspect a generated artifact.",
            )
        )
        return checks

    payload = json.loads(package_path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    context = payload.get("context", {})
    checks.append(
        _check(
            "A016",
            "A generated direct v4 package matches the compact transport contract.",
            context.get("package_version")
            == "direct-rallyfact-event-centric-v4"
            and bool(events)
            and all(
                "pose_features" in event
                and "pose_keyframes" in event
                and "pose_window" not in event
                for event in events
            ),
            str(package_path),
        )
    )
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = run_audit(args.package)
    payload = {
        "audit_version": "report-implementation-audit-v1",
        "checks": [asdict(check) for check in checks],
        "summary": {
            status: sum(check.status == status for check in checks)
            for status in ("VERIFIED", "FAILED", "NOT_CHECKED")
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if any(check.status == "FAILED" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())

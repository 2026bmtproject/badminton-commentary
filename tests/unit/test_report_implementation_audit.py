import json
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = runpy.run_path(
    str(REPO_ROOT / "tools" / "report_implementation_audit.py")
)


def test_report_audit_verifies_runtime_contracts_offline():
    checks = SCRIPT["run_audit"]()

    assert not [check for check in checks if check.status == "FAILED"]
    assert {check.check_id for check in checks} == {
        f"A{index:03d}" for index in range(1, 17)
    }
    assert checks[-1].status == "NOT_CHECKED"


def test_report_audit_can_inspect_compact_v4_artifact(tmp_path):
    package = tmp_path / "rally_stage_input.json"
    package.write_text(
        json.dumps(
            {
                "context": {
                    "package_version": "direct-rallyfact-event-centric-v4"
                },
                "events": [
                    {
                        "pose_features": {},
                        "pose_keyframes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    checks = SCRIPT["run_audit"](package)

    artifact_check = next(check for check in checks if check.check_id == "A016")
    assert artifact_check.status == "VERIFIED"

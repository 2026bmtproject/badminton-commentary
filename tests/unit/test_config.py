from pathlib import Path

import pytest

from badminton_commentary.config import load_config


EXAMPLE_CONFIG = Path(__file__).parents[2] / "config.yaml.example"


def test_example_config_is_valid():
    config = load_config(EXAMPLE_CONFIG)

    assert config.provider.name == "gemini"
    assert config.provider.gemini.model == "gemini-flash-latest"
    assert config.provider.gemini.api_key_env == "GEMINI_API_KEY"
    assert config.provider.gemini.timeout_seconds == 30.0
    assert config.provider.gemini.max_attempts == 3
    assert config.tactical_analyzer.model == "gemini-3.1-pro-preview"
    assert config.tactical_analyzer.fallback_models == ["gemini-3.6-flash"]
    assert config.tactical_analyzer.max_facts == 5


def test_config_rejects_invalid_retry_count(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "provider:\n  gemini:\n    max_attempts: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_attempts"):
        load_config(path)


def test_config_rejects_non_mapping_root(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n- root\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)


def test_config_rejects_invalid_tactical_fact_limit(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "tactical_analyzer:\n  max_facts: 6\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_facts"):
        load_config(path)


def test_config_rejects_duplicate_tactical_models(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "tactical_analyzer:\n"
        "  model: same-model\n"
        "  fallback_models: [same-model]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be unique"):
        load_config(path)

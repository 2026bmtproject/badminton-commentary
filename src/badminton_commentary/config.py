from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .schemas import StrictModel


class GeminiConfig(StrictModel):
    model: str = "gemini-flash-latest"
    api_key_env: str = "GEMINI_API_KEY"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)


class ProviderConfig(StrictModel):
    name: Literal["fake", "gemini"] = "fake"
    fake_response: str = "測試用固定賽評"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)


class AppConfig(StrictModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config file {config_path}: {exc}") from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    return AppConfig.model_validate(payload)

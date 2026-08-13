from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .schemas import StrictModel


class GeminiConfig(StrictModel):
    model: str = "gemini-flash-latest"
    api_key_env: str = "GEMINI_API_KEY"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)


class TacticalAnalyzerConfig(StrictModel):
    model: str = "gemini-3.1-pro-preview"
    fallback_models: list[str] = Field(
        default_factory=lambda: ["gemini-3.6-flash"]
    )
    max_facts: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def validate_models(self) -> "TacticalAnalyzerConfig":
        models = [self.model, *self.fallback_models]
        if any(not model.strip() for model in models):
            raise ValueError("tactical analyzer models must not be empty")
        if len(set(models)) != len(models):
            raise ValueError("tactical analyzer models must be unique")
        return self


class ProviderConfig(StrictModel):
    name: Literal["fake", "gemini"] = "fake"
    fake_response: str = "測試用固定賽評"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)


class AppConfig(StrictModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    tactical_analyzer: TacticalAnalyzerConfig = Field(
        default_factory=TacticalAnalyzerConfig
    )


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

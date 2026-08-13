import os
from typing import Protocol

from google import genai

from badminton_commentary.config import GeminiConfig

from .base import ProviderError


DEFAULT_MODEL = "gemini-flash-latest"
RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]
MODEL_FALLBACK_STATUS_CODES = {404, 500, 502, 503, 504}


class _GeminiModels(Protocol):
    def generate_content(self, *, model: str, contents: str, config: dict): ...


class _GeminiClient(Protocol):
    models: _GeminiModels


def _error_status_code(exc: Exception) -> int | None:
    """Read HTTP status from google-genai and common transport exceptions."""
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        nested = details.get("error", details)
        if isinstance(nested, dict):
            candidate = nested.get("code")
            if isinstance(candidate, int):
                return candidate
    return None


class GeminiProvider:
    @classmethod
    def from_config(
        cls,
        config: GeminiConfig,
        *,
        model_override: str | None = None,
        fallback_models: list[str] | None = None,
        client: _GeminiClient | None = None,
    ) -> "GeminiProvider":
        api_key = os.getenv(config.api_key_env) if client is None else None
        if client is None and not api_key:
            raise ValueError(
                f"Gemini API key is required; set {config.api_key_env}"
            )
        return cls(
            model=model_override or config.model,
            fallback_models=fallback_models,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            client=client,
        )

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        fallback_models: list[str] | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        client: _GeminiClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        resolved_fallbacks = list(fallback_models or [])
        if any(not fallback.strip() for fallback in resolved_fallbacks):
            raise ValueError("fallback models must not be empty")
        if model in resolved_fallbacks or len(set(resolved_fallbacks)) != len(
            resolved_fallbacks
        ):
            raise ValueError("model and fallback models must be unique")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")

        self.model = model
        self.fallback_models = resolved_fallbacks
        self.last_model_used: str | None = None
        if client is not None:
            self._client = client
            return

        resolved_api_key = (
            api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "Gemini API key is required; set GOOGLE_API_KEY or GEMINI_API_KEY"
            )

        self._client = genai.Client(
            api_key=resolved_api_key,
            http_options={
                "timeout": int(timeout_seconds * 1000),
                "retry_options": {
                    "attempts": max_attempts,
                    "initial_delay": 1.0,
                    "exp_base": 2.0,
                    "max_delay": 8.0,
                    "jitter": 0.1,
                    "http_status_codes": RETRYABLE_STATUS_CODES,
                },
            },
        )

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.last_model_used = None
        models = [self.model, *self.fallback_models]
        failures: list[tuple[str, Exception]] = []
        for index, model in enumerate(models):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config={"system_instruction": system_prompt},
                )
            except Exception as exc:
                failures.append((model, exc))
                status_code = _error_status_code(exc)
                can_fallback = (
                    index < len(models) - 1
                    and status_code in MODEL_FALLBACK_STATUS_CODES
                )
                if can_fallback:
                    continue
                attempted = ", ".join(name for name, _ in failures)
                raise ProviderError(
                    f"Gemini request failed: {exc} (attempted models: {attempted})"
                ) from exc

            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise ProviderError(
                    f"Gemini returned an empty text response from model {model}"
                )
            self.last_model_used = model
            return text

        raise ProviderError("Gemini request failed without an attempted model")

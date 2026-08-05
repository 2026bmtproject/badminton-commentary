import os
from typing import Protocol

from google import genai

from badminton_commentary.config import GeminiConfig

from .base import ProviderError


DEFAULT_MODEL = "gemini-flash-latest"
RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]


class _GeminiModels(Protocol):
    def generate_content(self, *, model: str, contents: str, config: dict): ...


class _GeminiClient(Protocol):
    models: _GeminiModels


class GeminiProvider:
    @classmethod
    def from_config(
        cls,
        config: GeminiConfig,
        *,
        client: _GeminiClient | None = None,
    ) -> "GeminiProvider":
        api_key = os.getenv(config.api_key_env) if client is None else None
        if client is None and not api_key:
            raise ValueError(
                f"Gemini API key is required; set {config.api_key_env}"
            )
        return cls(
            model=config.model,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            client=client,
        )

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        client: _GeminiClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")

        self.model = model
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
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config={"system_instruction": system_prompt},
            )
        except Exception as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Gemini returned an empty text response")
        return text

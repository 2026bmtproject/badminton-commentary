from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a usable response."""


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...

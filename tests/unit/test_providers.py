from types import SimpleNamespace

import pytest

from badminton_commentary.config import GeminiConfig
from badminton_commentary.providers.base import LLMProvider, ProviderError
from badminton_commentary.providers.fake import FakeProvider, PromptCall
from badminton_commentary.providers.gemini import GeminiProvider
from badminton_commentary.providers.timed import TimedProvider, TimingStats


class StubModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class StubClient:
    def __init__(self, models):
        self.models = models


def test_fake_provider_returns_configured_response_and_records_prompts():
    provider = FakeProvider(response='{"ok": true}')

    result = provider.generate(system_prompt="system", user_prompt="user")

    assert result == '{"ok": true}'
    assert provider.calls == [PromptCall("system", "user")]


def test_fake_provider_satisfies_protocol():
    provider = FakeProvider(response="stable")

    assert isinstance(provider, LLMProvider)
    assert provider.generate(system_prompt="s", user_prompt="u") == "stable"


def test_timed_provider_records_successful_call_duration():
    stats = TimingStats()
    clock_values = iter((10.0, 12.5))
    provider = TimedProvider(
        FakeProvider(response="ok"),
        label="group/rally:0/fake",
        stats=stats,
        clock=lambda: next(clock_values),
    )

    assert provider.generate(system_prompt="system", user_prompt="user") == "ok"
    assert stats.call_count == 1
    assert stats.total_seconds == 2.5
    assert stats.records[0].succeeded is True


def test_timed_provider_records_failed_call_duration():
    class FailingProvider:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            raise ProviderError("failed")

    stats = TimingStats()
    clock_values = iter((3.0, 4.25))
    provider = TimedProvider(
        FailingProvider(),
        label="group/rally:1/gemini",
        stats=stats,
        clock=lambda: next(clock_values),
    )

    with pytest.raises(ProviderError, match="failed"):
        provider.generate(system_prompt="system", user_prompt="user")

    assert stats.total_seconds == 1.25
    assert stats.records[0].succeeded is False


def test_gemini_provider_calls_injected_client():
    models = StubModels(response=SimpleNamespace(text="賽評內容"))
    provider = GeminiProvider(model="test-model", client=StubClient(models))

    result = provider.generate(system_prompt="系統提示", user_prompt="使用者提示")

    assert result == "賽評內容"
    assert models.calls == [
        {
            "model": "test-model",
            "contents": "使用者提示",
            "config": {"system_instruction": "系統提示"},
        }
    ]


def test_gemini_provider_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key is required"):
        GeminiProvider()


def test_gemini_provider_configures_timeout_and_retry(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return StubClient(StubModels())

    monkeypatch.setattr(
        "badminton_commentary.providers.gemini.genai.Client", fake_client
    )
    GeminiProvider(api_key="secret", timeout_seconds=12.5, max_attempts=4)

    assert captured["api_key"] == "secret"
    assert captured["http_options"]["timeout"] == 12_500
    assert captured["http_options"]["retry_options"]["attempts"] == 4
    assert 429 in captured["http_options"]["retry_options"]["http_status_codes"]


def test_gemini_provider_can_be_created_from_config(monkeypatch):
    models = StubModels(response=SimpleNamespace(text="ok"))
    config = GeminiConfig(
        model="configured-model",
        api_key_env="CUSTOM_GEMINI_KEY",
        timeout_seconds=10,
        max_attempts=2,
    )
    monkeypatch.setenv("CUSTOM_GEMINI_KEY", "secret")

    provider = GeminiProvider.from_config(config, client=StubClient(models))
    provider.generate(system_prompt="system", user_prompt="user")

    assert models.calls[0]["model"] == "configured-model"


def test_gemini_provider_wraps_sdk_errors():
    client = StubClient(StubModels(error=RuntimeError("network down")))
    provider = GeminiProvider(client=client)

    with pytest.raises(ProviderError, match="Gemini request failed: network down"):
        provider.generate(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize("text", [None, "", "   "])
def test_gemini_provider_rejects_empty_response(text):
    provider = GeminiProvider(
        client=StubClient(StubModels(response=SimpleNamespace(text=text)))
    )

    with pytest.raises(ProviderError, match="empty text response"):
        provider.generate(system_prompt="system", user_prompt="user")

import pytest

from ml_auto_tune.config import AdvisorSettings
from ml_auto_tune.llm import resolve_llm_settings


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


def test_local_llm_settings_discover_model_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def fake_get(url, headers, timeout):
        requests.append((url, headers, timeout))
        return FakeResponse({"data": [{"id": "local-test-model"}]})

    monkeypatch.delenv("ML_AUTO_TUNE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ML_AUTO_TUNE_LLM_MODEL", raising=False)
    monkeypatch.setattr("ml_auto_tune.llm.httpx.get", fake_get)

    settings = resolve_llm_settings(
        AdvisorSettings(provider="openai_compatible", base_url="http://127.0.0.1:8000/v1"),
        "test advisor",
    )

    assert settings.api_key is None
    assert settings.base_url == "http://127.0.0.1:8000/v1"
    assert settings.model == "local-test-model"
    assert settings.headers == {"Content-Type": "application/json"}
    assert requests == [("http://127.0.0.1:8000/v1/models", {"Content-Type": "application/json"}, 10)]


def test_remote_llm_settings_still_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ML_AUTO_TUNE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ML_AUTO_TUNE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ML_AUTO_TUNE_LLM_MODEL", raising=False)

    with pytest.raises(ValueError, match="ML_AUTO_TUNE_LLM_API_KEY"):
        resolve_llm_settings(AdvisorSettings(provider="openai_compatible"), "test advisor")


def test_explicit_model_skips_local_model_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get(*args, **kwargs):
        raise AssertionError("model discovery should not be called")

    monkeypatch.delenv("ML_AUTO_TUNE_LLM_API_KEY", raising=False)
    monkeypatch.setattr("ml_auto_tune.llm.httpx.get", fail_get)

    settings = resolve_llm_settings(
        AdvisorSettings(
            provider="openai_compatible",
            base_url="http://localhost:8000/v1",
            model="configured-model",
        ),
        "test advisor",
    )

    assert settings.model == "configured-model"

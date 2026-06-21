from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from ml_auto_tune.config import AdvisorSettings

DEFAULT_BASE_URL = "https://api.openai.com/v1"
LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"


@dataclass(frozen=True)
class LLMClientSettings:
    api_key: str | None
    base_url: str
    model: str

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def resolve_llm_settings(settings: AdvisorSettings, usage: str) -> LLMClientSettings:
    api_key = settings.api_key or os.getenv("ML_AUTO_TUNE_LLM_API_KEY")
    base_url = settings.base_url or os.getenv("ML_AUTO_TUNE_LLM_BASE_URL") or DEFAULT_BASE_URL
    model = settings.model or os.getenv("ML_AUTO_TUNE_LLM_MODEL")

    if not api_key and not _is_local_base_url(base_url):
        raise ValueError(f"ML_AUTO_TUNE_LLM_API_KEY is required for {usage}.")

    if not model:
        if _is_local_base_url(base_url):
            model = discover_local_model(base_url, api_key)
        else:
            raise ValueError(f"ML_AUTO_TUNE_LLM_MODEL is required for {usage}.")

    return LLMClientSettings(api_key=api_key, base_url=base_url, model=model)


def discover_local_model(base_url: str = LOCAL_BASE_URL, api_key: str | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=10)
    response.raise_for_status()
    payload: Any = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models:
        raise ValueError(f"No models were returned by {base_url.rstrip('/')}/models.")
    first = models[0]
    if not isinstance(first, dict) or not isinstance(first.get("id"), str) or not first["id"]:
        raise ValueError(f"The first model returned by {base_url.rstrip('/')}/models lacks an id.")
    return first["id"]


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ml_auto_tune.config import AdvisorSettings


@dataclass(frozen=True)
class AdvisorContext:
    study_name: str
    task: str
    metric: str
    direction: str
    best_score: float | None
    best_params: dict[str, Any]
    trials_since_improvement: int
    available_models: list[str]
    active_models: list[str]
    recent_trials: list[dict[str, Any]]
    validation_metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class AdvisorResponse:
    markdown: str
    structured_suggestions: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None


class Advisor(Protocol):
    def advise(self, context: AdvisorContext) -> AdvisorResponse:
        """Return tuning advice for the current run state."""


class MockAdvisor:
    def advise(self, context: AdvisorContext) -> AdvisorResponse:
        models = [
            model
            for model in [
                "linear_regression",
                "ridge",
                "elastic_net",
                "hist_gradient_boosting",
                "random_forest",
                "extra_trees",
                "logistic_regression",
                "hist_gradient_boosting_classifier",
                "random_forest_classifier",
                "extra_trees_classifier",
            ]
            if model in context.available_models
        ]
        if context.validation_metrics:
            metrics = context.validation_metrics
            if context.task == "classification":
                markdown = (
                    "Mock advisor: evaluated the current validation metrics. "
                    f"Accuracy={metrics['accuracy']:.4f}, F1 macro={metrics['f1_macro']:.4f}, "
                    f"ROC-AUC={metrics['roc_auc']:.4f}. Confirm class balance and compare a linear "
                    "classifier against tree-based classifiers before declaring the model best."
                )
            else:
                markdown = (
                    "Mock advisor: evaluated the current validation metrics. "
                    f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}. "
                    "Linear regression is a useful baseline; treat it as best only after comparing it "
                    "against regularized linear models and nonlinear tree/boosting models on the same split."
                )
        else:
            markdown = (
                "Mock advisor: tuning has plateaued. Compare simple baselines against stronger model "
                "families before declaring the model best."
            )
        return AdvisorResponse(
            markdown=markdown,
            structured_suggestions={"model_candidates": models or context.active_models},
            raw=markdown,
        )


class OpenAICompatibleAdvisor:
    def __init__(self, settings: AdvisorSettings):
        self.api_key = settings.api_key or os.getenv("ML_AUTO_TUNE_LLM_API_KEY")
        self.base_url = settings.base_url or os.getenv("ML_AUTO_TUNE_LLM_BASE_URL") or "https://api.openai.com/v1"
        self.model = settings.model or os.getenv("ML_AUTO_TUNE_LLM_MODEL")
        if not self.api_key:
            raise ValueError("ML_AUTO_TUNE_LLM_API_KEY is required for the OpenAI-compatible advisor.")
        if not self.model:
            raise ValueError("ML_AUTO_TUNE_LLM_MODEL is required for the OpenAI-compatible advisor.")

    def advise(self, context: AdvisorContext) -> AdvisorResponse:
        prompt = _build_prompt(context)
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You advise an automated sklearn model tuner. "
                            "Use validation metrics to assess whether the current model appears strong. "
                            "Return compact JSON with keys markdown and structured_suggestions. "
                            "Only suggest model_candidates from the provided allowed models."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        parsed = _parse_advisor_json(raw)
        return AdvisorResponse(
            markdown=str(parsed.get("markdown") or raw),
            structured_suggestions=dict(parsed.get("structured_suggestions") or {}),
            raw=raw,
        )


def build_advisor(settings: AdvisorSettings) -> Advisor:
    if settings.provider == "mock":
        return MockAdvisor()
    if settings.provider == "openai_compatible":
        return OpenAICompatibleAdvisor(settings)
    raise ValueError(f"Unsupported advisor provider: {settings.provider}")


def _build_prompt(context: AdvisorContext) -> str:
    return json.dumps(
        {
            "study_name": context.study_name,
            "task": context.task,
            "metric": context.metric,
            "direction": context.direction,
            "best_score": context.best_score,
            "best_params": context.best_params,
            "trials_since_improvement": context.trials_since_improvement,
            "allowed_models": context.available_models,
            "active_models": context.active_models,
            "recent_trials": context.recent_trials,
            "validation_metrics": context.validation_metrics,
            "response_schema": {
                "markdown": "human-readable tuning advice",
                "structured_suggestions": {
                    "model_candidates": "optional list containing only allowed model names"
                },
            },
        },
        indent=2,
        default=str,
    )


def _parse_advisor_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"markdown": raw, "structured_suggestions": {}}
    return parsed if isinstance(parsed, dict) else {"markdown": raw, "structured_suggestions": {}}

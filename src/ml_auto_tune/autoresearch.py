from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from ml_auto_tune.config import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_MODELS,
    REGRESSION_METRICS,
    REGRESSION_MODELS,
    TuningConfig,
    parse_config,
)
from ml_auto_tune.tuning import RunResult, run_tuning

SAFE_PATCH_KEYS = {
    "models": None,
    "data": {"features"},
    "optimization": {"n_trials", "repeated_splits", "plateau_trials", "min_delta"},
    "advisor": {"enabled", "provider", "trigger"},
}


@dataclass(frozen=True)
class ResearchSuggestion:
    description: str
    hypothesis: str
    config_patch: dict[str, Any]
    expected_improvement: str
    risk_notes: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchExperiment:
    experiment_id: int
    status: str
    score: float
    task: str
    metric: str
    model: str
    config_path: Path
    output_directory: Path
    description: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoresearchResult:
    output_directory: Path
    best_experiment: ResearchExperiment
    experiments: list[ResearchExperiment]


class ResearchAdvisor(Protocol):
    def suggest(self, context: dict[str, Any]) -> ResearchSuggestion:
        """Suggest the next safe experiment config patch."""


class DeterministicResearchAdvisor:
    def suggest(self, context: dict[str, Any]) -> ResearchSuggestion:
        task = context["task"]
        experiment_id = int(context["next_experiment_id"])
        candidates = (
            [
                ["linear_regression"],
                ["ridge"],
                ["elastic_net"],
                ["random_forest"],
                ["hist_gradient_boosting"],
                ["random_forest", "hist_gradient_boosting"],
            ]
            if task == "regression"
            else [
                ["logistic_regression"],
                ["random_forest_classifier"],
                ["hist_gradient_boosting_classifier"],
                ["extra_trees_classifier"],
                ["random_forest_classifier", "hist_gradient_boosting_classifier"],
            ]
        )
        models = candidates[(experiment_id - 1) % len(candidates)]
        return ResearchSuggestion(
            description=f"Try model candidates: {', '.join(models)}",
            hypothesis="A different model family may improve the validation metric.",
            config_patch={
                "models": models,
                "optimization": {
                    "n_trials": max(2, min(int(context["base_config"]["optimization"]["n_trials"]), 8)),
                    "repeated_splits": context["base_config"]["optimization"].get("repeated_splits", False),
                },
            },
            expected_improvement="Explore an alternative safe model family.",
            risk_notes="Deterministic fallback suggestion; no external LLM was used.",
        )


class OpenAIResearchAdvisor:
    def __init__(self, config: TuningConfig):
        self.api_key = config.advisor.api_key or os.getenv("ML_AUTO_TUNE_LLM_API_KEY")
        self.base_url = config.advisor.base_url or os.getenv("ML_AUTO_TUNE_LLM_BASE_URL") or "https://api.openai.com/v1"
        self.model = config.advisor.model or os.getenv("ML_AUTO_TUNE_LLM_MODEL")
        if not self.api_key:
            raise ValueError("ML_AUTO_TUNE_LLM_API_KEY is required for LLM autoresearch.")
        if not self.model:
            raise ValueError("ML_AUTO_TUNE_LLM_MODEL is required for LLM autoresearch.")

    def suggest(self, context: dict[str, Any]) -> ResearchSuggestion:
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
                            "You are a config-only ML autoresearch agent. Suggest exactly one next "
                            "experiment as compact JSON with keys description, hypothesis, config_patch, "
                            "expected_improvement, and risk_notes. Only use safe config_patch fields."
                        ),
                    },
                    {"role": "user", "content": json.dumps(context, indent=2, default=str)},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]
        raw = _parse_json_object(raw_text)
        return _coerce_suggestion(raw)


def run_autoresearch(
    config: TuningConfig,
    research_advisor: ResearchAdvisor | None = None,
) -> AutoresearchResult:
    if not config.research.enabled:
        raise ValueError("research.enabled must be true for autoresearch runs.")

    root_dir = config.output.directory
    configs_dir = root_dir / "configs"
    experiments_dir = root_dir / "experiments"
    root_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    program_text = _read_program(config.research.program_path)
    base_raw = _raw_config(config)
    advisor = research_advisor or _build_research_advisor(config)

    experiments: list[ResearchExperiment] = []
    suggestions_file = root_dir / "llm_suggestions.jsonl"
    results_file = root_dir / "research_results.tsv"
    log_file = root_dir / "research_log.md"
    _write_results_header(results_file)
    log_file.write_text("# Autoresearch Log\n\n", encoding="utf-8")

    best: ResearchExperiment | None = None
    best_config_raw: dict[str, Any] | None = None

    for experiment_id in range(config.research.max_experiments):
        if experiment_id == 0:
            description = "baseline"
            experiment_raw = copy.deepcopy(base_raw)
            suggestion_payload: dict[str, Any] | None = None
        else:
            context = _research_context(
                config=config,
                program_text=program_text,
                base_raw=base_raw,
                best=best,
                experiments=experiments,
                next_experiment_id=experiment_id,
            )
            suggestion = _safe_suggest(advisor, context)
            suggestion_payload = {
                "experiment_id": experiment_id,
                "suggestion": suggestion.raw or {
                    "description": suggestion.description,
                    "hypothesis": suggestion.hypothesis,
                    "config_patch": suggestion.config_patch,
                    "expected_improvement": suggestion.expected_improvement,
                    "risk_notes": suggestion.risk_notes,
                },
                "accepted": "rejected_error" not in suggestion.raw,
                "fallback_used": "rejected_error" in suggestion.raw,
            }
            _append_jsonl(suggestions_file, suggestion_payload)
            description = suggestion.description
            experiment_raw = apply_safe_config_patch(base_raw, suggestion.config_patch)

        experiment_raw = _prepare_experiment_raw(config, experiment_raw, experiment_id, experiments_dir)
        config_path = configs_dir / f"experiment_{experiment_id:03d}.yaml"
        _write_yaml(config_path, experiment_raw)

        try:
            experiment_config = parse_config(experiment_raw, base_dir=Path.cwd())
            result = run_tuning(experiment_config)
            status = _decide_status(config, result, best, experiment_id)
            experiment = _experiment_from_result(
                experiment_id=experiment_id,
                status=status,
                result=result,
                config=experiment_config,
                config_path=config_path,
                description=description,
            )
        except Exception as exc:  # noqa: BLE001 - crashes are research outcomes.
            experiment = ResearchExperiment(
                experiment_id=experiment_id,
                status="crash",
                score=0.0,
                task=config.task,
                metric=config.optimization.metric,
                model="unknown",
                config_path=config_path,
                output_directory=Path(experiment_raw["output"]["directory"]),
                description=f"{description}: {exc}",
            )

        experiments.append(experiment)
        _append_result(results_file, experiment)
        _append_log(log_file, experiment, suggestion_payload)

        if experiment.status == "keep":
            best = experiment
            best_config_raw = experiment_raw
            _write_yaml(root_dir / "best_config.yaml", best_config_raw)
            metrics_path = experiment.output_directory / "metrics.json"
            if metrics_path.exists():
                (root_dir / "best_metrics.json").write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")

    if best is None:
        raise RuntimeError("Autoresearch completed without a successful experiment.")
    return AutoresearchResult(output_directory=root_dir, best_experiment=best, experiments=experiments)


def apply_safe_config_patch(base_raw: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("config_patch must be a mapping.")

    updated = copy.deepcopy(base_raw)
    for key, value in patch.items():
        if key not in SAFE_PATCH_KEYS:
            raise ValueError(f"Unsafe config_patch key: {key}")
        allowed_children = SAFE_PATCH_KEYS[key]
        if allowed_children is None:
            updated[key] = value
            continue
        if not isinstance(value, dict):
            raise ValueError(f"config_patch.{key} must be a mapping.")
        target = updated.setdefault(key, {})
        for child_key, child_value in value.items():
            if child_key not in allowed_children:
                raise ValueError(f"Unsafe config_patch key: {key}.{child_key}")
            target[child_key] = child_value
    return updated


def is_improvement(current: float, best: float, direction: str, min_delta: float) -> bool:
    if direction == "maximize":
        return current > best + min_delta
    return current < best - min_delta


def _build_research_advisor(config: TuningConfig) -> ResearchAdvisor:
    if config.research.llm_enabled:
        return OpenAIResearchAdvisor(config)
    return DeterministicResearchAdvisor()


def _safe_suggest(advisor: ResearchAdvisor, context: dict[str, Any]) -> ResearchSuggestion:
    try:
        suggestion = advisor.suggest(context)
        return _coerce_suggestion(
            {
                "description": suggestion.description,
                "hypothesis": suggestion.hypothesis,
                "config_patch": suggestion.config_patch,
                "expected_improvement": suggestion.expected_improvement,
                "risk_notes": suggestion.risk_notes,
            }
            | (suggestion.raw or {})
        )
    except Exception as exc:  # noqa: BLE001 - invalid suggestions fall back.
        fallback = DeterministicResearchAdvisor().suggest(context)
        return ResearchSuggestion(
            description=f"{fallback.description} (fallback after rejected suggestion)",
            hypothesis=fallback.hypothesis,
            config_patch=fallback.config_patch,
            expected_improvement=fallback.expected_improvement,
            risk_notes=f"Rejected advisor suggestion: {exc}",
            raw={
                "description": fallback.description,
                "config_patch": fallback.config_patch,
                "rejected_error": str(exc),
            },
        )


def _coerce_suggestion(raw: dict[str, Any]) -> ResearchSuggestion:
    required = {"description", "hypothesis", "config_patch", "expected_improvement", "risk_notes"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Research suggestion missing fields: {missing}")
    apply_safe_config_patch({}, raw["config_patch"])
    return ResearchSuggestion(
        description=str(raw["description"]),
        hypothesis=str(raw["hypothesis"]),
        config_patch=dict(raw["config_patch"]),
        expected_improvement=str(raw["expected_improvement"]),
        risk_notes=str(raw["risk_notes"]),
        raw=raw,
    )


def _raw_config(config: TuningConfig) -> dict[str, Any]:
    raw = config.to_dict()
    raw.pop("direction", None)
    return raw


def _prepare_experiment_raw(
    base_config: TuningConfig,
    raw: dict[str, Any],
    experiment_id: int,
    experiments_dir: Path,
) -> dict[str, Any]:
    updated = copy.deepcopy(raw)
    updated.setdefault("output", {})["directory"] = str(experiments_dir / f"experiment_{experiment_id:03d}")
    optimization = updated.setdefault("optimization", {})
    optimization["study_name"] = f"{base_config.optimization.study_name}-research-{experiment_id:03d}"
    updated.setdefault("research", {})["enabled"] = False
    return updated


def _decide_status(
    config: TuningConfig,
    result: RunResult,
    best: ResearchExperiment | None,
    experiment_id: int,
) -> str:
    if experiment_id == 0 or best is None:
        return "keep"
    if is_improvement(result.best_score, best.score, config.direction, config.research.improvement_min_delta):
        return "keep"
    return "discard"


def _experiment_from_result(
    experiment_id: int,
    status: str,
    result: RunResult,
    config: TuningConfig,
    config_path: Path,
    description: str,
) -> ResearchExperiment:
    return ResearchExperiment(
        experiment_id=experiment_id,
        status=status,
        score=result.best_score,
        task=config.task,
        metric=config.optimization.metric,
        model=str(result.best_params.get("model", "unknown")),
        config_path=config_path,
        output_directory=result.output_directory,
        description=description,
        metrics=result.metrics,
    )


def _research_context(
    config: TuningConfig,
    program_text: str,
    base_raw: dict[str, Any],
    best: ResearchExperiment | None,
    experiments: list[ResearchExperiment],
    next_experiment_id: int,
) -> dict[str, Any]:
    allowed_models = sorted(CLASSIFICATION_MODELS if config.task == "classification" else REGRESSION_MODELS)
    allowed_metrics = sorted(CLASSIFICATION_METRICS if config.task == "classification" else REGRESSION_METRICS)
    return {
        "program": program_text,
        "next_experiment_id": next_experiment_id,
        "task": config.task,
        "direction": config.direction,
        "metric": config.optimization.metric,
        "allowed_models": allowed_models,
        "allowed_metrics": allowed_metrics,
        "safe_patch_fields": SAFE_PATCH_KEYS,
        "base_config": base_raw,
        "best_experiment": best.__dict__ if best else None,
        "history": [experiment.__dict__ for experiment in experiments],
        "response_schema": {
            "description": "short experiment description",
            "hypothesis": "why this may improve the metric",
            "config_patch": "safe config patch only",
            "expected_improvement": "expected metric impact",
            "risk_notes": "risk or tradeoff notes",
        },
    }


def _read_program(path: Path | None) -> str:
    if path is None:
        return "Run bounded config-only experiments. Keep improvements and discard regressions."
    if not path.exists():
        raise FileNotFoundError(f"Research program not found: {path}")
    return path.read_text(encoding="utf-8")


def _write_results_header(path: Path) -> None:
    path.write_text("experiment_id\tscore\tstatus\ttask\tmetric\tmodel\tconfig_path\tdescription\n", encoding="utf-8")


def _append_result(path: Path, experiment: ResearchExperiment) -> None:
    line = "\t".join(
        [
            f"{experiment.experiment_id:03d}",
            f"{experiment.score:.6f}",
            experiment.status,
            experiment.task,
            experiment.metric,
            experiment.model,
            str(experiment.config_path),
            experiment.description.replace("\t", " ").replace("\n", " "),
        ]
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _append_log(path: Path, experiment: ResearchExperiment, suggestion_payload: dict[str, Any] | None) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"## Experiment {experiment.experiment_id:03d}\n\n")
        handle.write(f"- Status: `{experiment.status}`\n")
        handle.write(f"- Score: `{experiment.score:.6f}`\n")
        handle.write(f"- Model: `{experiment.model}`\n")
        handle.write(f"- Config: `{experiment.config_path}`\n")
        handle.write(f"- Description: {experiment.description}\n")
        if suggestion_payload:
            handle.write("\n```json\n")
            handle.write(json.dumps(suggestion_payload, indent=2, default=str))
            handle.write("\n```\n")
        handle.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Research advisor response must be a JSON object.")
    return parsed

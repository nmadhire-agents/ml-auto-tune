from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ml_auto_tune.advisor import Advisor, AdvisorContext, AdvisorResponse, build_advisor
from ml_auto_tune.config import ALLOWED_MODELS, TuningConfig
from ml_auto_tune.data import split_features_target
from ml_auto_tune.models import build_pipeline


@dataclass(frozen=True)
class RunResult:
    output_directory: Path
    best_score: float
    best_params: dict[str, Any]
    metrics: dict[str, float]
    advisor_responses: list[AdvisorResponse] = field(default_factory=list)


def run_tuning(config: TuningConfig, advisor: Advisor | None = None) -> RunResult:
    X_train, X_valid, y_train, y_valid = split_features_target(config.data)
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction=config.direction,
        study_name=config.optimization.study_name,
        sampler=TPESampler(seed=config.optimization.random_state),
    )

    active_models = list(config.models)
    advisor_client = advisor or (build_advisor(config.advisor) if config.advisor.enabled else None)
    advisor_responses: list[AdvisorResponse] = []
    best_score: float | None = None
    trials_since_improvement = 0

    def objective(trial: optuna.Trial) -> float:
        pipeline = build_pipeline(trial, X_train, config.models, config.optimization.random_state)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)
        return _score(config.optimization.metric, y_valid, predictions)

    for _ in range(config.optimization.n_trials):
        study.optimize(objective, n_trials=1, timeout=config.optimization.timeout_seconds)
        current_best = float(study.best_value)
        if best_score is None or _is_improved(
            current_best,
            best_score,
            config.direction,
            config.optimization.min_delta,
        ):
            best_score = current_best
            trials_since_improvement = 0
        else:
            trials_since_improvement += 1

        if (
            advisor_client
            and config.advisor.trigger == "plateau"
            and trials_since_improvement >= config.optimization.plateau_trials
        ):
            response = advisor_client.advise(
                _build_advisor_context(config, study, active_models, trials_since_improvement)
            )
            advisor_responses.append(response)
            active_models = _apply_safe_suggestions(response, list(config.models))
            for model_name in active_models:
                study.enqueue_trial({"model": model_name}, skip_if_exists=True)
            trials_since_improvement = 0

    if advisor_client and config.advisor.trigger == "end":
        advisor_responses.append(
            advisor_client.advise(_build_advisor_context(config, study, active_models, trials_since_improvement))
        )

    evaluation_pipeline = build_pipeline(study.best_trial, X_train, list(config.models), config.optimization.random_state)
    evaluation_pipeline.fit(X_train, y_train)
    validation_predictions = evaluation_pipeline.predict(X_valid)
    metrics = _all_metrics(y_valid, validation_predictions)

    best_pipeline = build_pipeline(study.best_trial, X_train, list(config.models), config.optimization.random_state)
    best_pipeline.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

    _write_artifacts(config, output_dir, study, best_pipeline, metrics, advisor_responses)
    return RunResult(
        output_directory=output_dir,
        best_score=float(study.best_value),
        best_params=dict(study.best_params),
        metrics=metrics,
        advisor_responses=advisor_responses,
    )


def _score(metric: str, y_true, predictions) -> float:
    if metric == "rmse":
        return float(root_mean_squared_error(y_true, predictions))
    if metric == "mae":
        return float(mean_absolute_error(y_true, predictions))
    if metric == "r2":
        return float(r2_score(y_true, predictions))
    raise ValueError(f"Unsupported metric: {metric}")


def _all_metrics(y_true, predictions) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _is_improved(current: float, previous: float, direction: str, min_delta: float) -> bool:
    if direction == "maximize":
        return current > previous + min_delta
    return current < previous - min_delta


def _build_advisor_context(
    config: TuningConfig,
    study: optuna.Study,
    active_models: list[str],
    trials_since_improvement: int,
) -> AdvisorContext:
    trials = sorted(study.trials, key=lambda item: item.number)[-10:]
    recent_trials = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
        }
        for trial in trials
    ]
    return AdvisorContext(
        study_name=config.optimization.study_name,
        metric=config.optimization.metric,
        direction=config.direction,
        best_score=float(study.best_value) if study.best_trial else None,
        best_params=dict(study.best_params) if study.best_trial else {},
        trials_since_improvement=trials_since_improvement,
        available_models=sorted(ALLOWED_MODELS.intersection(config.models)),
        active_models=list(active_models),
        recent_trials=recent_trials,
    )


def _apply_safe_suggestions(response: AdvisorResponse, allowed_models: list[str]) -> list[str]:
    suggested = response.structured_suggestions.get("model_candidates")
    if not isinstance(suggested, list):
        return allowed_models
    safe_models = [model for model in suggested if isinstance(model, str) and model in allowed_models]
    return safe_models or allowed_models


def _write_artifacts(
    config: TuningConfig,
    output_dir: Path,
    study: optuna.Study,
    best_pipeline,
    metrics: dict[str, float],
    advisor_responses: list[AdvisorResponse],
) -> None:
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    metrics_payload = {
        "optimized_metric": config.optimization.metric,
        "direction": config.direction,
        "best_score": float(study.best_value),
        "best_params": dict(study.best_params),
        "validation_metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.to_csv(output_dir / "trials.csv", index=False)
    joblib.dump(best_pipeline, output_dir / "best_model.joblib")

    advice_text = "\n\n".join(response.markdown for response in advisor_responses)
    if not advice_text:
        advice_text = "No advisor advice was requested during this run.\n"
    (output_dir / "advisor_advice.md").write_text(advice_text, encoding="utf-8")

    summary = [
        f"# Run Summary",
        "",
        f"- Study: `{config.optimization.study_name}`",
        f"- Optimized metric: `{config.optimization.metric}` ({config.direction})",
        f"- Best score: `{float(study.best_value):.6f}`",
        f"- Best model: `{study.best_params.get('model', 'unknown')}`",
        f"- Advisor calls: `{len(advisor_responses)}`",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

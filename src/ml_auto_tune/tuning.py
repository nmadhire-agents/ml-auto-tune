from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import optuna
import yaml
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error

from ml_auto_tune.advisor import Advisor, AdvisorContext, AdvisorResponse, build_advisor
from ml_auto_tune.config import ALLOWED_MODELS, TuningConfig
from ml_auto_tune.data import load_regression_frame, split_frame_features_target
from ml_auto_tune.models import build_pipeline


@dataclass(frozen=True)
class RunResult:
    output_directory: Path
    best_score: float
    best_params: dict[str, Any]
    metrics: dict[str, float]
    advisor_responses: list[AdvisorResponse] = field(default_factory=list)


def run_tuning(config: TuningConfig, advisor: Advisor | None = None) -> RunResult:
    frame = load_regression_frame(config.data)
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
        split_random_state = _trial_split_random_state(config, trial.number)
        X_train, X_valid, y_train, y_valid = split_frame_features_target(
            frame,
            target=config.data.target,
            features=config.data.features,
            validation_size=config.data.validation_size,
            random_state=split_random_state,
            task=config.task,
        )
        pipeline = build_pipeline(trial, X_train, config.models, config.optimization.random_state, task=config.task)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)
        metrics = _all_metrics(config.task, y_valid, predictions, pipeline, X_valid)
        trial.set_user_attr("split_random_state", split_random_state)
        trial.set_user_attr("validation_metrics", metrics)
        return metrics[config.optimization.metric]

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

        latest_trial = study.trials[-1]
        if advisor_client and config.advisor.trigger == "each_trial":
            advisor_responses.append(
                advisor_client.advise(
                    _build_advisor_context(
                        config,
                        study,
                        active_models,
                        trials_since_improvement,
                        validation_metrics=latest_trial.user_attrs.get("validation_metrics"),
                    )
                )
            )

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

    X_train, X_valid, y_train, y_valid = split_frame_features_target(
        frame,
        target=config.data.target,
        features=config.data.features,
        validation_size=config.data.validation_size,
        random_state=int(study.best_trial.user_attrs.get("split_random_state", config.data.random_state)),
        task=config.task,
    )
    evaluation_pipeline = build_pipeline(
        study.best_trial,
        X_train,
        list(config.models),
        config.optimization.random_state,
        task=config.task,
    )
    evaluation_pipeline.fit(X_train, y_train)
    validation_predictions = evaluation_pipeline.predict(X_valid)
    metrics = _all_metrics(config.task, y_valid, validation_predictions, evaluation_pipeline, X_valid)

    if advisor_client and config.advisor.trigger == "end":
        advisor_responses.append(
            advisor_client.advise(
                _build_advisor_context(
                    config,
                    study,
                    active_models,
                    trials_since_improvement,
                    validation_metrics=metrics,
                )
            )
        )

    best_pipeline = build_pipeline(
        study.best_trial,
        X_train,
        list(config.models),
        config.optimization.random_state,
        task=config.task,
    )
    X_all = frame[list(config.data.features)] if config.data.features is not None else frame.drop(columns=[config.data.target])
    y_all = frame[config.data.target]
    best_pipeline.fit(X_all, y_all)

    _write_artifacts(config, output_dir, study, best_pipeline, metrics, advisor_responses)
    return RunResult(
        output_directory=output_dir,
        best_score=float(study.best_value),
        best_params=dict(study.best_params),
        metrics=metrics,
        advisor_responses=advisor_responses,
    )


def _all_metrics(task: str, y_true, predictions, pipeline, X_valid) -> dict[str, float]:
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "f1_macro": float(f1_score(y_true, predictions, average="macro")),
            "roc_auc": _roc_auc(y_true, pipeline, X_valid),
        }
    return {
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _roc_auc(y_true, pipeline, X_valid) -> float:
    probabilities = pipeline.predict_proba(X_valid)
    if probabilities.shape[1] == 2:
        return float(roc_auc_score(y_true, probabilities[:, 1]))
    return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))


def _trial_split_random_state(config: TuningConfig, trial_number: int) -> int:
    if config.optimization.repeated_splits:
        return config.data.random_state + trial_number
    return config.data.random_state


def _is_improved(current: float, previous: float, direction: str, min_delta: float) -> bool:
    if direction == "maximize":
        return current > previous + min_delta
    return current < previous - min_delta


def _build_advisor_context(
    config: TuningConfig,
    study: optuna.Study,
    active_models: list[str],
    trials_since_improvement: int,
    validation_metrics: dict[str, float] | None = None,
) -> AdvisorContext:
    trials = sorted(study.trials, key=lambda item: item.number)[-10:]
    recent_trials = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
            "split_random_state": trial.user_attrs.get("split_random_state"),
            "validation_metrics": trial.user_attrs.get("validation_metrics"),
        }
        for trial in trials
    ]
    return AdvisorContext(
        study_name=config.optimization.study_name,
        task=config.task,
        metric=config.optimization.metric,
        direction=config.direction,
        best_score=float(study.best_value) if study.best_trial else None,
        best_params=dict(study.best_params) if study.best_trial else {},
        trials_since_improvement=trials_since_improvement,
        available_models=sorted(ALLOWED_MODELS.intersection(config.models)),
        active_models=list(active_models),
        recent_trials=recent_trials,
        validation_metrics=validation_metrics,
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
        "best_split_random_state": study.best_trial.user_attrs.get("split_random_state"),
        "validation_metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
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
        f"- Best split random state: `{study.best_trial.user_attrs.get('split_random_state')}`",
        f"- Advisor calls: `{len(advisor_responses)}`",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

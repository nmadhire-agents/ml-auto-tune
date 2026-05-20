from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_TASKS = {"regression", "classification"}
REGRESSION_METRICS = {"rmse", "mae", "r2"}
CLASSIFICATION_METRICS = {"accuracy", "f1_macro", "roc_auc"}
ALLOWED_METRICS = REGRESSION_METRICS | CLASSIFICATION_METRICS
REGRESSION_MODELS = {
    "linear_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "ridge",
    "elastic_net",
}
CLASSIFICATION_MODELS = {
    "logistic_regression",
    "random_forest_classifier",
    "extra_trees_classifier",
    "hist_gradient_boosting_classifier",
}
ALLOWED_MODELS = REGRESSION_MODELS | CLASSIFICATION_MODELS


@dataclass(frozen=True)
class DataSettings:
    path: Path
    target: str
    features: tuple[str, ...] | None = None
    validation_size: float = 0.25
    random_state: int = 42


@dataclass(frozen=True)
class OptimizationSettings:
    metric: str = "rmse"
    n_trials: int = 20
    timeout_seconds: int | None = None
    plateau_trials: int = 5
    min_delta: float = 0.001
    study_name: str = "ml-auto-tune"
    random_state: int = 42
    repeated_splits: bool = False


@dataclass(frozen=True)
class AdvisorSettings:
    enabled: bool = True
    provider: str = "mock"
    trigger: str = "plateau"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class OutputSettings:
    directory: Path = Path("runs/latest")


@dataclass(frozen=True)
class TuningConfig:
    data: DataSettings
    task: str = "regression"
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    advisor: AdvisorSettings = field(default_factory=AdvisorSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    models: tuple[str, ...] = ("linear_regression",)

    @property
    def direction(self) -> str:
        if self.task == "classification":
            return "maximize"
        return "maximize" if self.optimization.metric == "r2" else "minimize"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data"]["path"] = str(self.data.path)
        result["output"]["directory"] = str(self.output.directory)
        result["models"] = list(self.models)
        result["direction"] = self.direction
        return result


def load_config(path: str | Path) -> TuningConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return parse_config(raw, base_dir=config_path.parent)


def parse_config(raw: dict[str, Any], base_dir: Path | None = None) -> TuningConfig:
    base = base_dir or Path.cwd()
    task = str(raw.get("task", "regression")).lower()
    if task not in ALLOWED_TASKS:
        raise ValueError(f"task must be one of {sorted(ALLOWED_TASKS)}.")

    data_raw = _mapping(raw.get("data"), "data")
    opt_raw = _mapping(raw.get("optimization", {}), "optimization")
    advisor_raw = _mapping(raw.get("advisor", {}), "advisor")
    output_raw = _mapping(raw.get("output", {}), "output")

    target = data_raw.get("target")
    if not target:
        raise ValueError("data.target is required.")

    data_path_value = data_raw.get("path")
    if not data_path_value:
        raise ValueError("data.path is required.")
    data_path = _resolve_path(data_path_value, base)

    validation_size = float(data_raw.get("validation_size", 0.25))
    if not 0 < validation_size < 1:
        raise ValueError("data.validation_size must be between 0 and 1.")

    features_raw = data_raw.get("features")
    if features_raw is not None:
        if not isinstance(features_raw, list) or not all(isinstance(item, str) for item in features_raw):
            raise ValueError("data.features must be a list of column names.")
        if not features_raw:
            raise ValueError("data.features must include at least one column when provided.")

    metric = str(opt_raw.get("metric", "rmse")).lower()
    allowed_metrics = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    if metric not in allowed_metrics:
        raise ValueError(f"optimization.metric must be one of {sorted(allowed_metrics)} for task '{task}'.")

    n_trials = int(opt_raw.get("n_trials", 20))
    if n_trials < 1:
        raise ValueError("optimization.n_trials must be at least 1.")

    plateau_trials = int(opt_raw.get("plateau_trials", 5))
    if plateau_trials < 1:
        raise ValueError("optimization.plateau_trials must be at least 1.")

    default_models = ("logistic_regression",) if task == "classification" else ("linear_regression",)
    allowed_models = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    models_raw = raw.get("models", default_models)
    if not isinstance(models_raw, list | tuple) or not all(isinstance(item, str) for item in models_raw):
        raise ValueError("models must be a list of model names.")
    model_names = tuple(models_raw)
    unknown_models = sorted(set(model_names) - allowed_models)
    if unknown_models:
        raise ValueError(f"Unknown model names for task '{task}': {unknown_models}.")
    if not model_names:
        raise ValueError("models must include at least one model name.")

    timeout_value = opt_raw.get("timeout_seconds")
    timeout_seconds = int(timeout_value) if timeout_value is not None else None
    if timeout_seconds is not None and timeout_seconds < 1:
        raise ValueError("optimization.timeout_seconds must be at least 1 when provided.")

    trigger = str(advisor_raw.get("trigger", "plateau"))
    if trigger not in {"plateau", "end", "each_trial"}:
        raise ValueError("advisor.trigger must be 'plateau', 'end', or 'each_trial'.")

    provider = str(advisor_raw.get("provider", "mock"))
    if provider not in {"mock", "openai_compatible"}:
        raise ValueError("advisor.provider must be 'mock' or 'openai_compatible'.")

    advisor_enabled = _as_bool(advisor_raw.get("enabled", True), "advisor.enabled")
    repeated_splits = _as_bool(opt_raw.get("repeated_splits", False), "optimization.repeated_splits")

    output_directory = _resolve_path(output_raw.get("directory", "runs/latest"), base)

    return TuningConfig(
        task=task,
        data=DataSettings(
            path=data_path,
            target=str(target),
            features=tuple(features_raw) if features_raw else None,
            validation_size=validation_size,
            random_state=int(data_raw.get("random_state", 42)),
        ),
        optimization=OptimizationSettings(
            metric=metric,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            plateau_trials=plateau_trials,
            min_delta=float(opt_raw.get("min_delta", 0.001)),
            study_name=str(opt_raw.get("study_name", "ml-auto-tune")),
            random_state=int(opt_raw.get("random_state", 42)),
            repeated_splits=repeated_splits,
        ),
        advisor=AdvisorSettings(
            enabled=advisor_enabled,
            provider=provider,
            trigger=trigger,
            api_key=advisor_raw.get("api_key"),
            base_url=advisor_raw.get("base_url"),
            model=advisor_raw.get("model"),
        ),
        output=OutputSettings(directory=output_directory),
        models=model_names,
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be a boolean.")

from pathlib import Path

import pytest

from ml_auto_tune.config import parse_config


def test_parse_config_resolves_paths(tmp_path: Path) -> None:
    config = parse_config(
        {
            "data": {"path": "sample.csv", "target": "target"},
            "output": {"directory": "runs/test"},
            "optimization": {"repeated_splits": True},
            "advisor": {"trigger": "each_trial"},
            "models": ["linear_regression"],
        },
        base_dir=tmp_path,
    )

    assert config.data.path == tmp_path / "sample.csv"
    assert config.output.directory == tmp_path / "runs/test"
    assert config.direction == "minimize"
    assert config.models == ("linear_regression",)
    assert config.optimization.repeated_splits is True
    assert config.advisor.trigger == "each_trial"


def test_parse_config_with_features(tmp_path: Path) -> None:
    config = parse_config(
        {
            "data": {
                "path": "sample.csv",
                "target": "target",
                "features": ["age", "bmi"],
            },
            "models": ["linear_regression"],
        },
        base_dir=tmp_path,
    )
    assert config.data.features == ("age", "bmi")


def test_parse_classification_config(tmp_path: Path) -> None:
    config = parse_config(
        {
            "task": "classification",
            "data": {"path": "sample.csv", "target": "target"},
            "optimization": {"metric": "f1_macro"},
            "models": ["logistic_regression", "random_forest_classifier"],
        },
        base_dir=tmp_path,
    )

    assert config.task == "classification"
    assert config.direction == "maximize"
    assert config.models == ("logistic_regression", "random_forest_classifier")


def test_parse_classification_config_rejects_regression_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="for task 'classification'"):
        parse_config(
            {
                "task": "classification",
                "data": {"path": "sample.csv", "target": "target"},
                "optimization": {"metric": "rmse"},
                "models": ["logistic_regression"],
            },
            base_dir=tmp_path,
        )


def test_parse_config_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        parse_config(
            {
                "data": {"path": "sample.csv", "target": "target"},
                "models": ["made_up_model"],
            },
            base_dir=tmp_path,
        )


def test_parse_config_rejects_non_list_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="models must be a list"):
        parse_config(
            {
                "data": {"path": "sample.csv", "target": "target"},
                "models": "linear_regression",
            },
            base_dir=tmp_path,
        )


def test_parse_config_rejects_non_boolean_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="optimization.repeated_splits"):
        parse_config(
            {
                "data": {"path": "sample.csv", "target": "target"},
                "optimization": {"repeated_splits": "yes"},
            },
            base_dir=tmp_path,
        )


def test_parse_research_config_resolves_program_path(tmp_path: Path) -> None:
    config = parse_config(
        {
            "data": {"path": "sample.csv", "target": "target"},
            "research": {
                "enabled": True,
                "max_experiments": 3,
                "improvement_min_delta": 0.01,
                "llm_enabled": False,
                "program_path": "program.md",
            },
        },
        base_dir=tmp_path,
    )

    assert config.research.enabled is True
    assert config.research.max_experiments == 3
    assert config.research.improvement_min_delta == 0.01
    assert config.research.program_path == tmp_path / "program.md"

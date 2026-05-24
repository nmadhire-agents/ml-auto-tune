from pathlib import Path

import pytest

from ml_auto_tune.autoresearch import (
    ResearchSuggestion,
    apply_safe_config_patch,
    is_improvement,
    run_autoresearch,
)
from ml_auto_tune.config import parse_config
from ml_auto_tune.sample_data import make_sample_data


class StaticResearchAdvisor:
    def __init__(self, suggestion: ResearchSuggestion):
        self.suggestion = suggestion

    def suggest(self, context):
        return self.suggestion


class InvalidResearchAdvisor:
    def suggest(self, context):
        return ResearchSuggestion(
            description="Unsafe output change",
            hypothesis="Invalid suggestion should be rejected.",
            config_patch={"output": {"directory": "unsafe"}},
            expected_improvement="None",
            risk_notes="Unsafe patch.",
        )


def test_apply_safe_config_patch_allows_safe_fields() -> None:
    updated = apply_safe_config_patch(
        {
            "models": ["linear_regression"],
            "optimization": {"n_trials": 2},
            "advisor": {"enabled": False},
        },
        {
            "models": ["ridge"],
            "optimization": {"n_trials": 3, "repeated_splits": True},
            "advisor": {"enabled": True, "trigger": "end"},
        },
    )

    assert updated["models"] == ["ridge"]
    assert updated["optimization"]["n_trials"] == 3
    assert updated["optimization"]["repeated_splits"] is True
    assert updated["advisor"]["trigger"] == "end"


def test_apply_safe_config_patch_rejects_unsafe_fields() -> None:
    with pytest.raises(ValueError, match="Unsafe config_patch key"):
        apply_safe_config_patch({}, {"output": {"directory": "somewhere"}})

    with pytest.raises(ValueError, match="Unsafe config_patch key"):
        apply_safe_config_patch({}, {"data": {"path": "other.csv"}})


def test_is_improvement_handles_minimize_and_maximize() -> None:
    assert is_improvement(0.8, 1.0, "minimize", 0.01)
    assert not is_improvement(0.995, 1.0, "minimize", 0.01)
    assert is_improvement(0.9, 0.8, "maximize", 0.01)
    assert not is_improvement(0.805, 0.8, "maximize", 0.01)


def test_run_autoresearch_regression_with_mock_advisor(tmp_path: Path) -> None:
    data_path = make_sample_data(tmp_path / "sample.csv", rows=60)
    program_path = tmp_path / "program.md"
    program_path.write_text("Try safe config-only changes.", encoding="utf-8")
    config = parse_config(
        {
            "data": {"path": str(data_path), "target": "target"},
            "optimization": {"metric": "rmse", "n_trials": 2, "study_name": "research-regression-test"},
            "advisor": {"enabled": False},
            "research": {"enabled": True, "max_experiments": 2, "program_path": str(program_path)},
            "output": {"directory": str(tmp_path / "research")},
            "models": ["linear_regression"],
        },
        base_dir=tmp_path,
    )
    advisor = StaticResearchAdvisor(
        ResearchSuggestion(
            description="Try ridge",
            hypothesis="Regularization may improve validation RMSE.",
            config_patch={"models": ["ridge"], "optimization": {"n_trials": 2}},
            expected_improvement="Lower RMSE",
            risk_notes="Small search budget.",
        )
    )

    result = run_autoresearch(config, research_advisor=advisor)

    assert len(result.experiments) == 2
    assert result.best_experiment.status == "keep"
    assert (tmp_path / "research" / "research_results.tsv").exists()
    assert (tmp_path / "research" / "configs" / "experiment_001.yaml").exists()
    assert (tmp_path / "research" / "llm_suggestions.jsonl").exists()


def test_run_autoresearch_records_rejected_suggestion_and_fallback(tmp_path: Path) -> None:
    data_path = make_sample_data(tmp_path / "sample.csv", rows=60)
    config = parse_config(
        {
            "data": {"path": str(data_path), "target": "target"},
            "optimization": {"metric": "rmse", "n_trials": 1, "study_name": "research-fallback-test"},
            "advisor": {"enabled": False},
            "research": {"enabled": True, "max_experiments": 2},
            "output": {"directory": str(tmp_path / "research-fallback")},
            "models": ["linear_regression"],
        },
        base_dir=tmp_path,
    )

    run_autoresearch(config, research_advisor=InvalidResearchAdvisor())

    suggestions = (tmp_path / "research-fallback" / "llm_suggestions.jsonl").read_text(encoding="utf-8")
    assert '"accepted": false' in suggestions
    assert '"fallback_used": true' in suggestions


def test_run_autoresearch_classification_with_mock_advisor(tmp_path: Path) -> None:
    data_path = make_sample_data(tmp_path / "classification.csv", rows=80, task="classification")
    config = parse_config(
        {
            "task": "classification",
            "data": {"path": str(data_path), "target": "target"},
            "optimization": {"metric": "f1_macro", "n_trials": 2, "study_name": "research-classification-test"},
            "advisor": {"enabled": False},
            "research": {"enabled": True, "max_experiments": 2},
            "output": {"directory": str(tmp_path / "classification-research")},
            "models": ["logistic_regression"],
        },
        base_dir=tmp_path,
    )
    advisor = StaticResearchAdvisor(
        ResearchSuggestion(
            description="Try random forest classifier",
            hypothesis="Tree ensembles may improve macro F1.",
            config_patch={"models": ["random_forest_classifier"], "optimization": {"n_trials": 2}},
            expected_improvement="Higher F1",
            risk_notes="Slightly less interpretable.",
        )
    )

    result = run_autoresearch(config, research_advisor=advisor)

    assert len(result.experiments) == 2
    assert result.best_experiment.metric == "f1_macro"
    assert (tmp_path / "classification-research" / "best_config.yaml").exists()

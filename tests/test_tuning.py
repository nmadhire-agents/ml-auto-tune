from pathlib import Path

import pandas as pd

from ml_auto_tune.config import parse_config
from ml_auto_tune.sample_data import make_sample_data
from ml_auto_tune.tuning import run_tuning


def test_run_tuning_writes_expected_artifacts(tmp_path: Path) -> None:
    data_path = make_sample_data(tmp_path / "sample.csv", rows=60)
    config = parse_config(
        {
            "data": {"path": str(data_path), "target": "target"},
            "optimization": {
                "metric": "rmse",
                "n_trials": 3,
                "plateau_trials": 1,
                "min_delta": 1_000_000,
                "study_name": "test-study",
            },
            "advisor": {"enabled": True, "provider": "mock", "trigger": "plateau"},
            "output": {"directory": str(tmp_path / "run")},
            "models": ["ridge", "elastic_net"],
        },
        base_dir=tmp_path,
    )

    result = run_tuning(config)

    assert result.best_score > 0
    assert (tmp_path / "run" / "best_model.joblib").exists()
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "trials.csv").exists()
    assert (tmp_path / "run" / "advisor_advice.md").exists()
    assert result.advisor_responses


def test_sample_data_has_numeric_target(tmp_path: Path) -> None:
    data_path = make_sample_data(tmp_path / "sample.csv", rows=50)
    frame = pd.read_csv(data_path)

    assert pd.api.types.is_numeric_dtype(frame["target"])

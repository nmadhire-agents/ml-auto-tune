from pathlib import Path

import pandas as pd

from ml_auto_tune.sample_data import make_sample_data


def test_make_sample_data_writes_regression_csv(tmp_path: Path) -> None:
    output = make_sample_data(tmp_path / "sample.csv", rows=40)

    frame = pd.read_csv(output)
    assert len(frame) == 40
    assert "target" in frame.columns
    assert "bmi_band" in frame.columns


def test_make_sample_data_writes_classification_csv(tmp_path: Path) -> None:
    output = make_sample_data(tmp_path / "classification.csv", rows=50, task="classification")

    frame = pd.read_csv(output)
    assert len(frame) == 50
    assert "target" in frame.columns
    assert "mean_radius_band" in frame.columns
    assert sorted(frame["target"].unique().tolist()) == [0, 1]

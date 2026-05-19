from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from ml_auto_tune.data import ensure_parent


def make_sample_data(
    output: str | Path,
    rows: int = 180,
    random_state: int = 42,
    task: str = "regression",
) -> Path:
    if rows < 20:
        raise ValueError("rows must be at least 20.")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'.")

    dataset = load_breast_cancer(as_frame=True) if task == "classification" else load_diabetes(as_frame=True)
    frame = dataset.frame.sample(
        n=min(rows, len(dataset.frame)),
        random_state=random_state,
    ).reset_index(drop=True)
    if task == "classification":
        frame = frame.rename(columns={column: _clean_column_name(column) for column in frame.columns})
        frame["mean_radius_band"] = pd.qcut(
            frame["mean_radius"],
            q=4,
            labels=["low", "medium_low", "medium_high", "high"],
        ).astype(str)
    else:
        frame["bmi_band"] = pd.qcut(
            frame["bmi"],
            q=4,
            labels=["low", "medium_low", "medium_high", "high"],
        ).astype(str)

    output_path = Path(output).expanduser().resolve()
    ensure_parent(output_path)
    frame.to_csv(output_path, index=False)
    return output_path


def _clean_column_name(column: str) -> str:
    return column.replace(" ", "_").replace("-", "_")

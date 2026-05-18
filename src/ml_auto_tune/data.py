from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_auto_tune.config import DataSettings


def load_regression_frame(settings: DataSettings) -> pd.DataFrame:
    if not settings.path.exists():
        raise FileNotFoundError(f"Dataset not found: {settings.path}")
    frame = pd.read_csv(settings.path)
    if settings.target not in frame.columns:
        raise ValueError(f"Target column '{settings.target}' not found in {settings.path}.")
    if len(frame) < 5:
        raise ValueError("Dataset must contain at least 5 rows.")
    return frame


def split_features_target(settings: DataSettings):
    frame = load_regression_frame(settings)
    y = frame[settings.target]
    X = frame.drop(columns=[settings.target])
    return train_test_split(
        X,
        y,
        test_size=settings.validation_size,
        random_state=settings.random_state,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


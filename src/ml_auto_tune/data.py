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
    if settings.features is not None:
        missing = sorted(set(settings.features) - set(frame.columns))
        if missing:
            raise ValueError(f"Configured feature columns not found in {settings.path}: {missing}")
    if len(frame) < 5:
        raise ValueError("Dataset must contain at least 5 rows.")
    return frame


def split_features_target(settings: DataSettings):
    frame = load_regression_frame(settings)
    return split_frame_features_target(
        frame,
        target=settings.target,
        features=settings.features,
        validation_size=settings.validation_size,
        random_state=settings.random_state,
        task="regression",
    )


def split_frame_features_target(
    frame: pd.DataFrame,
    target: str,
    features: tuple[str, ...] | None,
    validation_size: float,
    random_state: int,
    task: str = "regression",
):
    y = frame[target]
    X = frame[list(features)] if features is not None else frame.drop(columns=[target])
    stratify = y if task == "classification" and y.nunique() > 1 else None
    return train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

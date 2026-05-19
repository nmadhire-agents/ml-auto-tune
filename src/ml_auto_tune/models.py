from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(
    trial: optuna.Trial,
    X: pd.DataFrame,
    model_candidates: Sequence[str],
    random_state: int,
    task: str = "regression",
) -> Pipeline:
    model_name = trial.suggest_categorical("model", list(model_candidates))
    estimator = _build_estimator(trial, model_name, random_state, task)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", estimator),
        ]
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers=transformers)


def _build_estimator(trial: optuna.Trial, model_name: str, random_state: int, task: str):
    if task == "classification":
        return _build_classifier(trial, model_name, random_state)
    return _build_regressor(trial, model_name, random_state)


def _build_regressor(trial: optuna.Trial, model_name: str, random_state: int):
    if model_name == "linear_regression":
        return LinearRegression()
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("random_forest_n_estimators", 50, 250, step=50),
            max_depth=trial.suggest_int("random_forest_max_depth", 2, 18),
            min_samples_leaf=trial.suggest_int("random_forest_min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("random_forest_max_features", 0.4, 1.0),
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("extra_trees_n_estimators", 50, 250, step=50),
            max_depth=trial.suggest_int("extra_trees_max_depth", 2, 18),
            min_samples_leaf=trial.suggest_int("extra_trees_min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("extra_trees_max_features", 0.4, 1.0),
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=trial.suggest_float("hgb_learning_rate", 0.01, 0.25, log=True),
            max_iter=trial.suggest_int("hgb_max_iter", 50, 250, step=50),
            max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 8, 48),
            l2_regularization=trial.suggest_float("hgb_l2_regularization", 1e-8, 1.0, log=True),
            random_state=random_state,
        )
    if model_name == "ridge":
        return Ridge(alpha=trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True))
    if model_name == "elastic_net":
        return ElasticNet(
            alpha=trial.suggest_float("elastic_net_alpha", 1e-4, 10.0, log=True),
            l1_ratio=trial.suggest_float("elastic_net_l1_ratio", 0.05, 0.95),
            random_state=random_state,
            max_iter=10_000,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _build_classifier(trial: optuna.Trial, model_name: str, random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("logistic_regression_c", 1e-3, 100.0, log=True),
            solver="lbfgs",
            max_iter=2_000,
            random_state=random_state,
        )
    if model_name == "random_forest_classifier":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_classifier_n_estimators", 50, 250, step=50),
            max_depth=trial.suggest_int("rf_classifier_max_depth", 2, 18),
            min_samples_leaf=trial.suggest_int("rf_classifier_min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("rf_classifier_max_features", 0.4, 1.0),
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extra_trees_classifier":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_classifier_n_estimators", 50, 250, step=50),
            max_depth=trial.suggest_int("et_classifier_max_depth", 2, 18),
            min_samples_leaf=trial.suggest_int("et_classifier_min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("et_classifier_max_features", 0.4, 1.0),
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "hist_gradient_boosting_classifier":
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("hgb_classifier_learning_rate", 0.01, 0.25, log=True),
            max_iter=trial.suggest_int("hgb_classifier_max_iter", 50, 250, step=50),
            max_leaf_nodes=trial.suggest_int("hgb_classifier_max_leaf_nodes", 8, 48),
            l2_regularization=trial.suggest_float("hgb_classifier_l2_regularization", 1e-8, 1.0, log=True),
            random_state=random_state,
        )
    raise ValueError(f"Unsupported classification model: {model_name}")

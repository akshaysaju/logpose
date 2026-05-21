"""Classical ML directional classifiers (random forest, gradient boosting)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import Config


def _class_weight(y: np.ndarray) -> dict[int, float]:
    pos = float(y.mean())
    pos = min(max(pos, 1e-6), 1 - 1e-6)
    return {0: 0.5 / (1 - pos), 1: 0.5 / pos}


def train_random_forest(X: np.ndarray, y: np.ndarray, cfg: Config):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.seed,
    )
    model.fit(X, y)
    return model


def train_xgboost(X: np.ndarray, y: np.ndarray, cfg: Config):
    from xgboost import XGBClassifier

    pos = float(y.mean())
    scale_pos_weight = (1 - pos) / pos if 0 < pos < 1 else 1.0
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=cfg.seed,
        tree_method="hist",
    )
    model.fit(X, y)
    return model


def predict_proba_up(model, X: np.ndarray) -> np.ndarray:
    """P(label == 1) for any sklearn-style classifier."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    up_col = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return proba[:, up_col]


ML_TRAINERS = {
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}

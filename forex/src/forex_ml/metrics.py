"""Classification metrics for directional predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba_up: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba_up = np.asarray(proba_up, dtype=float)
    y_pred = (proba_up >= 0.5).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "base_rate_up": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    # ROC-AUC is undefined when only one class is present in y_true.
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba_up))
    else:
        out["roc_auc"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out

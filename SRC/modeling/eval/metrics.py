"""
===========================================================================
METRIKLER + KARAR ESIGI OPTIMIZASYONU
===========================================================================
SYZ2026 / MedicalVision

Dengesiz veride birincil metrik MCC ve macro-F1'dir. Karar esigi varsayilan
0.5 yerine, validasyonda secilen metrigi (mcc|macro_f1) maksimize edecek
sekilde Precision-Recall taramasiyla kalibre edilir.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def optimize_threshold(y_true, y_proba, metric: str = "mcc") -> float:
    """[0.05, 0.95] araliginda metrigi maksimize eden esigi dondurur."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    grid = np.linspace(0.05, 0.95, 91)
    best_t, best_s = 0.5, -2.0
    for t in grid:
        pred = (y_proba >= t).astype(int)
        if metric == "macro_f1":
            s = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            s = matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else -1.0
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    out = {
        "threshold": round(float(threshold), 4),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        out["auc"] = float("nan")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}

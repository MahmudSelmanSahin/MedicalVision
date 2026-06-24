"""
===========================================================================
FOCAL LOSS (ikili) - XGBoost/LightGBM custom objective
===========================================================================
SYZ2026 / MedicalVision

Dengesiz sinifta kolay (cogunluk) orneklerin katkisini azaltip zor orneklere
odaklanir. Grad/Hess kapali formu Wang et al. (imbalance-xgboost, 2019)
uygulamasindandir. sklearn API icin imza: objective(y_true, y_pred) -> (grad, hess)
(hem XGBClassifier hem LGBMClassifier bu imzayi kabul eder).
"""
from __future__ import annotations

import numpy as np


def _robust_pow(base, p):
    return np.sign(base) * (np.abs(base) ** p)


def make_focal_objective(gamma: float = 2.0):
    """Ikili focal loss objective (sklearn API: (y_true, y_pred))."""
    def focal(y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        x = np.asarray(y_pred, dtype=float)          # raw margin (logit)
        p = 1.0 / (1.0 + np.exp(-x))
        sign = (-1.0) ** y
        g1 = p * (1.0 - p)
        g2 = y + sign * p
        g3 = p + y - 1.0
        g4 = 1.0 - y - sign * p
        g5 = y + sign * p
        grad = (gamma * g3 * _robust_pow(g2, gamma) * np.log(g4 + 1e-9)
                + sign * _robust_pow(g5, gamma + 1.0))
        hess_1 = (_robust_pow(g2, gamma)
                  + gamma * sign * g3 * _robust_pow(g2, gamma - 1.0))
        hess_2 = sign * g3 * _robust_pow(g2, gamma) / g4
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma
                + (gamma + 1.0) * _robust_pow(g5, gamma)) * g1
        return grad, hess
    return focal

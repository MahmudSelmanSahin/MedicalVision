"""
===========================================================================
ACIKLANABILIRLIK - Feature Importance + SHAP
===========================================================================
SYZ2026 / MedicalVision

  * feature_importance : yerlesik (feature_importances_/coef_) -> yoksa atla
  * SHAP               : agac ailesi modellerinde TreeExplainer (hizli);
                         lineer modellerde LinearExplainer. Kernel SHAP cok
                         yavas oldugu icin ana donguden DISLANIR (opsiyonel).

Pipeline son adimi clf'tir; ondan onceki adimlar (fe->impute->scale->select)
ile X donusturulup nihai ozellik isimleri cikarilir. SelectFromModel maskesi
ve SimpleImputer'in dusurebilecegi sutunlar dikkate alinir.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _unwrap(clf):
    """GBMWrapper / sklearn estimator -> gercek booster/estimator."""
    if hasattr(clf, "model_"):       # GBMWrapper (fit edilmis)
        return clf.model_
    if hasattr(clf, "base"):         # GBMWrapper (fit oncesi) - kullanilmaz
        return clf.base
    return clf


def final_feature_names(pipe):
    fe = pipe.named_steps.get("fe")
    names = list(fe.feature_names_) if getattr(fe, "feature_names_", None) else None
    if not names:
        return None
    sel = pipe.named_steps.get("select")
    if sel is not None and sel != "passthrough" and hasattr(sel, "get_support"):
        try:
            mask = sel.get_support()
            if len(mask) == len(names):
                names = [n for n, m in zip(names, mask) if m]
        except Exception:
            pass
    return names


def _transform_to_clf(pipe, X):
    Xt = X
    for name, step in pipe.steps[:-1]:
        if step == "passthrough" or step is None:
            continue
        Xt = step.transform(Xt)
    return np.asarray(Xt, dtype=float)


def native_importance(pipe, names) -> pd.DataFrame | None:
    clf = _unwrap(pipe.named_steps["clf"])
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        coef = getattr(clf, "coef_", None)
        if coef is not None:
            imp = np.abs(np.asarray(coef)).ravel()
    if imp is None:
        return None
    imp = np.asarray(imp, dtype=float).ravel()
    if names is None or len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]
    return (pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def shap_summary(pipe, X, names, family: str, out_png: Path, max_rows: int = 200):
    """Agac/lineer modellerde SHAP ozet (bar) grafigi kaydeder. Donus: ust 5 ozellik."""
    try:
        import shap
    except Exception:
        return None
    clf = _unwrap(pipe.named_steps["clf"])
    Xt = _transform_to_clf(pipe, X)
    if len(Xt) > max_rows:
        idx = np.random.RandomState(42).choice(len(Xt), max_rows, replace=False)
        Xt = Xt[idx]
    if names is None or len(names) != Xt.shape[1]:
        names = [f"f{i}" for i in range(Xt.shape[1])]

    # XGBoost 3.x base_score'u JSON'a '[5E-1]' formatinda yazar; SHAP 0.49
    # parse edemez. XGBoost'ta SHAP atlanir (native importance kullanilir).
    if "xgboost" in type(clf).__module__:
        return None

    try:
        if family in ("gbdt", "bagging"):
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) and len(sv) == 2 else sv
        elif family == "linear":
            expl = shap.LinearExplainer(clf, Xt)
            sv = expl.shap_values(Xt)
        else:
            return None
        sv = np.asarray(sv)
        if sv.ndim == 3:                 # (n, f, classes)
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:20]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([names[i] for i in order][::-1], mean_abs[order][::-1], color="steelblue")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title("SHAP - ozellik onemi")
        fig.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        top = [names[i] for i in np.argsort(mean_abs)[::-1][:5]]
        return top
    except Exception:
        return None

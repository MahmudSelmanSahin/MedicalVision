"""
===========================================================================
MODEL REGISTRY - yetenek bayraklarina gore tam pipeline kurar
===========================================================================
SYZ2026 / MedicalVision

build_pipeline(model, scenario, ablation, y_train) ->
  imblearn Pipeline: fe -> impute -> scale -> select -> smote -> clf

Sizinti guvenli: tum on-isleme adimlari Pipeline icindedir, CV'de her fold'da
yeniden fit edilir. Senaryo (ek_scaler/ek_missing) FeaturePipeline parametresi.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capabilities import CAPABILITIES                      # noqa: E402
from data.fe_pipeline import FeaturePipeline               # noqa: E402
from models.focal import make_focal_objective             # noqa: E402


# --------------------------------------------------------------------------
# Birlesik GBM wrapper: erken durdurma (+) focal-loss proba duzeltmesi
# --------------------------------------------------------------------------
class GBMWrapper:
    """xgboost/lightgbm sklearn modelini sarar.

    * early_stopping=True : fit icinde stratified train/val ayirip eval_set +
      early_stopping ile egitir (X Pipeline'da zaten sayisallastirilmistir).
    * focal=True : custom objective kullanildiginda booster ham margin uretir;
      predict_proba ham margin'e sigmoid uygulayip 2-sutun olasilik dondurur
      (ozellikle LightGBM custom-objective predict_proba'yi 2-sutun vermez)."""

    def __init__(self, base, early_stopping: bool = False, focal: bool = False,
                 rounds: int = 50, val_frac: float = 0.2, seed: int = 42):
        self.base = base
        self.early_stopping = early_stopping
        self.focal = focal
        self.rounds, self.val_frac, self.seed = rounds, val_frac, seed

    @property
    def _is_lgbm(self):
        return "lightgbm" in type(self.base).__module__

    def fit(self, X, y):
        from sklearn.base import clone
        self.model_ = clone(self.base)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if not self.early_stopping or len(np.unique(y)) < 2 or len(y) < 20:
            self.model_.fit(X, y)
            return self
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=self.val_frac, stratify=y, random_state=self.seed)
        n_est = max(self.model_.get_params().get("n_estimators", 100) or 100, 500)
        try:
            if self._is_lgbm:
                import lightgbm as lgb
                self.model_.set_params(n_estimators=n_est)
                self.model_.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                                callbacks=[lgb.early_stopping(self.rounds, verbose=False)])
            else:
                self.model_.set_params(n_estimators=n_est, early_stopping_rounds=self.rounds)
                self.model_.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        except Exception:
            self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.focal:
            if self._is_lgbm:
                raw = self.model_.predict(X, raw_score=True)
            else:
                raw = self.model_.predict(X, output_margin=True)
            raw = np.asarray(raw, dtype=float).ravel()
            p = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1.0 - p, p])
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"base": self.base, "early_stopping": self.early_stopping,
                "focal": self.focal, "rounds": self.rounds,
                "val_frac": self.val_frac, "seed": self.seed}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


# --------------------------------------------------------------------------
def _scenario_params(scenario: str):
    # 'ek-standard_nan' -> ('standard', 'nan')
    body = scenario.replace("ek-", "")
    scaler, missing = body.split("_")
    return scaler, missing


def _class_balance(y):
    y = np.asarray(y).astype(int)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return n_neg, n_pos


def _apply_hp(est, hp):
    """HPO override'lari estimator'a uygular (early-stopping wrapper'da base'e)."""
    if not hp:
        return est
    target = est.base if isinstance(est, GBMWrapper) else est
    target.set_params(**hp)
    return est


def _n_jobs() -> int:
    """Paralel modda env ile 1'e cekilir (asiri abonelik onlemi); yoksa -1."""
    return int(os.environ.get("MODELING_N_JOBS", "-1"))


def _build_estimator(model: str, ablation: dict, y, seed: int, hp: dict | None = None):
    on = lambda f: ablation.get(f, "off") == "on"
    caps = CAPABILITIES[model]
    n_neg, n_pos = _class_balance(y)
    cw = "balanced" if on("class_weight") else None
    nj = _n_jobs()

    if model == "xgboost":
        from xgboost import XGBClassifier
        kw = dict(n_estimators=150, max_depth=4, learning_rate=0.1,
                  subsample=0.8 if on("row_subsample") else 1.0,
                  colsample_bytree=0.8 if on("feature_subsample") else 1.0,
                  reg_lambda=1.0 if on("l2") else 0.0,
                  eval_metric="logloss", tree_method="hist",
                  random_state=seed, n_jobs=nj)
        if on("class_weight"):
            kw["scale_pos_weight"] = n_neg / n_pos
        if on("focal_loss"):
            kw["objective"] = make_focal_objective(2.0)
        return GBMWrapper(XGBClassifier(**kw), early_stopping=on("early_stopping"),
                          focal=on("focal_loss"), seed=seed)

    if model == "lightgbm":
        from lightgbm import LGBMClassifier
        kw = dict(n_estimators=150, max_depth=-1, learning_rate=0.1,
                  subsample=0.8 if on("row_subsample") else 1.0,
                  subsample_freq=1 if on("row_subsample") else 0,
                  colsample_bytree=0.8 if on("feature_subsample") else 1.0,
                  reg_lambda=1.0 if on("l2") else 0.0,
                  class_weight=cw, random_state=seed, n_jobs=nj, verbose=-1)
        if on("focal_loss"):
            kw["objective"] = make_focal_objective(2.0)
        return GBMWrapper(LGBMClassifier(**kw), early_stopping=on("early_stopping"),
                          focal=on("focal_loss"), seed=seed)

    if model == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.1,
            l2_leaf_reg=3.0 if on("l2") else 1.0,
            rsm=0.8 if on("feature_subsample") else 1.0,
            auto_class_weights="Balanced" if on("class_weight") else None,
            random_state=seed, verbose=False)

    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight=cw,
            max_features="sqrt" if on("feature_subsample") else None,
            bootstrap=True, random_state=seed, n_jobs=nj)

    if model == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200, class_weight=cw,
            max_features="sqrt" if on("feature_subsample") else None,
            bootstrap=on("row_subsample"), random_state=seed, n_jobs=nj)

    if model == "adaboost":
        return AdaBoostClassifier(n_estimators=120, random_state=seed)

    if model == "svm":
        return SVC(C=1.0, kernel="rbf", probability=True,
                   class_weight=cw, random_state=seed)

    if model == "knn":
        return KNeighborsClassifier(n_neighbors=min(15, max(3, n_pos + n_neg - 1)))

    if model == "logreg":
        return LogisticRegression(
            C=1.0 if on("l2") else 1e6, penalty="l2",
            class_weight=cw, max_iter=2000, random_state=seed)

    if model in ("tabnet", "tabpfn"):
        raise NotImplementedError(f"{model} Faz 2'de eklenecek")

    raise ValueError(f"bilinmeyen model: {model}")


def build_pipeline(model: str, scenario: str, ablation: dict, y_train,
                   seed: int = 42, hp: dict | None = None) -> Pipeline:
    caps = CAPABILITIES[model]
    scaler, missing = _scenario_params(scenario)
    on = lambda f: ablation.get(f, "off") == "on"

    steps = [("fe", FeaturePipeline(ek_scaler=scaler, ek_missing=missing))]

    # NaN-native degilse VEYA SMOTE/scale gerekiyorsa imputer sart
    need_impute = (not caps.nan_native) or on("smote")
    steps.append(("impute", SimpleImputer(strategy="median") if need_impute
                  else "passthrough"))

    steps.append(("scale", StandardScaler() if caps.needs_scaling else "passthrough"))

    if on("feature_selection"):
        from sklearn.ensemble import RandomForestClassifier as _RF
        steps.append(("select", SelectFromModel(
            _RF(n_estimators=200, random_state=seed, n_jobs=_n_jobs()),
            threshold="median")))
    else:
        steps.append(("select", "passthrough"))

    if on("smote"):
        n_min = int(np.minimum((np.asarray(y_train) == 0).sum(),
                               (np.asarray(y_train) == 1).sum()))
        steps.append(("smote", SMOTE(random_state=seed,
                                     k_neighbors=max(1, min(5, n_min - 1)))))

    est = _apply_hp(_build_estimator(model, ablation, y_train, seed), hp)
    steps.append(("clf", est))
    return Pipeline(steps)

"""
XGBoost — Parametre Deneyi (baseline vs NotebookLM-ayarlı)
Aynı veri + aynı özellik seti (MI top-50), sadece hiperparametreler değişir.
Amaç: regularization'ın küçük/dengesiz panellerde etkisini ölçmek.

Çalıştırma:
  python run_param_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "VERİLER" / "TEMİZLENMİŞ"
PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
ID_COL, LABEL_COL, SEED, K = "Variant_ID", "Label", 42, 50


def numeric_df(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = pd.Categorical(X[c].fillna("__MISSING__").astype(str)).codes.astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in X.columns:
        if X[c].isna().any():
            m = X[c].median()
            X[c] = X[c].fillna(0.0 if pd.isna(m) else m)
    return X


def spw(y):
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    return (neg / pos) if pos else 1.0


BASELINE = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.9, colsample_bytree=0.9,
                tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED)

TUNED = dict(n_estimators=600, learning_rate=0.03, max_depth=4,
             subsample=0.8, colsample_bytree=0.5,
             reg_lambda=5.0, reg_alpha=1.0, min_child_weight=5, gamma=0.0,
             tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED)


def evaluate(params, Xtr, ytr, Xte, yte):
    model = XGBClassifier(scale_pos_weight=spw(ytr), **params)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte).astype(int)
    pr = model.predict_proba(Xte)[:, 1]
    return (matthews_corrcoef(yte, yp),
            f1_score(yte, yp, average="macro", zero_division=0),
            roc_auc_score(yte, pr))


def main():
    print(f"{'Panel':8} {'Param':10} {'MCC':>7} {'F1mac':>7} {'ROC':>7}")
    print("-" * 45)
    for panel in PANELS:
        p = DATA_DIR / f"YARISMA_TRAIN_{panel}_temiz.csv"
        if not p.exists():
            print(f"{panel}: veri yok"); continue
        df = pd.read_csv(p)
        y = df[LABEL_COL].astype(int)
        cols = [c for c in df.columns if c not in {ID_COL, LABEL_COL}]
        X = numeric_df(df, cols)
        k = min(K, X.shape[1])
        sel = SelectKBest(mutual_info_classif, k=k).fit(X.values, y.values)
        feats = [cols[i] for i, m in enumerate(sel.get_support()) if m]
        Xtr, Xte, ytr, yte = train_test_split(
            X[feats], y, test_size=0.2, random_state=SEED, stratify=y)

        for name, params in [("baseline", BASELINE), ("tuned", TUNED)]:
            mcc, f1m, roc = evaluate(params, Xtr, ytr, Xte, yte)
            print(f"{panel:8} {name:10} {mcc:7.3f} {f1m:7.3f} {roc:7.3f}")
        print()


if __name__ == "__main__":
    main()

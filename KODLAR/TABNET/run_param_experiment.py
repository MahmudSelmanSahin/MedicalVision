"""
TabNet — Parametre Deneyi (baseline vs NotebookLM-ayarlı)
Aynı veri + aynı özellik seti (MI top-30), sadece hiperparametreler değişir.
baseline : varsayılan TabNet, sabit 100 epoch, early stopping yok
tuned    : n_d=n_a=8, n_steps=3, gamma=1.5, mask_type='entmax', lr=2e-2,
           küçük batch (32/16), validation split + patience=15 erken durdurma

Gereksinim: pip install pytorch-tabnet torch
Çalıştırma:
  python run_param_experiment.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "VERİLER" / "TEMİZLENMİŞ"
PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
ID_COL, LABEL_COL, SEED, K = "Variant_ID", "Label", 42, 30


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
    return X.astype(np.float32)


def metrics(yte, yp, pr):
    return (matthews_corrcoef(yte, yp),
            f1_score(yte, yp, average="macro", zero_division=0),
            roc_auc_score(yte, pr))


def fit_baseline(Xtr, ytr, Xte):
    n = len(Xtr)
    bs = min(256, max(16, n)); vbs = max(8, bs // 2)
    m = TabNetClassifier(seed=SEED, verbose=0)
    m.fit(Xtr.values, ytr.values, max_epochs=100, patience=0,
          batch_size=bs, virtual_batch_size=vbs, weights=1)
    return m.predict(Xte.values).astype(int), m.predict_proba(Xte.values)[:, 1]


def fit_tuned(Xtr, ytr, Xte):
    # küçük panel için küçük batch + early stopping (validation split)
    n = len(Xtr)
    bs = 32 if n < 600 else 128
    vbs = max(8, bs // 2)
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                      random_state=SEED, stratify=ytr)
    m = TabNetClassifier(
        n_d=8, n_a=8, n_steps=3, gamma=1.5, lambda_sparse=1e-3,
        mask_type="entmax", seed=SEED, verbose=0,
        optimizer_params=dict(lr=2e-2),
    )
    m.fit(Xt.values, yt.values,
          eval_set=[(Xv.values, yv.values)], eval_metric=["auc"],
          max_epochs=200, patience=15,
          batch_size=bs, virtual_batch_size=vbs, weights=1)
    return m.predict(Xte.values).astype(int), m.predict_proba(Xte.values)[:, 1]


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
        feats = [cols[i] for i, mm in enumerate(sel.get_support()) if mm]
        Xtr, Xte, ytr, yte = train_test_split(
            X[feats], y, test_size=0.2, random_state=SEED, stratify=y)

        for name, fn in [("baseline", fit_baseline), ("tuned", fit_tuned)]:
            try:
                yp, pr = fn(Xtr, ytr, Xte)
                mcc, f1m, roc = metrics(yte.values, yp, pr)
                print(f"{panel:8} {name:10} {mcc:7.3f} {f1m:7.3f} {roc:7.3f}")
            except Exception as exc:
                print(f"{panel:8} {name:10}  HATA: {exc}")
        print()


if __name__ == "__main__":
    main()

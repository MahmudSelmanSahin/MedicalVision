"""
Hibrit Korelasyon + Sabit-Sütun Silme — TabNet ile değerlendirme
================================================================
Silme kararları modelden bağımsızdır ve XGBoost deneyinde üretilip
MODELLER/_KORELASYON_HIBRIT/silinen_ozellikler.json içine kaydedilmiştir.
Bu script aynı 4 senaryoyu (a/b/c/d) TabNet ile ölçer (apples-to-apples).

Senaryolar:
  (a) Tüm özellikler
  (b) MASTER-global silme        (master_global_drop)
  (c) Hibrit silme               (hybrid_drop[panel])
  (d) Hibrit + sabit-sütun       (hybrid_plus_constant_drop[panel])

Değerlendirme: 3-fold Stratified CV, TabNet (StandardScaler train-fold'da fit),
erken durdurma için train içinden küçük iç-validasyon (sızıntısız).

Çıktı: MODELLER/_KORELASYON_HIBRIT/hibrit_korelasyon_sonuclar_TABNET.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "VERİLER" / "ORİJİNAL"
FILE_TMPL = "YARISMA_TRAIN_{panel}.csv"
HIB_DIR = ROOT / "MODELLER" / "_KORELASYON_HIBRIT"
DROP_JSON = HIB_DIR / "silinen_ozellikler.json"
PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
ID_COL, LABEL_COL, SEED = "Variant_ID", "Label", 42
N_SPLITS, MAX_EPOCHS, PATIENCE = 3, 80, 12

import warnings
warnings.filterwarnings("ignore")
import torch
torch.manual_seed(SEED)


def numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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


def eval_tabnet(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> dict:
    Xv = X[feats].values.astype(np.float32)
    yv = y.values.astype(int)
    bs = 256 if len(yv) > 1500 else (64 if len(yv) > 300 else 32)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    mccs, f1s, rocs = [], [], []
    for tr, va in cv.split(Xv, yv):
        sc = StandardScaler().fit(Xv[tr])
        Xtr, Xte = sc.transform(Xv[tr]).astype(np.float32), sc.transform(Xv[va]).astype(np.float32)
        ytr, yte = yv[tr], yv[va]
        # erken durdurma için iç-validasyon (held-out fold'a dokunmaz)
        Xt2, Xiv, yt2, yiv = train_test_split(
            Xtr, ytr, test_size=0.15, random_state=SEED, stratify=ytr)
        clf = TabNetClassifier(seed=SEED, verbose=0, mask_type="entmax")
        clf.fit(Xt2, yt2, eval_set=[(Xiv, yiv)], eval_metric=["auc"],
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                batch_size=bs, virtual_batch_size=max(8, bs // 2), weights=1)
        yp = clf.predict(Xte).astype(int)
        pr = clf.predict_proba(Xte)[:, 1]
        mccs.append(matthews_corrcoef(yte, yp))
        f1s.append(f1_score(yte, yp, average="macro", zero_division=0))
        rocs.append(roc_auc_score(yte, pr))
    return {"n_features": len(feats), "mcc": float(np.mean(mccs)),
            "f1_macro": float(np.mean(f1s)), "roc_auc": float(np.mean(rocs))}


def main():
    drop = json.load(open(DROP_JSON, encoding="utf-8"))
    g_drop = set(drop["master_global_drop"])
    h_drop = {p: set(drop["hybrid_drop"][p]) for p in PANELS}
    hp_drop = {p: set(drop["hybrid_plus_constant_drop"][p]) for p in PANELS}

    rows = []
    print(f"{'Panel':7} {'Senaryo':26} {'feat':>4} {'MCC':>7} {'F1m':>7} {'ROC':>7}")
    print("-" * 64)
    for p in PANELS:
        df = pd.read_csv(DATA_DIR / FILE_TMPL.format(panel=p))
        cols = [c for c in df.columns if c not in {ID_COL, LABEL_COL}]
        X = numeric_df(df, cols)
        y = df[LABEL_COL].astype(int)

        sets = {
            "(a) Tum ozellikler": cols,
            "(b) MASTER-global silme": [c for c in cols if c not in g_drop],
            "(c) Hibrit silme": [c for c in cols if c not in h_drop[p]],
            "(d) Hibrit + sabit-sutun": [c for c in cols if c not in hp_drop[p]],
        }
        for name, feats in sets.items():
            m = eval_tabnet(X, y, feats)
            rows.append({"panel": p, "senaryo": name, **m})
            print(f"{p:7} {name:26} {m['n_features']:>4} "
                  f"{m['mcc']:>7.3f} {m['f1_macro']:>7.3f} {m['roc_auc']:>7.3f}")
        print("-" * 64)

    out = HIB_DIR / "hibrit_korelasyon_sonuclar_TABNET.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[OK] Kaydedildi: {out}")


if __name__ == "__main__":
    main()

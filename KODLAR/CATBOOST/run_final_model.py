"""
CatBoost — Final Model (Ham Veriden En İyi Konfigürasyon)

Her panel için ham veriden başlayıp en iyi ablasyon konfigürasyonunu çalıştırır.

En iyi konfigürasyonlar:
  MASTER : TEMİZLENMİŞ + Eksik_Veri_Korundu (NaN → CatBoost native)
  KANSER : TEMİZLENMİŞ + KNNImputer
  PAH    : ORİJİNAL   + Eksik_Veri_Korundu (NaN → CatBoost native)
  CFTR   : TEMİZLENMİŞ + AL_Sifir_EK_Medyan

Çalıştırma:
  python run_final_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))
from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, split_data,
)

ROOT      = Path("/Users/mahmudselmansahin/Teknofest")
DATA_TEMIZ = ROOT / "VERİLER" / "TEMİZLENMİŞ"
DATA_ORI   = ROOT / "VERİLER" / "ORİJİNAL"
OUT_DIR    = MODELS_ROOT / "CatBoost" / "FINAL"

AL_PREFIX = "AL_"
EK_PREFIX = "EK_"
EK_EXTRAS  = {"BLOSUM62", "DELTA_HIDRO", "SNV_PATHWAY_COUNT"}

CATBOOST_PARAMS = dict(
    iterations=250,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
)


# ---------------------------------------------------------------------------
# Yükleme & ön işleme
# ---------------------------------------------------------------------------

def load_raw(panel: str, temiz: bool) -> pd.DataFrame:
    if temiz:
        return pd.read_csv(DATA_TEMIZ / f"YARISMA_TRAIN_{panel}_temiz.csv")
    return pd.read_csv(DATA_ORI / f"YARISMA_TRAIN_{panel}.csv")


def get_xy(df: pd.DataFrame):
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[LABEL_COL] + ([ID_COL] if ID_COL in df.columns else []))
    return x, y


def cat_indices(x: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(x.columns)
            if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_string_dtype(x[c])]


def fill_cat_missing(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for c in x.columns:
        if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_string_dtype(x[c]):
            x[c] = x[c].fillna("__MISSING__").astype(str)
    return x


def impute_eksik_korundu(x: pd.DataFrame) -> pd.DataFrame:
    """NaN'ları CatBoost'a bırak; yalnızca kategorikleri string yap."""
    return fill_cat_missing(x)


def impute_knn(x: pd.DataFrame) -> pd.DataFrame:
    x = fill_cat_missing(x)
    num_cols = [c for c in x.columns
                if not pd.api.types.is_object_dtype(x[c])
                and not pd.api.types.is_string_dtype(x[c])]
    data = x[num_cols].to_numpy(copy=True).astype(float)
    x[num_cols] = pd.DataFrame(
        KNNImputer(n_neighbors=5).fit_transform(data),
        columns=num_cols, index=x.index,
    )
    return x


def impute_al_sifir_ek_medyan(x: pd.DataFrame) -> pd.DataFrame:
    """AL_→0, EK_→medyan, diğer numerikler NaN bırakılır (CatBoost native işler)."""
    x = fill_cat_missing(x)
    for c in x.columns:
        if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_string_dtype(x[c]):
            continue
        if c.startswith(AL_PREFIX):
            x[c] = x[c].fillna(0.0)
        elif c.startswith(EK_PREFIX) or c in EK_EXTRAS:
            x[c] = x[c].fillna(x[c].median())
    return x


PANEL_CONFIG = {
    "MASTER": dict(temiz=True,  impute=impute_eksik_korundu),
    "KANSER": dict(temiz=True,  impute=impute_knn),
    "PAH":    dict(temiz=False, impute=impute_eksik_korundu),
    "CFTR":   dict(temiz=True,  impute=impute_al_sifir_ek_medyan),
}


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------

def compute(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "pr_auc":    roc_auc_score(y_true, y_prob),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def run_panel(panel: str) -> dict:
    cfg = PANEL_CONFIG[panel]
    df  = load_raw(panel, cfg["temiz"])
    x, y = get_xy(df)
    x = cfg["impute"](x)
    cat_idx = cat_indices(x)

    x_train, x_test, y_train, y_test = split_data(x, y)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(x_train, y_train, cat_features=cat_idx)

    y_pred = model.predict(x_test).astype(int)
    y_prob = model.predict_proba(x_test)[:, 1]

    m = compute(y_test.values, y_pred, y_prob)
    veri = "TEMİZLENMİŞ" if cfg["temiz"] else "ORİJİNAL"
    print(f"  {panel:8s} [{veri}]  MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  "
          f"PR-AUC={m['pr_auc']:.4f}  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    return {"panel": panel, "veri": veri, "n_feat": x.shape[1], **m}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("CatBoost Final Model\n")
    rows = [run_panel(p) for p in ["MASTER", "KANSER", "PAH", "CFTR"]]
    df = pd.DataFrame(rows)
    out = OUT_DIR / "CatBoost_final_results.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()

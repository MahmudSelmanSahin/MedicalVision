"""
CatBoost — Final Model (Ham Veriden En İyi Konfigürasyon)

Her panel için ham veriden başlayıp en iyi konfigürasyonu çalıştırır.

En iyi konfigürasyonlar (deterministik):
  MASTER : TEMİZLENMİŞ + WRAPPER_RFE özellik seçimi (k=50) → 0.577
  KANSER : TEMİZLENMİŞ + KNNImputer (tüm özellikler)       → 0.756
  PAH    : ORİJİNAL   + ANOVA+RFE+EMBEDDED (k=30 her biri) → 0.518
  CFTR   : TEMİZLENMİŞ + AL_Sifir_EK_Medyan (tüm özellik) → 0.673

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))
from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, split_data,
)

ROOT       = Path("/Users/mahmudselmansahin/Teknofest")
DATA_TEMIZ = ROOT / "VERİLER" / "TEMİZLENMİŞ"
DATA_ORI   = ROOT / "VERİLER" / "ORİJİNAL"
OUT_DIR    = MODELS_ROOT / "CatBoost" / "FINAL"

AL_PREFIX = "AL_"
EK_PREFIX = "EK_"
EK_EXTRAS  = {"BLOSUM62", "DELTA_HIDRO", "SNV_PATHWAY_COUNT"}

# Özellik seçimi k değerleri (FS script ile eşleştirildi)
N_FS_TEMIZ = 50  # run_feature_selection_temiz.py → N_FEATURES=50
N_FS_ORI   = 30  # run_feature_selection.py       → N_FEATURES=30

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
# Yükleme & genel ön işleme
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


# ---------------------------------------------------------------------------
# İmputation yöntemleri
# ---------------------------------------------------------------------------

def impute_eksik_korundu(x: pd.DataFrame) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Özellik seçimi yardımcıları
# ---------------------------------------------------------------------------

def to_numeric_array_for_fs(x: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """sklearn için: kat→kod, NaN→medyan."""
    X = x.copy()
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
            X[c] = pd.Categorical(X[c].fillna("__MISSING__").astype(str)).codes.astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in X.columns:
        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(0.0 if pd.isna(med) else med)
    return X.values, list(X.columns)


def union_features(*lists: list[str]) -> list[str]:
    seen, result = set(), []
    for lst in lists:
        for f in lst:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


def compute_filter_anova(X: np.ndarray, y: np.ndarray, names: list[str], k: int) -> list[str]:
    k = min(k, X.shape[1])
    mask = SelectKBest(f_classif, k=k).fit(X, y).get_support()
    return [names[i] for i, m in enumerate(mask) if m]


def compute_wrapper_rfe(X: np.ndarray, y: np.ndarray, names: list[str], k: int) -> list[str]:
    proxy = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    step = max(1, X.shape[1] // 15)
    rfe = RFE(proxy, n_features_to_select=min(k, X.shape[1]), step=step)
    rfe.fit(X, y)
    return [names[i] for i, s in enumerate(rfe.support_) if s]


def compute_embedded_cb(x: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    cols = list(x.columns)
    X_enc = x.copy()
    cat_idx = cat_indices(X_enc)
    for c in X_enc.columns:
        if pd.api.types.is_object_dtype(X_enc[c]) or pd.api.types.is_string_dtype(X_enc[c]):
            X_enc[c] = X_enc[c].fillna("__MISSING__").astype(str)
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )
    model.fit(X_enc, y, cat_features=cat_idx)
    top = np.argsort(model.get_feature_importance())[::-1][: min(k, len(cols))]
    return [cols[i] for i in top]


def fit_catboost_fs(x: pd.DataFrame, selected: list[str], y: pd.Series):
    """Seçilmiş özelliklerle CatBoost eğitir."""
    X_sel = x[selected].copy()
    for c in X_sel.columns:
        if pd.api.types.is_object_dtype(X_sel[c]) or pd.api.types.is_string_dtype(X_sel[c]):
            X_sel[c] = X_sel[c].fillna("__MISSING__").astype(str)
    cat_idx = cat_indices(X_sel)
    x_tr, x_te, y_tr, y_te = split_data(X_sel, y)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(x_tr, y_tr, cat_features=cat_idx)
    y_pred = model.predict(x_te).astype(int)
    y_prob = model.predict_proba(x_te)[:, 1]
    return y_te, y_pred, y_prob


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
# Panel akışları
# ---------------------------------------------------------------------------

def run_master() -> dict:
    df = load_raw("MASTER", temiz=True)
    x, y = get_xy(df)
    X_num, names = to_numeric_array_for_fs(x)
    wr = compute_wrapper_rfe(X_num, y.values, names, k=N_FS_TEMIZ)
    print(f"  MASTER   [TEMİZLENMİŞ+WRAPPER_RFE(k={N_FS_TEMIZ}): {len(wr)} özellik]")
    y_te, y_pred, y_prob = fit_catboost_fs(x, wr, y)
    m = compute(y_te.values, y_pred, y_prob)
    print(f"           MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    return {"panel": "MASTER", "config": f"TEMİZLENMİŞ+WRAPPER_RFE(k={len(wr)})", "n_feat": len(wr), **m}


def run_kanser() -> dict:
    df = load_raw("KANSER", temiz=True)
    x, y = get_xy(df)
    x = impute_knn(x)
    cat_idx = cat_indices(x)
    x_tr, x_te, y_tr, y_te = split_data(x, y)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(x_tr, y_tr, cat_features=cat_idx)
    y_pred = model.predict(x_te).astype(int)
    y_prob = model.predict_proba(x_te)[:, 1]
    m = compute(y_te.values, y_pred, y_prob)
    print(f"  KANSER   [TEMİZLENMİŞ+KNN(tüm özellikler)]")
    print(f"           MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    return {"panel": "KANSER", "config": "TEMİZLENMİŞ+KNN", "n_feat": x.shape[1], **m}


def run_pah() -> dict:
    df = load_raw("PAH", temiz=False)
    x, y = get_xy(df)
    X_num, names = to_numeric_array_for_fs(x)
    k = N_FS_ORI
    fa = compute_filter_anova(X_num, y.values, names, k=k)
    wr = compute_wrapper_rfe(X_num, y.values, names, k=k)
    em = compute_embedded_cb(x, y, k=k)
    sel = union_features(fa, wr, em)
    print(f"  PAH      [ORİJİNAL+ANOVA+RFE+EMBEDDED(k={k}): {len(sel)} özellik]")
    y_te, y_pred, y_prob = fit_catboost_fs(x, sel, y)
    m = compute(y_te.values, y_pred, y_prob)
    print(f"           MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    return {"panel": "PAH", "config": f"ORİJİNAL+ANOVA+RFE+EMBEDDED(k={len(sel)})", "n_feat": len(sel), **m}


def run_cftr() -> dict:
    df = load_raw("CFTR", temiz=True)
    x, y = get_xy(df)
    x = impute_al_sifir_ek_medyan(x)
    cat_idx = cat_indices(x)
    x_tr, x_te, y_tr, y_te = split_data(x, y)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(x_tr, y_tr, cat_features=cat_idx)
    y_pred = model.predict(x_te).astype(int)
    y_prob = model.predict_proba(x_te)[:, 1]
    m = compute(y_te.values, y_pred, y_prob)
    print(f"  CFTR     [TEMİZLENMİŞ+AL_Sifir_EK_Medyan]")
    print(f"           MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    return {"panel": "CFTR", "config": "TEMİZLENMİŞ+AL_Sifir_EK_Medyan", "n_feat": x.shape[1], **m}


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("CatBoost Final Model\n")
    rows = [run_master(), run_kanser(), run_pah(), run_cftr()]
    df = pd.DataFrame(rows)
    out = OUT_DIR / "CatBoost_final_results.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()

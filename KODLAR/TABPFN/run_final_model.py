"""
TabPFN — Final Model (Ham Veriden En İyi Konfigürasyon)

Her panel için ham veriden başlayıp en iyi konfigürasyonu çalıştırır.

En iyi konfigürasyonlar:
  MASTER : TEMİZLENMİŞ + OrdinalEncode + MinMaxScaler + Medyan → TabPFN(n=1)
  KANSER : TEMİZLENMİŞ + OrdinalEncode + AL_Sifir_EK_Medyan
           + FILTER_ANOVA∪WRAPPER_RFE∪EMBEDDED (k=30 her biri) → TabPFN(n=1)
  PAH    : TEMİZLENMİŞ + OrdinalEncode + Medyan → TabPFN(n=1)
  CFTR   : TEMİZLENMİŞ + OrdinalEncode + AL_Sifir_EK_Medyan
           + QuantileTransformer + BaggingClassifier(TabPFN, n=10, max_feat=0.3)

Çalıştırma:
  python run_final_model.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ABLASYON_CSV_SCRIPTLERI"))
from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, split_data,
)
from ablation_csv_factory import build_category_levels, apply_ordinal_encoding

ROOT       = Path("/Users/mahmudselmansahin/Teknofest")
DATA_TEMIZ = ROOT / "VERİLER" / "TEMİZLENMİŞ"
OUT_DIR    = MODELS_ROOT / "TabPFN" / "FINAL"

AL_PREFIX = "AL_"
EK_PREFIX = "EK_"
EK_EXTRAS  = {"BLOSUM62", "DELTA_HIDRO", "SNV_PATHWAY_COUNT"}
N_FS       = 30  # her feature selection yöntemi başına k

TABPFN_BASE = dict(
    n_estimators=1,
    random_state=RANDOM_STATE,
    ignore_pretraining_limits=True,
    device="cpu",
    show_progress_bar=False,
    fit_mode="low_memory",
)


# ---------------------------------------------------------------------------
# Yükleme & genel ön işleme
# ---------------------------------------------------------------------------

PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]


def load_raw(panel: str) -> pd.DataFrame:
    return pd.read_csv(DATA_TEMIZ / f"YARISMA_TRAIN_{panel}_temiz.csv")


def get_xy(df: pd.DataFrame):
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[LABEL_COL] + ([ID_COL] if ID_COL in df.columns else []))
    return x, y


def build_shared_encoder() -> dict:
    """Tüm panellerin ham verisinden ortak kategori seviyelerini oluşturur."""
    all_dfs = {p: pd.read_csv(DATA_TEMIZ / f"YARISMA_TRAIN_{p}_temiz.csv")
               for p in PANELS}
    return build_category_levels(all_dfs)


def factory_ordinal_encode(df: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    """Factory'nin tutarlı ordinal encoding'ini uygular."""
    encoded, _ = apply_ordinal_encoding(df, cat_levels)
    # ID ve Label sütunlarını koru; diğerleri artık numerik
    return encoded


def fill_medyan(x: pd.DataFrame) -> pd.DataFrame:
    return x.fillna(x.median())


def fill_al_sifir_ek_medyan(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for c in x.columns:
        if c.startswith(AL_PREFIX):
            x[c] = x[c].fillna(0.0)
        elif c.startswith(EK_PREFIX) or c in EK_EXTRAS:
            x[c] = x[c].fillna(x[c].median())
        else:
            x[c] = x[c].fillna(0.0)
    return x


def to_float32(x: pd.DataFrame) -> np.ndarray:
    return x.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature selection (KANSER için)
# ---------------------------------------------------------------------------

def union_features(*lists: list[str]) -> list[str]:
    seen, result = set(), []
    for lst in lists:
        for f in lst:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


def select_anova(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    k = min(N_FS, X.shape[1])
    mask = SelectKBest(f_classif, k=k).fit(X, y).get_support()
    return [names[i] for i, m in enumerate(mask) if m]


def select_rfe(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    proxy = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    step = max(1, X.shape[1] // 15)
    rfe  = RFE(proxy, n_features_to_select=min(N_FS, X.shape[1]), step=step)
    rfe.fit(X, y)
    return [names[i] for i, s in enumerate(rfe.support_) if s]


def select_embedded(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X, y)
    top = np.argsort(rf.feature_importances_)[::-1][: min(N_FS, len(names))]
    return [names[i] for i in top]


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

def run_master(cat_levels: dict) -> tuple:
    df = load_raw("MASTER")
    df = factory_ordinal_encode(df, cat_levels)
    x, y = get_xy(df)
    x = fill_medyan(x)
    arr = MinMaxScaler().fit_transform(to_float32(x)).astype(np.float32)
    xdf = pd.DataFrame(arr, columns=x.columns)
    x_tr, x_te, y_tr, y_te = split_data(xdf, y)
    model = TabPFNClassifier(**TABPFN_BASE)
    model.fit(x_tr, y_tr)
    return ("MASTER", "Medyan+MinMaxScaler", x.shape[1],
            y_te, model.predict(x_te).astype(int), model.predict_proba(x_te)[:, 1])


def run_kanser(cat_levels: dict) -> tuple:
    df = load_raw("KANSER")
    df = factory_ordinal_encode(df, cat_levels)
    x, y = get_xy(df)
    x = fill_medyan(x)
    X = to_float32(x)
    names = list(x.columns)

    fa  = select_anova(X, y.values, names)
    wr  = select_rfe(X, y.values, names)
    em  = select_embedded(X, y.values, names)
    sel = union_features(fa, wr, em)
    idx = [names.index(f) for f in sel]
    print(f"    KANSER feature selection: {len(sel)} feature seçildi")

    X_sel = X[:, idx].astype(np.float32)
    x_tr, x_te, y_tr, y_te = split_data(pd.DataFrame(X_sel, columns=sel), y)
    model = TabPFNClassifier(**TABPFN_BASE)
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te).astype(int)
    y_prob = model.predict_proba(x_te)[:, 1]
    return ("KANSER", f"Medyan+FS(ANOVA+RFE+EMBEDDED,k={len(sel)})",
            len(sel), y_te, y_pred, y_prob)


def run_pah(cat_levels: dict) -> tuple:
    df = load_raw("PAH")
    df = factory_ordinal_encode(df, cat_levels)
    x, y = get_xy(df)
    x = fill_medyan(x)
    xdf = pd.DataFrame(to_float32(x), columns=x.columns)
    x_tr, x_te, y_tr, y_te = split_data(xdf, y)
    model = TabPFNClassifier(**TABPFN_BASE)
    model.fit(x_tr, y_tr)
    return ("PAH", "Medyan", x.shape[1],
            y_te, model.predict(x_te).astype(int), model.predict_proba(x_te)[:, 1])


def run_cftr(cat_levels: dict) -> tuple:
    df = load_raw("CFTR")
    df = factory_ordinal_encode(df, cat_levels)
    x, y = get_xy(df)
    x = fill_al_sifir_ek_medyan(x)
    X = to_float32(x)
    splits = split_data(pd.DataFrame(X, columns=x.columns), y)
    x_tr_arr, x_te_arr, y_tr, y_te = splits[0].values, splits[1].values, splits[2], splits[3]

    qt = QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)
    x_tr_qt = qt.fit_transform(x_tr_arr).astype(np.float32)
    x_te_qt = qt.transform(x_te_arr).astype(np.float32)

    bag = BaggingClassifier(
        estimator=TabPFNClassifier(**TABPFN_BASE),
        n_estimators=10,
        max_features=0.3,
        bootstrap=True,
        bootstrap_features=False,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    bag.fit(x_tr_qt, y_tr)
    y_pred = bag.predict(x_te_qt).astype(int)
    y_prob = bag.predict_proba(x_te_qt)[:, 1]
    return ("CFTR", "AL_Sifir_EK_Medyan+QT+Bagging(n=10,mf=0.3)", x.shape[1],
            y_te, y_pred, y_prob)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("TabPFN Final Model\n")
    print("Ortak kategori seviyeleri oluşturuluyor...")
    cat_levels = build_shared_encoder()
    rows = []

    for panel_fn in [
        lambda: run_master(cat_levels),
        lambda: run_kanser(cat_levels),
        lambda: run_pah(cat_levels),
        lambda: run_cftr(cat_levels),
    ]:
        panel, config, n_feat, y_te, y_pred, y_prob = panel_fn()
        m = compute(y_te.values, y_pred, y_prob)
        print(f"  {panel:8s} [{config[:45]}]")
        print(f"           MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        rows.append({"panel": panel, "config": config, "n_feat": n_feat, **m})

    df = pd.DataFrame(rows)
    out = OUT_DIR / "TabPFN_final_results.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()

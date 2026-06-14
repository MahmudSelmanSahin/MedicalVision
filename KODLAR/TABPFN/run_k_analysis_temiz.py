"""
TabPFN — k Analizi (TEMİZLENMİŞ veri)
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : MODELLER/TabPFN/TEMİZLENMİŞ/ABLASYON/k_analysis_summary.csv

String sütunlar (CAT_*, AA_*) ordinal encoding ile sayısala çevrilir.

Çalıştırma (TABPFN klasöründen):
  python run_k_analysis_temiz.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from tabpfn import TabPFNClassifier

from model_ablation_common import (
    ID_COL,
    LABEL_COL,
    MODELS_ROOT,
    RANDOM_STATE,
    ROOT,
    compute_metrics,
    split_data,
)

MODEL_NAME = "TabPFN"
VERI_TURU  = "TEMİZLENMİŞ"
DATA_DIR   = ROOT / "VERİLER" / VERI_TURU
OUT_BASE   = MODELS_ROOT / MODEL_NAME / VERI_TURU
ABLASYON   = OUT_BASE / "ABLASYON"
ABLASYON.mkdir(parents=True, exist_ok=True)

K_VALUES = [10, 20, 30, 50, 100]
PANELS   = ["MASTER", "KANSER", "PAH", "CFTR"]

ATOMIC_METHODS = ["FILTER_ANOVA", "FILTER_MUTUAL_INFO", "WRAPPER_RFE", "EMBEDDED"]
COMBO_METHODS  = [
    "FILTER_ANOVA + WRAPPER_RFE",
    "FILTER_ANOVA + EMBEDDED",
    "FILTER_MUTUAL_INFO + WRAPPER_RFE",
    "FILTER_MUTUAL_INFO + EMBEDDED",
    "WRAPPER_RFE + EMBEDDED",
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED",
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED",
]
ALL_METHODS = ATOMIC_METHODS + COMBO_METHODS


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_panels() -> dict[str, pd.DataFrame]:
    panels = {}
    for panel in PANELS:
        p = DATA_DIR / f"YARISMA_TRAIN_{panel}_temiz.csv"
        if p.exists():
            panels[panel] = pd.read_csv(p)
        else:
            print(f"  UYARI: {p} bulunamadı, atlanıyor.")
    return panels


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def to_numeric_array(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """String sütunları ordinal encode et; NaN kaldıysa medyanla doldur."""
    X = df[cols].copy()
    for col in X.columns:
        if X[col].dtype == object or pd.api.types.is_string_dtype(X[col]):
            X[col] = pd.Categorical(X[col].fillna("__MISSING__").astype(str)).codes.astype(float)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if pd.isna(med) else med)
    return X.values.astype(np.float32), list(X.columns)


def union_features(*lists: list[str]) -> list[str]:
    seen, result = set(), []
    for lst in lists:
        for f in lst:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


# ---------------------------------------------------------------------------
# Skor hesaplama — panel başına bir kez
# ---------------------------------------------------------------------------

def compute_scores(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    fa_scores, _ = f_classif(X, y)
    fa_scores = np.nan_to_num(fa_scores, nan=0.0)

    fmi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    proxy = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    step = max(1, X.shape[1] // 15)
    rfe = RFE(proxy, n_features_to_select=1, step=step)
    rfe.fit(X, y)
    wr_ranking = rfe.ranking_

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X, y)
    em_scores = rf.feature_importances_

    return {"fa": fa_scores, "fmi": fmi_scores, "wr": wr_ranking, "em": em_scores}


def top_k_from_scores(scores: dict, names: list[str], k: int) -> dict[str, list[str]]:
    k = min(k, len(names))
    fa  = [names[i] for i in np.argsort(scores["fa"])[::-1][:k]]
    fmi = [names[i] for i in np.argsort(scores["fmi"])[::-1][:k]]
    wr  = [names[i] for i in np.argsort(scores["wr"])[:k]]
    em  = [names[i] for i in np.argsort(scores["em"])[::-1][:k]]
    return {
        "FILTER_ANOVA":      fa,
        "FILTER_MUTUAL_INFO": fmi,
        "WRAPPER_RFE":       wr,
        "EMBEDDED":          em,
        "FILTER_ANOVA + WRAPPER_RFE":                  union_features(fa, wr),
        "FILTER_ANOVA + EMBEDDED":                     union_features(fa, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE":            union_features(fmi, wr),
        "FILTER_MUTUAL_INFO + EMBEDDED":               union_features(fmi, em),
        "WRAPPER_RFE + EMBEDDED":                      union_features(wr, em),
        "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       union_features(fa, wr, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": union_features(fmi, wr, em),
    }


# ---------------------------------------------------------------------------
# Model eğitimi
# ---------------------------------------------------------------------------

def fit_tabpfn(X_encoded: np.ndarray, selected_idx: list[int], y: pd.Series):
    X_sel = X_encoded[:, selected_idx].astype(np.float32)
    X_tr, X_te, y_tr, y_te = split_data(pd.DataFrame(X_sel), y)
    model = TabPFNClassifier(
        n_estimators=1, random_state=RANDOM_STATE,
        ignore_pretraining_limits=True, device="cpu",
        show_progress_bar=False, fit_mode="low_memory",
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te).astype(int)
    y_prob = model.predict_proba(X_te)[:, 1]
    return y_te, y_pred, y_prob, len(y_tr), len(y_te)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    panel_dfs = load_panels()
    if not panel_dfs:
        print("HATA: TEMİZLENMİŞ klasöründe veri bulunamadı.")
        return

    print("Skorlar hesaplanıyor (panel başına bir kez)...")
    panel_cache: dict[str, dict] = {}
    for panel, df in panel_dfs.items():
        print(f"  {panel}")
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        X, names = to_numeric_array(df, cols)
        scores = compute_scores(X, y.values, names)
        panel_cache[panel] = {"X": X, "names": names, "y": y, "scores": scores}

    rows, failures = [], []

    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        for panel, cache in panel_cache.items():
            method_feats = top_k_from_scores(cache["scores"], cache["names"], k)
            for method in ALL_METHODS:
                selected = method_feats[method]
                names = cache["names"]
                sel_idx = [names.index(f) for f in selected if f in names]
                try:
                    y_te, y_pred, y_prob, n_tr, n_te = fit_tabpfn(
                        cache["X"], sel_idx, cache["y"]
                    )
                    rows.append({
                        "model": MODEL_NAME, "k": k, "method": method,
                        "panel": panel, "veri_turu": VERI_TURU,
                        "n_rows": len(cache["y"]),
                        "n_selected_features": len(sel_idx),
                        "train_rows": n_tr, "test_rows": n_te,
                        **compute_metrics(y_te, y_pred, y_prob),
                    })
                    print(f"  {panel} | {method} | k={k} → MCC={rows[-1]['mcc']:.3f}")
                except Exception as exc:
                    failures.append({
                        "k": k, "method": method, "panel": panel,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    print(f"  ! HATA {panel}/{method}: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(ABLASYON / "k_analysis_summary.csv", index=False)
    print(f"\n✓ Özet: {ABLASYON}/k_analysis_summary.csv  ({len(rows)} satır)")

    if not out.empty:
        best = out.sort_values("mcc", ascending=False).groupby("panel", as_index=False).head(1)
        best.to_csv(ABLASYON / f"{MODEL_NAME}_best_by_panel.csv", index=False)
        print(f"✓ En iyi: {ABLASYON}/{MODEL_NAME}_best_by_panel.csv")

    (ABLASYON / "k_analysis_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Tamamlandı. Başarısız: {len(failures)}")


if __name__ == "__main__":
    main()

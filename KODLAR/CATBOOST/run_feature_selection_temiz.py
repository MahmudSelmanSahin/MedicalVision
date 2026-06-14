"""
CatBoost — Özellik Seçimi (TEMİZLENMİŞ veri)
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : MODELLER/CatBoost/TEMİZLENMİŞ/{YÖNTEM}/SONUCLAR|GRAFIKLER

Eski script (run_feature_selection.py) ABLASYON_MODEL_BAZLI verisini okur,
bu script TEMİZLENMİŞ verisini doğrudan okur.

Çalıştırma (CATBOOST klasöründen):
  python run_feature_selection_temiz.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif

from model_ablation_common import (
    ID_COL,
    LABEL_COL,
    MODELS_ROOT,
    RANDOM_STATE,
    ROOT,
    compute_metrics,
    save_confusion_matrix_png,
    save_precision_recall_png,
    split_data,
)

MODEL_NAME  = "CatBoost"
VERI_TURU   = "TEMİZLENMİŞ"
DATA_DIR    = ROOT / "VERİLER" / VERI_TURU
OUT_BASE    = MODELS_ROOT / MODEL_NAME / VERI_TURU
ABLASYON    = OUT_BASE / "ABLASYON"
N_FEATURES  = 50
PANELS      = ["MASTER", "KANSER", "PAH", "CFTR"]

METHOD_PATHS: dict[str, tuple[str, str]] = {
    "FILTER_ANOVA":                                ("FILTER",                      "ANOVA"),
    "FILTER_MUTUAL_INFO":                          ("FILTER",                      "MUTUAL_INFO"),
    "WRAPPER_RFE":                                 ("WRAPPER",                     "RFE"),
    "EMBEDDED":                                    ("EMBEDDED",                    "CATBOOST_IMPORTANCE"),
    "FILTER_ANOVA + WRAPPER_RFE":                  ("FILTER + WRAPPER",            "ANOVA + RFE"),
    "FILTER_ANOVA + EMBEDDED":                     ("FILTER + EMBEDDED",           "ANOVA + CATBOOST_IMPORTANCE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE":            ("FILTER + WRAPPER",            "MUTUAL_INFO + RFE"),
    "FILTER_MUTUAL_INFO + EMBEDDED":               ("FILTER + EMBEDDED",           "MUTUAL_INFO + CATBOOST_IMPORTANCE"),
    "WRAPPER_RFE + EMBEDDED":                      ("WRAPPER + EMBEDDED",          "RFE + CATBOOST_IMPORTANCE"),
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       ("FILTER + WRAPPER + EMBEDDED", "ANOVA + RFE + CATBOOST_IMPORTANCE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": ("FILTER + WRAPPER + EMBEDDED", "MUTUAL_INFO + RFE + CATBOOST_IMPORTANCE"),
}
METHODS = list(METHOD_PATHS.keys())


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
    return X.values, list(X.columns)


def cat_indices(df: pd.DataFrame, cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols)
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]


def union_features(*lists: list[str]) -> list[str]:
    seen, result = set(), []
    for lst in lists:
        for f in lst:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


# ---------------------------------------------------------------------------
# Atomik özellik seçimi — panel başına bir kez
# ---------------------------------------------------------------------------

def compute_filter_anova(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    k = min(N_FEATURES, X.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(X, y)
    return [names[i] for i, m in enumerate(sel.get_support()) if m]


def compute_filter_mi(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    k = min(N_FEATURES, X.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k).fit(X, y)
    return [names[i] for i, m in enumerate(sel.get_support()) if m]


def compute_wrapper_rfe(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    proxy = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    step = max(1, X.shape[1] // 15)
    rfe = RFE(proxy, n_features_to_select=min(N_FEATURES, X.shape[1]), step=step)
    rfe.fit(X, y)
    return [names[i] for i, s in enumerate(rfe.support_) if s]


def compute_embedded(df: pd.DataFrame, cols: list[str], y: pd.Series) -> list[str]:
    X = df[cols].copy()
    cat_idx = cat_indices(df, cols)
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].fillna("__MISSING__").astype(str)
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )
    model.fit(X, y, cat_features=cat_idx)
    top = np.argsort(model.get_feature_importance())[::-1][: min(N_FEATURES, len(cols))]
    return [cols[i] for i in top]


def resolve_method(
    method: str,
    fa: list[str], fmi: list[str], wr: list[str], em: list[str],
) -> list[str]:
    return {
        "FILTER_ANOVA": fa,
        "FILTER_MUTUAL_INFO": fmi,
        "WRAPPER_RFE": wr,
        "EMBEDDED": em,
        "FILTER_ANOVA + WRAPPER_RFE":                  union_features(fa, wr),
        "FILTER_ANOVA + EMBEDDED":                     union_features(fa, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE":            union_features(fmi, wr),
        "FILTER_MUTUAL_INFO + EMBEDDED":               union_features(fmi, em),
        "WRAPPER_RFE + EMBEDDED":                      union_features(wr, em),
        "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       union_features(fa, wr, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": union_features(fmi, wr, em),
    }[method]


# ---------------------------------------------------------------------------
# Model eğitimi
# ---------------------------------------------------------------------------

def fit_catboost(df: pd.DataFrame, selected: list[str], y: pd.Series):
    X = df[selected].copy()
    cat_idx = cat_indices(df, selected)
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].fillna("__MISSING__").astype(str)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = CatBoostClassifier(
        iterations=250, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )
    model.fit(X_tr, y_tr, cat_features=cat_idx)
    y_pred = model.predict(X_te).astype(int)
    y_prob = model.predict_proba(X_te)[:, 1]
    return y_te, y_pred, y_prob, len(y_tr), len(y_te)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def method_key(m: str) -> str:
    return m.replace(" ", "_").replace("+", "PLUS")


def main() -> None:
    panel_dfs = load_panels()
    if not panel_dfs:
        print("HATA: TEMİZLENMİŞ klasöründe veri bulunamadı.")
        return

    print("Özellik seçimi hesaplanıyor...")
    panel_cache: dict[str, dict] = {}
    for panel, df in panel_dfs.items():
        print(f"  {panel}")
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        X_num, names = to_numeric_array(df, cols)

        fa  = compute_filter_anova(X_num, y.values, names)
        fmi = compute_filter_mi(X_num, y.values, names)
        wr  = compute_wrapper_rfe(X_num, y.values, names)
        em  = compute_embedded(df, cols, y)
        print(f"    FA:{len(fa)}  MI:{len(fmi)}  RFE:{len(wr)}  EMB:{len(em)}")

        panel_cache[panel] = {"df": df, "y": y, "fa": fa, "fmi": fmi, "wr": wr, "em": em}

    all_failures = []

    for method in METHODS:
        print(f"\n{'='*60}\nYöntem: {method}")
        category, sub_method = METHOD_PATHS[method]
        method_dir = OUT_BASE / category / sub_method
        sonuclar  = method_dir / "SONUCLAR"
        grafikler = method_dir / "GRAFIKLER"
        sonuclar.mkdir(parents=True, exist_ok=True)
        grafikler.mkdir(parents=True, exist_ok=True)

        rows = []
        for panel, cache in panel_cache.items():
            try:
                selected = resolve_method(
                    method, cache["fa"], cache["fmi"], cache["wr"], cache["em"]
                )
                print(f"  {panel}: {len(selected)} özellik")

                y_te, y_pred, y_prob, n_tr, n_te = fit_catboost(
                    cache["df"], selected, cache["y"]
                )
                rows.append({
                    "model": MODEL_NAME, "method": method, "panel": panel,
                    "veri_turu": VERI_TURU, "n_rows": len(cache["y"]),
                    "n_selected_features": len(selected),
                    "train_rows": n_tr, "test_rows": n_te,
                    **compute_metrics(y_te, y_pred, y_prob),
                })
                base = f"{panel}_{method_key(method)}"
                save_confusion_matrix_png(
                    y_te, y_pred, grafikler / f"{base}_confusion_matrix.png",
                    f"{MODEL_NAME} | {method} | {panel} | {VERI_TURU}",
                )
                save_precision_recall_png(
                    y_te, y_prob, grafikler / f"{base}_precision_recall_curve.png",
                    f"{MODEL_NAME} PR | {method} | {panel} | {VERI_TURU}",
                )
            except Exception as exc:
                all_failures.append({
                    "method": method, "panel": panel,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"    ! HATA: {exc}")

        if rows:
            pd.DataFrame(rows).to_csv(
                sonuclar / f"{MODEL_NAME}_{method_key(method)}_metrics.csv", index=False
            )
            print(f"  ✓ → {category}/{sub_method}/SONUCLAR/")

    all_rows = []
    for m in METHODS:
        cat, sub = METHOD_PATHS[m]
        p = OUT_BASE / cat / sub / "SONUCLAR" / f"{MODEL_NAME}_{method_key(m)}_metrics.csv"
        if p.exists():
            all_rows.append(pd.read_csv(p))
    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        summary.to_csv(OUT_BASE / "feature_selection_summary.csv", index=False)
        print(f"\n✓ Özet: {OUT_BASE}/feature_selection_summary.csv")

        best = summary.sort_values("mcc", ascending=False).groupby("panel", as_index=False).head(1)
        best.to_csv(ABLASYON / f"{MODEL_NAME}_fs_best_by_panel.csv", index=False)
        print(f"✓ En iyi: {ABLASYON}/{MODEL_NAME}_fs_best_by_panel.csv")

    (OUT_BASE / "feature_selection_failures.json").write_text(
        json.dumps(all_failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Tamamlandı. Başarısız: {len(all_failures)}")


if __name__ == "__main__":
    main()

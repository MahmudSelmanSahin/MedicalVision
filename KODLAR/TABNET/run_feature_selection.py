"""
TabNet — Adım 2: Özellik Seçimi (ORİJİNAL 353-sütun veri)
Giriş : MODELLER/TabNet/ORİJİNAL_VERİ/ABLASYON/TabNet_best_by_panel.csv
        + VERİLER/ABLASYON_MODEL_BAZLI/{scenario}/{PANEL}_*.csv
Çıkış : MODELLER/TabNet/ORİJİNAL_VERİ/{YÖNTEM}/SONUCLAR|GRAFIKLER

CatBoost/run_feature_selection.py'nin TabNet karşılığıdır.
Veriyi best_by_panel'den seçilen senaryodan okur (ÖNCE run_ablation.py).

Gereksinim: pip install pytorch-tabnet torch

Çalıştırma (TABNET klasöründen):
  python run_feature_selection.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif

from model_ablation_common import (
    ID_COL,
    LABEL_COL,
    RANDOM_STATE,
    compute_metrics,
    save_confusion_matrix_png,
    save_precision_recall_png,
    split_data,
)

MODEL_NAME = "TabNet"
N_FEATURES = 30
MAX_EPOCHS = 100

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "VERİLER" / "ABLASYON_MODEL_BAZLI"
OUT_BASE = ROOT / "MODELLER" / MODEL_NAME / "ORİJİNAL_VERİ"
ABLASYON_DIR = OUT_BASE / "ABLASYON"

METHOD_PATHS: dict[str, tuple[str, str]] = {
    "FILTER_ANOVA":                                ("FILTER",                      "ANOVA"),
    "FILTER_MUTUAL_INFO":                          ("FILTER",                      "MUTUAL_INFO"),
    "WRAPPER_RFE":                                 ("WRAPPER",                     "RFE"),
    "EMBEDDED":                                    ("EMBEDDED",                    "TABNET_IMPORTANCE"),
    "FILTER_ANOVA + WRAPPER_RFE":                  ("FILTER + WRAPPER",            "ANOVA + RFE"),
    "FILTER_ANOVA + EMBEDDED":                     ("FILTER + EMBEDDED",           "ANOVA + TABNET_IMPORTANCE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE":            ("FILTER + WRAPPER",            "MUTUAL_INFO + RFE"),
    "FILTER_MUTUAL_INFO + EMBEDDED":               ("FILTER + EMBEDDED",           "MUTUAL_INFO + TABNET_IMPORTANCE"),
    "WRAPPER_RFE + EMBEDDED":                      ("WRAPPER + EMBEDDED",          "RFE + TABNET_IMPORTANCE"),
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       ("FILTER + WRAPPER + EMBEDDED", "ANOVA + RFE + TABNET_IMPORTANCE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": ("FILTER + WRAPPER + EMBEDDED", "MUTUAL_INFO + RFE + TABNET_IMPORTANCE"),
}
METHODS = list(METHOD_PATHS.keys())


def load_best_scenarios() -> dict[str, tuple[str, Path]]:
    best = pd.read_csv(ABLASYON_DIR / f"{MODEL_NAME}_best_by_panel.csv")
    result = {}
    for _, row in best.iterrows():
        panel, scenario = row["panel"], row["scenario"]
        csvs = sorted((DATA_ROOT / scenario).glob(f"{panel}_*.csv"))
        if csvs:
            result[panel] = (scenario, csvs[0])
    return result


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def to_numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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
    return X


def union_features(*lists: list[str]) -> list[str]:
    seen, result = set(), []
    for lst in lists:
        for f in lst:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


def compute_filter_anova(X, y, names):
    k = min(N_FEATURES, X.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(X, y)
    return [names[i] for i, m in enumerate(sel.get_support()) if m]


def compute_filter_mi(X, y, names):
    k = min(N_FEATURES, X.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k).fit(X, y)
    return [names[i] for i, m in enumerate(sel.get_support()) if m]


def compute_wrapper_rfe(X, y, names):
    proxy = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    step = max(1, X.shape[1] // 15)
    rfe = RFE(proxy, n_features_to_select=min(N_FEATURES, X.shape[1]), step=step)
    rfe.fit(X, y)
    return [names[i] for i, s in enumerate(rfe.support_) if s]


def _tabnet_batch(n: int) -> tuple[int, int]:
    bs = min(256, max(16, n))
    return bs, max(8, bs // 2)


def _fit_tabnet_model(X_np: np.ndarray, y_np: np.ndarray) -> TabNetClassifier:
    bs, vbs = _tabnet_batch(len(X_np))
    model = TabNetClassifier(seed=RANDOM_STATE, verbose=0)
    model.fit(
        X_np.astype(np.float32), y_np,
        max_epochs=MAX_EPOCHS, patience=0,
        batch_size=bs, virtual_batch_size=vbs, weights=1,
    )
    return model


def compute_embedded(Xdf: pd.DataFrame, y: pd.Series) -> list[str]:
    model = _fit_tabnet_model(Xdf.values, y.values)
    cols = list(Xdf.columns)
    top = np.argsort(model.feature_importances_)[::-1][: min(N_FEATURES, len(cols))]
    return [cols[i] for i in top]


def resolve_method(method, fa, fmi, wr, em):
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


def fit_tabnet(Xdf: pd.DataFrame, selected: list[str], y: pd.Series):
    X = Xdf[selected]
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = _fit_tabnet_model(X_tr.values, y_tr.values)
    X_te_np = X_te.values.astype(np.float32)
    y_pred = model.predict(X_te_np).astype(int)
    y_prob = model.predict_proba(X_te_np)[:, 1]
    return y_te, y_pred, y_prob, len(y_tr), len(y_te)


def method_key(m: str) -> str:
    return m.replace(" ", "_").replace("+", "PLUS")


def main() -> None:
    best_scenarios = load_best_scenarios()
    if not best_scenarios:
        print("HATA: best_by_panel.csv bulunamadı. Önce run_ablation.py çalıştırın.")
        return

    print("Özellik seçimi hesaplanıyor...")
    panel_cache: dict[str, dict] = {}
    for panel, (scenario, csv_path) in best_scenarios.items():
        print(f"  {panel}  ({scenario})")
        df = pd.read_csv(csv_path)
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        Xdf = to_numeric_df(df, cols)
        names = list(Xdf.columns)
        X_num = Xdf.values

        fa = compute_filter_anova(X_num, y.values, names)
        fmi = compute_filter_mi(X_num, y.values, names)
        wr = compute_wrapper_rfe(X_num, y.values, names)
        em = compute_embedded(Xdf, y)
        print(f"    FA:{len(fa)}  MI:{len(fmi)}  RFE:{len(wr)}  EMB:{len(em)}")

        panel_cache[panel] = {
            "Xdf": Xdf, "y": y, "scenario": scenario,
            "fa": fa, "fmi": fmi, "wr": wr, "em": em,
        }

    all_failures = []

    for method in METHODS:
        print(f"\n{'='*60}\nYöntem: {method}")
        category, sub_method = METHOD_PATHS[method]
        method_dir = OUT_BASE / category / sub_method
        sonuclar = method_dir / "SONUCLAR"
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

                y_te, y_pred, y_prob, n_tr, n_te = fit_tabnet(
                    cache["Xdf"], selected, cache["y"]
                )
                rows.append({
                    "model": MODEL_NAME, "method": method, "panel": panel,
                    "scenario": cache["scenario"], "n_rows": len(cache["y"]),
                    "n_selected_features": len(selected),
                    "train_rows": n_tr, "test_rows": n_te,
                    **compute_metrics(y_te, y_pred, y_prob),
                })
                base = f"{panel}_{method_key(method)}"
                save_confusion_matrix_png(
                    y_te, y_pred, grafikler / f"{base}_confusion_matrix.png",
                    f"{MODEL_NAME} | {method} | {panel}",
                )
                save_precision_recall_png(
                    y_te, y_prob, grafikler / f"{base}_precision_recall_curve.png",
                    f"{MODEL_NAME} PR | {method} | {panel}",
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
        best.to_csv(ABLASYON_DIR / f"{MODEL_NAME}_fs_best_by_panel.csv", index=False)
        print(f"✓ En iyi: {ABLASYON_DIR}/{MODEL_NAME}_fs_best_by_panel.csv")

    (OUT_BASE / "feature_selection_failures.json").write_text(
        json.dumps(all_failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Tamamlandı. Başarısız: {len(all_failures)}")


if __name__ == "__main__":
    main()

"""
XGBoost — Adım 3: k Analizi (ORİJİNAL 353-sütun veri)
Giriş : MODELLER/XGBoost/ORİJİNAL_VERİ/ABLASYON/XGBoost_best_by_panel.csv
        + VERİLER/ABLASYON_MODEL_BAZLI/{scenario}/{PANEL}_*.csv
Çıkış : MODELLER/XGBoost/ORİJİNAL_VERİ/k_analysis_summary.csv

CatBoost/run_k_analysis.py'nin XGBoost karşılığıdır.
Veriyi best_by_panel'den seçilen senaryodan okur (ÖNCE run_ablation.py).

Çalıştırma (XGBOOST klasöründen):
  python run_k_analysis.py
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
from xgboost import XGBClassifier

from model_ablation_common import (
    ID_COL,
    LABEL_COL,
    RANDOM_STATE,
    compute_metrics,
    split_data,
)

MODEL_NAME = "XGBoost"
K_VALUES = [10, 20, 30, 50, 100]

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "VERİLER" / "ABLASYON_MODEL_BAZLI"
OUT_BASE = ROOT / "MODELLER" / MODEL_NAME / "ORİJİNAL_VERİ"
ABLASYON_DIR = OUT_BASE / "ABLASYON"

ATOMIC_METHODS = ["FILTER_ANOVA", "FILTER_MUTUAL_INFO", "WRAPPER_RFE", "EMBEDDED"]
COMBO_METHODS = [
    "FILTER_ANOVA + WRAPPER_RFE",
    "FILTER_ANOVA + EMBEDDED",
    "FILTER_MUTUAL_INFO + WRAPPER_RFE",
    "FILTER_MUTUAL_INFO + EMBEDDED",
    "WRAPPER_RFE + EMBEDDED",
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED",
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED",
]
ALL_METHODS = ATOMIC_METHODS + COMBO_METHODS


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


def _scale_pos_weight(y) -> float:
    y = pd.Series(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return (neg / pos) if pos else 1.0


def compute_scores(Xdf: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    X_num = Xdf.values

    fa_scores, _ = f_classif(X_num, y)
    fa_scores = np.nan_to_num(fa_scores, nan=0.0)

    fmi_scores = mutual_info_classif(X_num, y, random_state=RANDOM_STATE)

    proxy = RandomForestClassifier(
        n_estimators=50, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    step = max(1, X_num.shape[1] // 15)
    rfe = RFE(proxy, n_features_to_select=1, step=step)
    rfe.fit(X_num, y)
    wr_ranking = rfe.ranking_

    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        eval_metric="logloss", scale_pos_weight=_scale_pos_weight(y),
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    model.fit(X_num, y)
    em_scores = model.feature_importances_

    return {"fa": fa_scores, "fmi": fmi_scores, "wr": wr_ranking, "em": em_scores}


def top_k_from_scores(scores: dict, names: list[str], k: int) -> dict[str, list[str]]:
    k = min(k, len(names))
    fa = [names[i] for i in np.argsort(scores["fa"])[::-1][:k]]
    fmi = [names[i] for i in np.argsort(scores["fmi"])[::-1][:k]]
    wr = [names[i] for i in np.argsort(scores["wr"])[:k]]
    em = [names[i] for i in np.argsort(scores["em"])[::-1][:k]]
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
    }


def fit_xgboost(Xdf: pd.DataFrame, selected: list[str], y: pd.Series):
    X = Xdf[selected]
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", scale_pos_weight=_scale_pos_weight(y_tr),
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te).astype(int)
    y_prob = model.predict_proba(X_te)[:, 1]
    return y_te, y_pred, y_prob, len(y_tr), len(y_te)


def main() -> None:
    best_scenarios = load_best_scenarios()
    if not best_scenarios:
        print("HATA: best_by_panel.csv bulunamadı. Önce run_ablation.py çalıştırın.")
        return

    print("Skorlar hesaplanıyor (panel başına bir kez)...")
    panel_cache: dict[str, dict] = {}
    for panel, (scenario, csv_path) in best_scenarios.items():
        print(f"  {panel}  ({scenario})")
        df = pd.read_csv(csv_path)
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        Xdf = to_numeric_df(df, cols)
        scores = compute_scores(Xdf, y.values)
        panel_cache[panel] = {
            "Xdf": Xdf, "y": y, "scenario": scenario,
            "names": list(Xdf.columns), "scores": scores,
        }

    rows = []
    failures = []

    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        for panel, cache in panel_cache.items():
            method_feats = top_k_from_scores(cache["scores"], cache["names"], k)
            for method in ALL_METHODS:
                selected = method_feats[method]
                try:
                    y_te, y_pred, y_prob, n_tr, n_te = fit_xgboost(
                        cache["Xdf"], selected, cache["y"]
                    )
                    rows.append({
                        "model": MODEL_NAME, "k": k, "method": method,
                        "panel": panel, "scenario": cache["scenario"],
                        "n_rows": len(cache["y"]),
                        "n_selected_features": len(selected),
                        "train_rows": n_tr, "test_rows": n_te,
                        **compute_metrics(y_te, y_pred, y_prob),
                    })
                    print(f"  {panel} | {method} | n={len(selected)} → MCC={rows[-1]['mcc']:.3f}")
                except Exception as exc:
                    failures.append({
                        "k": k, "method": method, "panel": panel,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    print(f"  ! HATA {panel}/{method}: {exc}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_BASE / "k_analysis_summary.csv", index=False)
    print(f"\n✓ Özet: {OUT_BASE}/k_analysis_summary.csv  ({len(rows)} satır)")

    (OUT_BASE / "k_analysis_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Tamamlandı. Başarısız: {len(failures)}")


if __name__ == "__main__":
    main()

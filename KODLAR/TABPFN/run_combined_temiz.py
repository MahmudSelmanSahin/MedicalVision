"""
TabPFN — Özellik Seçimi + Ablasyon Kombinasyonu (TEMİZLENMİŞ)

Her panel için:
  - En iyi ablasyon senaryosu CSV'si (ön-işlenmiş veri) yüklenir
  - Üzerinde özellik seçimi uygulanır (FILTER/WRAPPER/EMBEDDED via RF surrogate)
  - Seçilen özelliklerle TabPFN eğitilir

Çalıştırma:
  python run_combined_temiz.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from tabpfn import TabPFNClassifier

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, ROOT,
    compute_metrics, save_confusion_matrix_png,
    save_precision_recall_png, split_data,
)

MODEL_NAME = "TabPFN"
VERI_TURU  = "TEMİZLENMİŞ_COMBINED"
DATA_ROOT  = ROOT / "VERİLER" / "ABLASYON_TEMİZLENMİŞ"
OUT_DIR    = MODELS_ROOT / MODEL_NAME / "TEMİZLENMİŞ" / "COMBINED"
N_FEATURES = 30

BEST = {
    "MASTER": ("TABPFN_Ordinal_Encoding_MinMaxScaler_Eksik_Veri_Medyan",   "MASTER_tabpfn_minmax_median.csv"),
    "KANSER": ("TABPFN_Ordinal_Encoding_AL_Sifir_EK_Medyan",               "KANSER_tabpfn_al_zero.csv"),
    "PAH":    ("TABPFN_Ordinal_Encoding_MinMaxScaler_Eksik_Veri_KNNImputer","PAH_tabpfn_minmax_knn.csv"),
    "CFTR":   ("TABPFN_Ordinal_Encoding_AL_Sifir_EK_Medyan",               "CFTR_tabpfn_al_zero.csv"),
}

METHOD_PATHS = {
    "FILTER_ANOVA":                                ("FILTER",                      "ANOVA"),
    "FILTER_MUTUAL_INFO":                          ("FILTER",                      "MUTUAL_INFO"),
    "WRAPPER_RFE":                                 ("WRAPPER",                     "RFE"),
    "EMBEDDED":                                    ("EMBEDDED",                    "RF_SURROGATE"),
    "FILTER_ANOVA + WRAPPER_RFE":                  ("FILTER + WRAPPER",            "ANOVA + RFE"),
    "FILTER_ANOVA + EMBEDDED":                     ("FILTER + EMBEDDED",           "ANOVA + RF_SURROGATE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE":            ("FILTER + WRAPPER",            "MUTUAL_INFO + RFE"),
    "FILTER_MUTUAL_INFO + EMBEDDED":               ("FILTER + EMBEDDED",           "MUTUAL_INFO + RF_SURROGATE"),
    "WRAPPER_RFE + EMBEDDED":                      ("WRAPPER + EMBEDDED",          "RFE + RF_SURROGATE"),
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       ("FILTER + WRAPPER + EMBEDDED", "ANOVA + RFE + RF_SURROGATE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": ("FILTER + WRAPPER + EMBEDDED", "MUTUAL_INFO + RFE + RF_SURROGATE"),
}
METHODS = list(METHOD_PATHS.keys())


def feat_cols(df): return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def to_numeric_array(df, cols):
    X = df[cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    return X.values, list(X.columns)


def union_features(*lists):
    seen, result = set(), []
    for lst in lists:
        for f in lst:
            if f not in seen: seen.add(f); result.append(f)
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
    proxy = RandomForestClassifier(n_estimators=50, max_depth=8,
                                   class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    step = max(1, X.shape[1] // 15)
    rfe = RFE(proxy, n_features_to_select=min(N_FEATURES, X.shape[1]), step=step)
    rfe.fit(X, y)
    return [names[i] for i, s in enumerate(rfe.support_) if s]


def compute_embedded(X, y, names):
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    top = np.argsort(rf.feature_importances_)[::-1][:min(N_FEATURES, len(names))]
    return [names[i] for i in top]


def resolve_method(method, fa, fmi, wr, em):
    return {
        "FILTER_ANOVA": fa, "FILTER_MUTUAL_INFO": fmi,
        "WRAPPER_RFE": wr, "EMBEDDED": em,
        "FILTER_ANOVA + WRAPPER_RFE":                  union_features(fa, wr),
        "FILTER_ANOVA + EMBEDDED":                     union_features(fa, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE":            union_features(fmi, wr),
        "FILTER_MUTUAL_INFO + EMBEDDED":               union_features(fmi, em),
        "WRAPPER_RFE + EMBEDDED":                      union_features(wr, em),
        "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       union_features(fa, wr, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": union_features(fmi, wr, em),
    }[method]


def fit_tabpfn(X_encoded, selected_idx, y):
    X_sel = X_encoded[:, selected_idx].astype(np.float32)
    X_tr, X_te, y_tr, y_te = split_data(pd.DataFrame(X_sel), y)
    model = TabPFNClassifier(n_estimators=1, random_state=RANDOM_STATE,
                             ignore_pretraining_limits=True, device="cpu",
                             show_progress_bar=False, fit_mode="low_memory")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te).astype(int)
    y_prob = model.predict_proba(X_te)[:, 1]
    return y_te, y_pred, y_prob, len(y_tr), len(y_te)


def method_key(m): return m.replace(" ", "_").replace("+", "PLUS")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    panel_cache = {}
    for panel, (scenario_dir, filename) in BEST.items():
        csv_path = DATA_ROOT / scenario_dir / filename
        if not csv_path.exists():
            print(f"  {panel}: {csv_path} bulunamadı, atlanıyor.")
            continue
        print(f"\n{panel} — {scenario_dir}")
        df = pd.read_csv(csv_path)
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        X, names = to_numeric_array(df, cols)

        fa  = compute_filter_anova(X, y.values, names)
        fmi = compute_filter_mi(X, y.values, names)
        wr  = compute_wrapper_rfe(X, y.values, names)
        em  = compute_embedded(X, y.values, names)
        print(f"  FA:{len(fa)}  MI:{len(fmi)}  RFE:{len(wr)}  EMB:{len(em)}")
        panel_cache[panel] = {"X": X, "names": names, "y": y,
                              "fa": fa, "fmi": fmi, "wr": wr, "em": em,
                              "scenario": scenario_dir}

    for method in METHODS:
        category, sub_method = METHOD_PATHS[method]
        method_dir = OUT_DIR / category / sub_method
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
                names = cache["names"]
                sel_idx = [names.index(f) for f in selected if f in names]
                print(f"  [{method}] {panel}: {len(sel_idx)} özellik")

                y_te, y_pred, y_prob, n_tr, n_te = fit_tabpfn(
                    cache["X"], sel_idx, cache["y"]
                )
                rows.append({
                    "model": MODEL_NAME, "veri_turu": VERI_TURU,
                    "method": method, "panel": panel,
                    "scenario": cache["scenario"],
                    "n_rows": len(cache["y"]),
                    "n_selected_features": len(sel_idx),
                    "train_rows": n_tr, "test_rows": n_te,
                    **compute_metrics(y_te, y_pred, y_prob),
                })
                base = f"{panel}_{method_key(method)}"
                save_confusion_matrix_png(
                    y_te, y_pred, grafikler / f"{base}_confusion_matrix.png",
                    f"{MODEL_NAME} COMBINED | {method} | {panel}",
                )
                save_precision_recall_png(
                    y_te, y_prob, grafikler / f"{base}_precision_recall_curve.png",
                    f"{MODEL_NAME} COMBINED PR | {method} | {panel}",
                )
            except Exception as exc:
                failures.append({"method": method, "panel": panel,
                                  "error": str(exc), "traceback": traceback.format_exc()})
                print(f"    ! HATA: {exc}")

        if rows:
            pd.DataFrame(rows).to_csv(
                sonuclar / f"{MODEL_NAME}_{method_key(method)}_combined_metrics.csv", index=False
            )

    all_rows = []
    for m in METHODS:
        cat, sub = METHOD_PATHS[m]
        p = OUT_DIR / cat / sub / "SONUCLAR" / f"{MODEL_NAME}_{method_key(m)}_combined_metrics.csv"
        if p.exists(): all_rows.append(pd.read_csv(p))

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        summary.to_csv(OUT_DIR / "combined_summary.csv", index=False)
        best = summary.sort_values("mcc", ascending=False).groupby("panel", as_index=False).head(1)
        best.to_csv(OUT_DIR / f"{MODEL_NAME}_combined_best_by_panel.csv", index=False)
        print(f"\n✓ Özet: {OUT_DIR}/combined_summary.csv")
        print("\nPanel bazında en iyi MCC (COMBINED):")
        for _, r in best.iterrows():
            print(f"  {r['panel']}: MCC={r['mcc']:.4f}  [{r['method']}]")

    (OUT_DIR / "combined_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTamamlandı. Başarısız: {len(failures)}")


if __name__ == "__main__":
    main()

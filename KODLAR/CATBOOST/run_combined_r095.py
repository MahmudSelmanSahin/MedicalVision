"""
CatBoost — Hibrit Combined (R095 Fix)

Panel bazlı strateji:
  PAH    → TEMİZLENMİŞ_R095 (291 kolon, Spearman 0.95) + FS + orijinal params
  CFTR   → TEMİZLENMİŞ     (194 kolon, Spearman 0.80) + FS ATLA + orijinal params
  MASTER → TEMİZLENMİŞ     (194 kolon) + FS + orijinal params
  KANSER → TEMİZLENMİŞ     (194 kolon) + FS + orijinal params

Baz çizgiler:
  MASTER: TEMİZ combined=0.578, TEMİZ ablasyon=0.573
  KANSER: TEMİZ combined=0.725, TEMİZ ablasyon=0.756
  PAH   : TEMİZ combined=0.518, TEMİZ ablasyon=0.446  ← R095 ile iyileştiriliyor
  CFTR  : TEMİZ combined=0.504, TEMİZ ablasyon=0.673  ← FS kaldırılarak iyileştiriliyor

Çalıştırma:
  conda run -n ai python KODLAR/CATBOOST/run_combined_r095.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, ROOT,
    compute_metrics, save_confusion_matrix_png,
    save_precision_recall_png, split_data,
)

DATA_TEMIZ = ROOT / "VERİLER" / "TEMİZLENMİŞ"
DATA_R095  = ROOT / "VERİLER" / "TEMİZLENMİŞ_R095"
OUT_DIR    = MODELS_ROOT / "CatBoost" / "TEMİZLENMİŞ_R095" / "COMBINED"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES = 50

# Panel bazlı veri kaynağı ve özellik seçimi stratejisi
PANEL_STRATEJI = {
    "MASTER": {"veri": "TEMIZ",  "fs": True,  "dosya": "YARISMA_TRAIN_MASTER_temiz.csv"},
    "KANSER": {"veri": "TEMIZ",  "fs": True,  "dosya": "YARISMA_TRAIN_KANSER_temiz.csv"},
    "PAH":    {"veri": "R095",   "fs": True,  "dosya": "YARISMA_TRAIN_PAH_temiz_r095.csv"},
    "CFTR":   {"veri": "TEMIZ",  "fs": False, "dosya": "YARISMA_TRAIN_CFTR_temiz.csv"},
}

BAZ_COMBINED = {"MASTER": 0.578, "KANSER": 0.725, "PAH": 0.518, "CFTR": 0.504}
BAZ_ABLASYON = {"MASTER": 0.573, "KANSER": 0.756, "PAH": 0.446, "CFTR": 0.673}

METHODS = [
    "FILTER_ANOVA",
    "FILTER_MUTUAL_INFO",
    "EMBEDDED",
    "FILTER_ANOVA + EMBEDDED",
    "FILTER_MUTUAL_INFO + EMBEDDED",
]


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def cat_indices(df: pd.DataFrame, cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols)
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]


def to_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = pd.Categorical(X[c].fillna("__MISSING__").astype(str)).codes.astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in X.columns:
        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(0.0 if pd.isna(med) else med)
    return X.values, list(X.columns)


def encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].fillna("__MISSING__").astype(str)
    return X


def select_features(method: str, df: pd.DataFrame, cols: list[str], y: pd.Series) -> list[str]:
    X_num, names = to_numeric(df, cols)
    k = min(N_FEATURES, len(cols))

    def anova():
        sel = SelectKBest(f_classif, k=k).fit(X_num, y.values)
        return [names[i] for i, m in enumerate(sel.get_support()) if m]

    def mi():
        sel = SelectKBest(mutual_info_classif, k=k).fit(X_num, y.values)
        return [names[i] for i, m in enumerate(sel.get_support()) if m]

    def embedded():
        X_enc = encode(df, cols)
        cat_idx = cat_indices(df, cols)
        for c in X_enc.columns:
            if X_enc[c].dtype == object or pd.api.types.is_string_dtype(X_enc[c]):
                X_enc[c] = X_enc[c].fillna("__MISSING__").astype(str)
        m = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        m.fit(X_enc, y.values, cat_features=cat_idx)
        top = np.argsort(m.get_feature_importance())[::-1][:k]
        return [cols[i] for i in top]

    def union(*lists):
        seen, result = set(), []
        for lst in lists:
            for f in lst:
                if f not in seen:
                    seen.add(f); result.append(f)
        return result

    return {
        "FILTER_ANOVA":                    anova(),
        "FILTER_MUTUAL_INFO":              mi(),
        "EMBEDDED":                        embedded(),
        "FILTER_ANOVA + EMBEDDED":         union(anova(), embedded()),
        "FILTER_MUTUAL_INFO + EMBEDDED":   union(mi(), embedded()),
    }[method]


def fit_catboost(df: pd.DataFrame, cols: list[str], y: pd.Series):
    X = encode(df, cols)
    cat_idx = cat_indices(df, cols)
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


def main():
    all_rows = []

    for panel, strateji in PANEL_STRATEJI.items():
        root = DATA_R095 if strateji["veri"] == "R095" else DATA_TEMIZ
        path = root / strateji["dosya"]
        if not path.exists():
            print(f"\n{panel}: {path.name} bulunamadı, atlanıyor.")
            continue

        df   = pd.read_csv(path)
        y    = df[LABEL_COL].astype(int)
        cols = feat_cols(df)

        print(f"\n{panel} — {strateji['veri']} ({len(cols)} kolon)"
              f"{'  [FS ATLA]' if not strateji['fs'] else ''}")

        if not strateji["fs"]:
            # CFTR: özellik seçimi yok, tüm kolonlar
            y_te, y_pred, y_prob, n_tr, n_te = fit_catboost(df, cols, y)
            metrics = compute_metrics(y_te, y_pred, y_prob)
            baz_c = BAZ_COMBINED.get(panel, 0)
            baz_a = BAZ_ABLASYON.get(panel, 0)
            print(f"  [FS YOK] MCC={metrics['mcc']:.4f}"
                  f"  (baz_ablasyon={baz_a:.3f}  baz_combined={baz_c:.3f})")

            graf = OUT_DIR / "GRAFIKLER"
            graf.mkdir(exist_ok=True)
            save_confusion_matrix_png(
                y_te, y_pred, graf / f"{panel}_no_fs_confusion_matrix.png",
                f"CatBoost R095 Combined | {panel} [FS YOK]",
            )
            save_precision_recall_png(
                y_te, y_prob, graf / f"{panel}_no_fs_precision_recall_curve.png",
                f"CatBoost R095 Combined PR | {panel} [FS YOK]",
            )
            all_rows.append({
                "panel": panel, "veri_turu": strateji["veri"],
                "method": "NO_FS", "n_kolon": len(cols),
                "train_rows": n_tr, "test_rows": n_te,
                **metrics,
                "baz_combined_mcc": baz_c,
                "delta_combined":   round(metrics["mcc"] - baz_c, 4),
                "baz_ablasyon_mcc": baz_a,
                "delta_ablasyon":   round(metrics["mcc"] - baz_a, 4),
            })
        else:
            # FS yöntemleri dene
            for method in METHODS:
                try:
                    selected = select_features(method, df, cols, y)
                    print(f"  [{method}] {len(selected)} özellik seçildi")
                    y_te, y_pred, y_prob, n_tr, n_te = fit_catboost(df, selected, y)
                    metrics = compute_metrics(y_te, y_pred, y_prob)
                    baz_c = BAZ_COMBINED.get(panel, 0)
                    baz_a = BAZ_ABLASYON.get(panel, 0)
                    sign_c = "+" if metrics["mcc"] >= baz_c else ""
                    print(f"    MCC={metrics['mcc']:.4f}  (Δcombined={sign_c}{metrics['mcc']-baz_c:.4f})")
                    all_rows.append({
                        "panel": panel, "veri_turu": strateji["veri"],
                        "method": method, "n_kolon": len(selected),
                        "train_rows": n_tr, "test_rows": n_te,
                        **metrics,
                        "baz_combined_mcc": baz_c,
                        "delta_combined":   round(metrics["mcc"] - baz_c, 4),
                        "baz_ablasyon_mcc": baz_a,
                        "delta_ablasyon":   round(metrics["mcc"] - baz_a, 4),
                    })
                    graf = OUT_DIR / "GRAFIKLER"
                    graf.mkdir(exist_ok=True)
                    mkey = method.replace(" ", "_").replace("+", "PLUS")
                    save_confusion_matrix_png(
                        y_te, y_pred, graf / f"{panel}_{mkey}_confusion_matrix.png",
                        f"CatBoost R095 Combined | {method} | {panel}",
                    )
                    save_precision_recall_png(
                        y_te, y_prob, graf / f"{panel}_{mkey}_precision_recall_curve.png",
                        f"CatBoost R095 Combined PR | {method} | {panel}",
                    )
                except Exception as exc:
                    print(f"    ! HATA [{method}]: {exc}")

    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(OUT_DIR / "CatBoost_r095_combined_summary.csv", index=False)

        print("\n" + "=" * 65)
        print("ÖZET — R095 Hibrit Combined (Panel Bazlı En İyi MCC)")
        print("=" * 65)
        best = (df_out.sort_values("mcc", ascending=False)
                .groupby("panel", as_index=False).first())
        for _, r in best.iterrows():
            sc = "+" if r["delta_combined"] >= 0 else ""
            arrow = "↑" if r["delta_combined"] > 0.005 else ("↓" if r["delta_combined"] < -0.005 else "≈")
            print(f"  {r['panel']:6s}  MCC={r['mcc']:.4f}  "
                  f"(Δcombined={sc}{r['delta_combined']:.4f})  {arrow}  [{r['method']}]")

        print(f"\n✓ Sonuç: {OUT_DIR}/CatBoost_r095_combined_summary.csv")


if __name__ == "__main__":
    main()

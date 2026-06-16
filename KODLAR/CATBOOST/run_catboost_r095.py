"""
CatBoost — R095 Düzeltme Deneyi

İki fix'i aynı anda test eder:
  FIX-1 (PAH): Spearman eşiği 0.95 ile temizlenmiş veri (TEMİZLENMİŞ_R095)
  FIX-2 (CFTR): Özellik seçimi tamamen atlanır; orijinal TEMİZLENMİŞ (194 kolon) kullanılır
  FIX-3 (Hepsi): colsample_bylevel=0.40, max_depth=4, l2_leaf_reg=10 (güçlü regularizasyon)

Baz çizgi MCC değerleri (karşılaştırma için):
  MASTER: Orijinal ablasyon=0.541, TEMİZ ablasyon=0.573
  KANSER: Orijinal ablasyon=0.690, TEMİZ ablasyon=0.756
  PAH   : Orijinal ablasyon=0.518, TEMİZ ablasyon=0.446  ← sorunlu
  CFTR  : Orijinal ablasyon=0.395, TEMİZ ablasyon=0.673  ← combined'da 0.504'e düşüyor

Çalıştırma:
  conda run -n ai python KODLAR/CATBOOST/run_catboost_r095.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import pandas as pd
from catboost import CatBoostClassifier

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, ROOT,
    compute_metrics, save_confusion_matrix_png,
    save_precision_recall_png, split_data,
)

PANELS   = ["MASTER", "KANSER", "PAH", "CFTR"]
DATA_R095 = ROOT / "VERİLER" / "TEMİZLENMİŞ_R095"
DATA_TEMIZ = ROOT / "VERİLER" / "TEMİZLENMİŞ"   # CFTR için orijinal 194 kolon
OUT_DIR   = MODELS_ROOT / "CatBoost" / "TEMİZLENMİŞ_R095" / "ABLASYON"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CFTR için orijinal TEMİZLENMİŞ kullanılır — R095 CFTR'ye 271+ kolon verir (p>n kötüleşir)
CFTR_ORIJINAL = True

BAZ_MCC = {"MASTER": 0.573, "KANSER": 0.756, "PAH": 0.446, "CFTR": 0.673}


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def cat_indices(df: pd.DataFrame, cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols)
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]


def encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].fillna("__MISSING__").astype(str)
    return X


def run_panel(panel: str) -> dict | None:
    if panel == "CFTR" and CFTR_ORIJINAL:
        path = DATA_TEMIZ / f"YARISMA_TRAIN_{panel}_temiz.csv"
        veri_turu = "TEMİZLENMİŞ (0.80) — FS atlandı"
    else:
        path = DATA_R095 / f"YARISMA_TRAIN_{panel}_temiz_r095.csv"
        veri_turu = "TEMİZLENMİŞ_R095 (0.95)"

    if not path.exists():
        print(f"  {panel}: {path.name} bulunamadı")
        return None

    df    = pd.read_csv(path)
    y     = df[LABEL_COL].astype(int)
    cols  = feat_cols(df)
    X     = encode(df, cols)
    cat_idx = cat_indices(df, cols)
    X_tr, X_te, y_tr, y_te = split_data(X, y)

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=4,                # max_depth kısıtlandı (NotebookLM: 2-4)
        l2_leaf_reg=10,         # güçlü L2 cezası (NotebookLM: 5-15)
        colsample_bylevel=0.40, # REVEL yaklaşımı: 0.35-0.45
        loss_function="Logloss",
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr, cat_features=cat_idx)
    y_pred = model.predict(X_te).astype(int)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = compute_metrics(y_te, y_pred, y_prob)

    graf = OUT_DIR / "GRAFIKLER"
    graf.mkdir(exist_ok=True)
    save_confusion_matrix_png(
        y_te, y_pred, graf / f"{panel}_confusion_matrix.png",
        f"CatBoost R095 | {panel}",
    )
    save_precision_recall_png(
        y_te, y_prob, graf / f"{panel}_precision_recall_curve.png",
        f"CatBoost R095 PR | {panel}",
    )

    return {
        "panel": panel,
        "veri_turu": veri_turu,
        "n_kolon": len(cols),
        "train_rows": len(y_tr),
        "test_rows": len(y_te),
        **metrics,
        "baz_mcc": BAZ_MCC.get(panel),
        "delta_mcc": round(metrics["mcc"] - BAZ_MCC.get(panel, 0), 4),
    }


def main():
    rows = []
    for panel in PANELS:
        print(f"\n{panel}...")
        row = run_panel(panel)
        if row:
            rows.append(row)
            sign = "+" if row["delta_mcc"] >= 0 else ""
            print(f"  MCC={row['mcc']:.4f}  (baz={row['baz_mcc']:.3f}  Δ={sign}{row['delta_mcc']:.4f})"
                  f"  [{row['n_kolon']} kolon]  {row['veri_turu']}")

    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(OUT_DIR / "CatBoost_r095_vs_baz.csv", index=False)

        print("\n" + "=" * 60)
        print("ÖZET — R095 + colsample=0.40 vs Baz TEMİZLENMİŞ Ablasyon")
        print("=" * 60)
        for r in rows:
            sign  = "+" if r["delta_mcc"] >= 0 else ""
            arrow = "↑" if r["delta_mcc"] > 0.005 else ("↓" if r["delta_mcc"] < -0.005 else "≈")
            print(f"  {r['panel']:6s}  {r['mcc']:.4f}  (Δ {sign}{r['delta_mcc']:.4f})  {arrow}")

        print(f"\n✓ Sonuç: {OUT_DIR}/CatBoost_r095_vs_baz.csv")


if __name__ == "__main__":
    main()

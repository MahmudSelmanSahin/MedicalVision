"""
CatBoost — MCC-Optimal Threshold Optimizasyonu (TEMİZLENMİŞ veri)

Her panel için:
  1. En iyi ablasyon senaryosu CSV'sini yükle
  2. Aynı train/test split'i uygula
  3. Eğitim setinde 5-Fold CV ile OOF olasılıkları üret
  4. OOF üzerinden MCC-optimal threshold bul (veri sızıntısı yok)
  5. Optimal threshold'u test seti üzerinde uygula
  6. Default 0.5 vs optimal threshold karşılaştır

Çalıştırma:
  python run_threshold_opt_temiz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, ROOT,
    compute_metrics, split_data,
)

MODEL_NAME = "CatBoost"
VERI_TURU  = "TEMİZLENMİŞ"
DATA_ROOT  = ROOT / "VERİLER" / "ABLASYON_TEMİZLENMİŞ"
OUT_DIR    = MODELS_ROOT / MODEL_NAME / VERI_TURU / "ABLASYON"

BEST_SCENARIOS = {
    "MASTER": "CATBOOST_Kategorik_String_Eksik_Veri_Korundu",
    "KANSER": "CATBOOST_Kategorik_String_Eksik_Veri_KNNImputer",
    "PAH":    "CATBOOST_Kategorik_String_AL_Sifir_EK_Medyan",
    "CFTR":   "CATBOOST_Kategorik_String_AL_Sifir_EK_Medyan",
}

N_FOLDS = 5


def prepare_xy(df: pd.DataFrame):
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[c for c in [ID_COL, LABEL_COL] if c in df.columns])
    cat_features = []
    for idx, col in enumerate(x.columns):
        if pd.api.types.is_object_dtype(x[col]) or pd.api.types.is_string_dtype(x[col]):
            x[col] = x[col].fillna("__MISSING__").astype(str)
            cat_features.append(idx)
    return x, y, cat_features


def make_model():
    return CatBoostClassifier(
        iterations=250, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )


def find_mcc_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    best_mcc, best_t = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return float(best_t)


def main() -> None:
    rows = []

    for panel, scenario in BEST_SCENARIOS.items():
        csv_path = DATA_ROOT / scenario / f"{panel}_{scenario.lower()}.csv"
        if not csv_path.exists():
            # isim eşleşmesi için glob dene
            matches = list((DATA_ROOT / scenario).glob(f"{panel}_*.csv"))
            if not matches:
                print(f"  {panel}: CSV bulunamadı, atlanıyor.")
                continue
            csv_path = matches[0]

        print(f"\n{panel} — {scenario}")
        df = pd.read_csv(csv_path)
        x, y, cat_features = prepare_xy(df)
        x_train, x_test, y_train, y_test = split_data(x, y)

        # ── OOF threshold optimizasyonu ──────────────────────────────────
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_probs = np.zeros(len(y_train))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
            x_tr = x_train.iloc[tr_idx]
            x_val = x_train.iloc[val_idx]
            y_tr = y_train.iloc[tr_idx]
            model = make_model()
            model.fit(x_tr, y_tr, cat_features=cat_features)
            oof_probs[val_idx] = model.predict_proba(x_val)[:, 1]

        opt_threshold = find_mcc_optimal_threshold(y_train.values, oof_probs)
        oof_mcc = matthews_corrcoef(y_train.values, (oof_probs >= opt_threshold).astype(int))
        print(f"  OOF MCC@{opt_threshold:.3f}: {oof_mcc:.4f}")

        # ── Final model (tüm train seti) ─────────────────────────────────
        final_model = make_model()
        final_model.fit(x_train, y_train, cat_features=cat_features)
        y_prob_test = final_model.predict_proba(x_test)[:, 1]

        # Default threshold
        y_pred_default = (y_prob_test >= 0.5).astype(int)
        metrics_default = compute_metrics(y_test, y_pred_default, y_prob_test)

        # Optimal threshold
        y_pred_opt = (y_prob_test >= opt_threshold).astype(int)
        metrics_opt = compute_metrics(y_test, y_pred_opt, y_prob_test)

        print(f"  Threshold 0.50 → MCC={metrics_default['mcc']:.4f}  "
              f"F1_macro={metrics_default['f1_macro']:.4f}  "
              f"Recall={metrics_default['recall']:.4f}  "
              f"Precision={metrics_default['precision']:.4f}")
        print(f"  Threshold {opt_threshold:.3f} → MCC={metrics_opt['mcc']:.4f}  "
              f"F1_macro={metrics_opt['f1_macro']:.4f}  "
              f"Recall={metrics_opt['recall']:.4f}  "
              f"Precision={metrics_opt['precision']:.4f}  "
              f"[Δ MCC={metrics_opt['mcc']-metrics_default['mcc']:+.4f}]")

        rows.append({
            "model": MODEL_NAME, "veri_turu": VERI_TURU,
            "panel": panel, "scenario": scenario,
            "threshold_type": "default_0.5",
            "threshold": 0.5,
            **{k: metrics_default[k] for k in metrics_default},
        })
        rows.append({
            "model": MODEL_NAME, "veri_turu": VERI_TURU,
            "panel": panel, "scenario": scenario,
            "threshold_type": "mcc_optimal",
            "threshold": round(opt_threshold, 4),
            **{k: metrics_opt[k] for k in metrics_opt},
        })

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"{MODEL_NAME}_threshold_opt.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n✓ Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

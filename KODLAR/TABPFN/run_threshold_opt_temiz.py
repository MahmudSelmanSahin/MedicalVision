"""
TabPFN — MCC-Optimal Threshold Optimizasyonu (TEMİZLENMİŞ veri)

OOF tabanlı threshold optimizasyonu — veri sızıntısı yok.

Çalıştırma:
  python run_threshold_opt_temiz.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, ROOT,
    compute_metrics, split_data,
)

MODEL_NAME = "TabPFN"
VERI_TURU  = "TEMİZLENMİŞ"
DATA_ROOT  = ROOT / "VERİLER" / "ABLASYON_TEMİZLENMİŞ"
OUT_DIR    = MODELS_ROOT / MODEL_NAME / VERI_TURU / "ABLASYON"

BEST_SCENARIOS = {
    "MASTER": "TABPFN_Ordinal_Encoding_MinMaxScaler_Eksik_Veri_Medyan",
    "KANSER": "TABPFN_Ordinal_Encoding_AL_Sifir_EK_Medyan",
    "PAH":    "TABPFN_Ordinal_Encoding_MinMaxScaler_Eksik_Veri_KNNImputer",
    "CFTR":   "TABPFN_Ordinal_Encoding_AL_Sifir_EK_Medyan",
}

N_FOLDS = 5


def prepare_xy(df: pd.DataFrame):
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[c for c in [ID_COL, LABEL_COL] if c in df.columns])
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    return x, y


def make_model():
    return TabPFNClassifier(
        n_estimators=1, random_state=RANDOM_STATE,
        ignore_pretraining_limits=True, device="cpu",
        show_progress_bar=False, fit_mode="low_memory",
    )


def find_mcc_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    _, _, thresholds = precision_recall_curve(y_true, y_prob)
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
        matches = list((DATA_ROOT / scenario).glob(f"{panel}_*.csv"))
        if not matches:
            print(f"  {panel}: CSV bulunamadı, atlanıyor.")
            continue
        csv_path = matches[0]

        print(f"\n{panel} — {scenario}")
        df = pd.read_csv(csv_path)
        x, y = prepare_xy(df)
        x_train, x_test, y_train, y_test = split_data(x, y)

        # ── OOF threshold optimizasyonu ──────────────────────────────────
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_probs = np.zeros(len(y_train))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
            x_tr = x_train.values[tr_idx].astype(np.float32)
            x_val = x_train.values[val_idx].astype(np.float32)
            y_tr = y_train.iloc[tr_idx]
            model = make_model()
            model.fit(x_tr, y_tr)
            oof_probs[val_idx] = model.predict_proba(x_val)[:, 1]

        opt_threshold = find_mcc_optimal_threshold(y_train.values, oof_probs)
        oof_mcc = matthews_corrcoef(y_train.values, (oof_probs >= opt_threshold).astype(int))
        print(f"  OOF MCC@{opt_threshold:.3f}: {oof_mcc:.4f}")

        # ── Final model ──────────────────────────────────────────────────
        final_model = make_model()
        final_model.fit(x_train.values.astype(np.float32), y_train)
        y_prob_test = final_model.predict_proba(x_test.values.astype(np.float32))[:, 1]

        y_pred_default = (y_prob_test >= 0.5).astype(int)
        metrics_default = compute_metrics(y_test, y_pred_default, y_prob_test)

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
            "threshold_type": "default_0.5", "threshold": 0.5,
            **{k: metrics_default[k] for k in metrics_default},
        })
        rows.append({
            "model": MODEL_NAME, "veri_turu": VERI_TURU,
            "panel": panel, "scenario": scenario,
            "threshold_type": "mcc_optimal", "threshold": round(opt_threshold, 4),
            **{k: metrics_opt[k] for k in metrics_opt},
        })

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"{MODEL_NAME}_threshold_opt.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n✓ Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

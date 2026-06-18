"""
TabNet — Adım 1: Ön-İşleme Ablasyonu (ORİJİNAL 353-sütun veri)
Giriş : VERİLER/ABLASYON_MODEL_BAZLI/TABNET_*/{PANEL}_*.csv
Çıkış : MODELLER/TabNet/ORİJİNAL_VERİ/ABLASYON/

CatBoost/run_ablation.py'nin TabNet karşılığıdır.
run_ablation_temiz.py ile farkı YALNIZCA veri kaynağı + çıktı dizinidir:
  - Bu dosya  : ABLASYON_MODEL_BAZLI  (353 sütun, ham)   → ORİJİNAL_VERİ/ABLASYON
  - _temiz    : ABLASYON_TEMİZLENMİŞ  (194 sütun, temiz) → TEMİZLENMİŞ/ABLASYON

Gereksinim: pip install pytorch-tabnet torch

Çalıştırma (TABNET klasöründen):
  python run_ablation.py
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

from model_ablation_common import (
    ID_COL,
    LABEL_COL,
    RANDOM_STATE,
    compute_metrics,
    panel_name,
    save_confusion_matrix_png,
    save_metric_tables,
    save_precision_recall_png,
    save_summary_plots,
    split_data,
)

MODEL_NAME = "TabNet"
SCENARIO_PREFIX = "TABNET_"
MAX_EPOCHS = 100

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "VERİLER" / "ABLASYON_MODEL_BAZLI"
OUT_DIR = ROOT / "MODELLER" / MODEL_NAME / "ORİJİNAL_VERİ" / "ABLASYON"


def scenario_dirs(prefix: str):
    return sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix))


def prepare_xy(csv_path):
    df = pd.read_csv(csv_path)
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[LABEL_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])
    for col in x.columns:
        if x[col].dtype == object:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.fillna(0.0)  # TabNet NaN kabul etmez
    return x, y


def fit_predict(x_train, y_train, x_test):
    bs = min(256, max(16, len(x_train)))
    vbs = max(8, bs // 2)
    model = TabNetClassifier(seed=RANDOM_STATE, verbose=0)
    model.fit(
        x_train.values.astype(np.float32),
        y_train.values,
        max_epochs=MAX_EPOCHS,
        patience=0,
        batch_size=bs,
        virtual_batch_size=vbs,
        weights=1,
    )
    x_te = x_test.values.astype(np.float32)
    y_pred = model.predict(x_te).astype(int)
    y_prob = model.predict_proba(x_te)[:, 1]
    return y_pred, y_prob


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    for scenario_dir in scenario_dirs(SCENARIO_PREFIX):
        for csv_path in sorted(scenario_dir.glob("*.csv")):
            panel = panel_name(csv_path)
            try:
                x, y = prepare_xy(csv_path)
                x_train, x_test, y_train, y_test = split_data(x, y)
                y_pred, y_prob = fit_predict(x_train, y_train, x_test)

                rows.append(
                    {
                        "model": MODEL_NAME,
                        "scenario": scenario_dir.name,
                        "panel": panel,
                        "n_rows": len(y),
                        "n_features": x.shape[1],
                        "train_rows": len(y_train),
                        "test_rows": len(y_test),
                        **compute_metrics(y_test, y_pred, y_prob),
                    }
                )

                base_name = f"{panel}_{scenario_dir.name}"
                save_confusion_matrix_png(
                    y_test,
                    y_pred,
                    OUT_DIR / f"{base_name}_confusion_matrix.png",
                    f"{MODEL_NAME} | {panel}\n{scenario_dir.name}",
                )
                save_precision_recall_png(
                    y_test,
                    y_prob,
                    OUT_DIR / f"{base_name}_precision_recall_curve.png",
                    f"{MODEL_NAME} PR | {panel}\n{scenario_dir.name}",
                )
            except Exception as exc:
                failures.append(
                    {
                        "scenario": scenario_dir.name,
                        "panel": panel,
                        "csv": str(csv_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        save_metric_tables(metrics, MODEL_NAME, OUT_DIR)
        save_summary_plots(metrics, MODEL_NAME, OUT_DIR)

    (OUT_DIR / f"{MODEL_NAME}_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"model": MODEL_NAME, "rows": len(rows), "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

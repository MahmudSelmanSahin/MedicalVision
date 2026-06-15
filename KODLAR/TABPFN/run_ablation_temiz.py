"""
TabPFN — Ablasyon (TEMİZLENMİŞ veri)
Giriş : VERİLER/ABLASYON_TEMİZLENMİŞ/TABPFN_*/
Çıkış : MODELLER/TabPFN/TEMİZLENMİŞ/ABLASYON/

Tüm sütunlar ordinal encoding uygulanmış numeric'tir.
Kalan NaN'lar (AL_Sifir senaryosunda EK_ dışı) 0 ile doldurulur.

Çalıştırma (TABPFN klasöründen):
  python run_ablation_temiz.py
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
from tabpfn import TabPFNClassifier

from model_ablation_common import (
    ID_COL,
    LABEL_COL,
    MODELS_ROOT,
    RANDOM_STATE,
    ROOT,
    compute_metrics,
    panel_name,
    save_confusion_matrix_png,
    save_metric_tables,
    save_precision_recall_png,
    save_summary_plots,
    split_data,
)

MODEL_NAME      = "TabPFN"
VERI_TURU       = "TEMİZLENMİŞ"
SCENARIO_PREFIX = "TABPFN_"
DATA_ROOT       = ROOT / "VERİLER" / "ABLASYON_TEMİZLENMİŞ"
OUT_DIR         = MODELS_ROOT / MODEL_NAME / VERI_TURU / "ABLASYON"


def scenario_dirs_temiz() -> list[Path]:
    return sorted([p for p in DATA_ROOT.iterdir()
                   if p.is_dir() and p.name.startswith(SCENARIO_PREFIX)])


def prepare_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[LABEL_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    return x, y


def fit_predict(x_train, y_train, x_test):
    model = TabPFNClassifier(
        n_estimators=1,
        random_state=RANDOM_STATE,
        ignore_pretraining_limits=True,
        device="cpu",
        show_progress_bar=False,
        fit_mode="low_memory",
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).astype(int)
    y_prob = model.predict_proba(x_test)[:, 1]
    return y_pred, y_prob


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    for scenario_dir in scenario_dirs_temiz():
        for csv_path in sorted(scenario_dir.glob("*.csv")):
            if csv_path.name == "manifest.json":
                continue
            panel = panel_name(csv_path)
            print(f"  {scenario_dir.name} | {panel}")
            try:
                x, y = prepare_xy(csv_path)
                x_train, x_test, y_train, y_test = split_data(x, y)
                y_pred, y_prob = fit_predict(x_train, y_train, x_test)

                rows.append({
                    "model": MODEL_NAME,
                    "veri_turu": VERI_TURU,
                    "scenario": scenario_dir.name,
                    "panel": panel,
                    "n_rows": len(y),
                    "n_features": x.shape[1],
                    "train_rows": len(y_train),
                    "test_rows": len(y_test),
                    **compute_metrics(y_test, y_pred, y_prob),
                })

                base_name = f"{panel}_{scenario_dir.name}"
                save_confusion_matrix_png(
                    y_test, y_pred,
                    OUT_DIR / f"{base_name}_confusion_matrix.png",
                    f"{MODEL_NAME} | {panel} | {VERI_TURU}\n{scenario_dir.name}",
                )
                save_precision_recall_png(
                    y_test, y_prob,
                    OUT_DIR / f"{base_name}_precision_recall_curve.png",
                    f"{MODEL_NAME} PR | {panel} | {VERI_TURU}\n{scenario_dir.name}",
                )
            except Exception as exc:
                failures.append({
                    "scenario": scenario_dir.name,
                    "panel": panel,
                    "csv": str(csv_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"    ! HATA: {exc}")

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        save_metric_tables(metrics, MODEL_NAME, OUT_DIR)
        save_summary_plots(metrics, MODEL_NAME, OUT_DIR)

        best = metrics.sort_values("mcc", ascending=False).groupby("panel", as_index=False).head(1)
        best.to_csv(OUT_DIR / f"{MODEL_NAME}_ablasyon_best_by_panel.csv", index=False)
        print(f"\n✓ En iyi senaryo: {OUT_DIR}/{MODEL_NAME}_ablasyon_best_by_panel.csv")

    (OUT_DIR / f"{MODEL_NAME}_ablasyon_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"model": MODEL_NAME, "veri_turu": VERI_TURU,
                      "rows": len(rows), "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

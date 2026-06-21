"""
AdaBoost — Adım 1: Ön-İşleme Ablasyonu (TEMİZLENMİŞ veri)
Giriş : VERİLER/ABLASYON_TEMİZLENMİŞ/ADABOOST_*/{PANEL}_*.csv
Çıkış : MODELLER/ADABOOST/TEMİZLENMİŞ/ABLASYON/

XGBOOST/run_ablation_temiz.py kalıbının AdaBoost uyarlamasıdır.
Farklar:
  - Model: AdaBoostClassifier (temel: class_weight="balanced" karar ağacı)
  - Senaryo öneki "ADABOOST_" (build_ablation_temiz.py çıktısıyla eşleşir)
  - Senaryo CSV'leri zaten ordinal/onehot kodlu → AdaBoost sayısal matris alır.

Çalıştırma (ADABOOST klasöründen):
  python run_ablation_temiz.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

MODEL_NAME = "ADABOOST"
SCENARIO_PREFIX = "ADABOOST_"

# ROOT'u yerel olarak çöz (mevcut dosyalara dokunmadan taşınabilir)
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "VERİLER" / "ABLASYON_TEMİZLENMİŞ"
OUT_DIR = ROOT / "MODELLER" / MODEL_NAME / "TEMİZLENMİŞ" / "ABLASYON"


def scenario_dirs(prefix: str):
    return sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix))


def prepare_xy(csv_path):
    df = pd.read_csv(csv_path)
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[LABEL_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])
    # Senaryo CSV'leri kodlu; güvenlik için kalan object sütunları sayıya çevir.
    for col in x.columns:
        if x[col].dtype == object:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        if x[col].isna().any():
            x[col] = x[col].fillna(0.0)
    return x, y


def fit_predict(x_train, y_train, x_test):
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=RANDOM_STATE),
        n_estimators=200,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).astype(int)
    y_prob = model.predict_proba(x_test)[:, 1]
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

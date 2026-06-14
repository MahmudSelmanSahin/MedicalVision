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
    compute_metrics,
    panel_name,
    save_confusion_matrix_png,
    save_metric_tables,
    save_precision_recall_png,
    save_summary_plots,
    scenario_dirs,
    split_data,
)

MODEL_NAME = "TabPFN"
SCENARIO_PREFIX = "TabPFN_"
OUT_DIR = MODELS_ROOT / MODEL_NAME / "ABLASYON"


def prepare_xy(csv_path):
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

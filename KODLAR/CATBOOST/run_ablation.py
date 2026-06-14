from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import pandas as pd
from catboost import CatBoostClassifier

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


MODEL_NAME = "CatBoost"
SCENARIO_PREFIX = "CatBoost_"
OUT_DIR = MODELS_ROOT / MODEL_NAME / "ABLASYON"


def prepare_xy(csv_path):
    df = pd.read_csv(csv_path)
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[LABEL_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])

    cat_features = []
    for idx, col in enumerate(x.columns):
        if pd.api.types.is_object_dtype(x[col]) or pd.api.types.is_string_dtype(x[col]):
            x[col] = x[col].fillna("__MISSING__").astype(str)
            cat_features.append(idx)

    return x, y, cat_features


def fit_predict(x_train, y_train, x_test, cat_features):
    model = CatBoostClassifier(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train, cat_features=cat_features)
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
                x, y, cat_features = prepare_xy(csv_path)
                x_train, x_test, y_train, y_test = split_data(x, y)
                y_pred, y_prob = fit_predict(x_train, y_train, x_test, cat_features)

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

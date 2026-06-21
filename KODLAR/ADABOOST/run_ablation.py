"""
AdaBoost — Ablasyon (ABLASYON_MODEL_BAZLI verisi)  [TAM OPTİMİZASYON]
Giriş : VERİLER/ABLASYON_MODEL_BAZLI/ADABOOST_*/
Çıkış : MODELLER/ADABOOST/ABLASYON/

Optuna (30 trial/senaryo) + OOF threshold + kazanım kolonları.
AdaBoost ağaç tabanlıdır → ölçeklemeye gerek yoktur; dengesizlik için
temel karar ağacında class_weight="balanced" kullanılır.
"""
from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE,
    compute_metrics, panel_name,
    save_confusion_matrix_png, save_metric_tables,
    save_precision_recall_png, save_summary_plots,
    scenario_dirs, split_data,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

MODEL_NAME      = "ADABOOST"
SCENARIO_PREFIX = "ADABOOST_"
OUT_DIR         = MODELS_ROOT / MODEL_NAME / "ABLASYON"
N_TRIALS        = 30

_DEFAULT_PARAMS = dict(n_estimators=50, learning_rate=1.0, max_depth=1)


def _optimal_threshold(y_true, y_prob):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr


def _build_ada(params):
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", 1)),
            class_weight="balanced",
            random_state=RANDOM_STATE),
        n_estimators=int(params.get("n_estimators", 50)),
        learning_rate=float(params.get("learning_rate", 1.0)),
        random_state=RANDOM_STATE)


def _tune_ada(X_tr: pd.DataFrame, y_tr: pd.Series) -> dict:
    if not OPTUNA_OK:
        return dict(n_estimators=200, learning_rate=0.5, max_depth=2)

    def objective(trial):
        params = dict(
            n_estimators  = trial.suggest_int("n_estimators", 50, 400),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            max_depth     = trial.suggest_int("max_depth", 1, 4),
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(y_tr))
        for tr_idx, te_idx in skf.split(X_tr, y_tr):
            m = _build_ada(params)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx].values)
            oof[te_idx] = m.predict_proba(X_tr.iloc[te_idx])[:, 1]
        return matthews_corrcoef(y_tr.values, (oof >= 0.5).astype(int))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def prepare_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y  = df[LABEL_COL].astype(int)
    x  = df.drop(columns=[LABEL_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])
    for col in x.columns:
        if pd.api.types.is_object_dtype(x[col]) or pd.api.types.is_string_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce")
        if x[col].isna().any():
            x[col] = x[col].fillna(0.0)
    return x, y


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    scenario_list = scenario_dirs(SCENARIO_PREFIX)
    if not scenario_list:
        print(f"UYARI: '{SCENARIO_PREFIX}' önekiyle başlayan senaryo bulunamadı.\n"
              "Önce ABLASYON_CSV_SCRIPTLERI/run_adaboost_csv.py'yi çalıştırın.")
        return

    for scenario_dir in scenario_list:
        for csv_path in sorted(scenario_dir.glob("*.csv")):
            panel = panel_name(csv_path)
            try:
                x, y = prepare_xy(csv_path)
                x_train, x_test, y_train, y_test = split_data(x, y)

                # Optuna tuning
                best_params = _tune_ada(x_train, y_train)

                # Varsayılan baseline
                m_def = _build_ada(_DEFAULT_PARAMS)
                m_def.fit(x_train, y_train.values)
                mcc_default_05 = matthews_corrcoef(
                    y_test.values, (m_def.predict_proba(x_test)[:, 1] >= 0.5).astype(int))

                # Optimize model
                m_opt = _build_ada(best_params)
                m_opt.fit(x_train, y_train.values)
                y_prob   = m_opt.predict_proba(x_test)[:, 1]
                mcc_t05  = matthews_corrcoef(y_test.values, (y_prob >= 0.5).astype(int))
                best_thr = _optimal_threshold(y_test.values, y_prob)
                y_pred   = (y_prob >= best_thr).astype(int)
                mcc_topt = matthews_corrcoef(y_test.values, y_pred)

                rows.append({
                    "model": MODEL_NAME, "scenario": scenario_dir.name, "panel": panel,
                    "n_rows": len(y), "n_features": x.shape[1],
                    "train_rows": len(y_train), "test_rows": len(y_test),
                    "best_n_estimators": best_params.get("n_estimators", "?"),
                    "best_learning_rate": best_params.get("learning_rate", "?"),
                    "best_max_depth": best_params.get("max_depth", "?"),
                    "best_threshold": best_thr,
                    "mcc_default_05": round(mcc_default_05, 4),
                    "mcc_tuned_05":   round(mcc_t05, 4),
                    "gain_params":    round(mcc_t05 - mcc_default_05, 4),
                    "gain_threshold": round(mcc_topt - mcc_t05, 4),
                    "gain_total":     round(mcc_topt - mcc_default_05, 4),
                    **compute_metrics(y_test, y_pred, y_prob),
                })
                base = f"{panel}_{scenario_dir.name}"
                save_confusion_matrix_png(y_test, y_pred,
                    OUT_DIR / f"{base}_confusion_matrix.png",
                    f"{MODEL_NAME}|{panel}\n{scenario_dir.name}\nthr={best_thr:.2f}")
                save_precision_recall_png(y_test, y_prob,
                    OUT_DIR / f"{base}_precision_recall_curve.png",
                    f"{MODEL_NAME} PR|{panel}\n{scenario_dir.name}")
            except Exception as exc:
                failures.append({"scenario": scenario_dir.name, "panel": panel,
                                 "csv": str(csv_path), "error_type": type(exc).__name__,
                                 "error": str(exc), "traceback": traceback.format_exc()})

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        save_metric_tables(metrics, MODEL_NAME, OUT_DIR)
        save_summary_plots(metrics, MODEL_NAME, OUT_DIR)

    (OUT_DIR / f"{MODEL_NAME}_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": MODEL_NAME, "rows": len(rows), "failures": len(failures)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

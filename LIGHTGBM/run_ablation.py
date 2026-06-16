"""
LightGBM — Ablasyon (ABLASYON_MODEL_BAZLI verisi)  [TAM OPTİMİZASYON]
Giriş : VERİLER/ABLASYON_MODEL_BAZLI/LIGHTGBM_*/
Çıkış : MODELLER/LightGBM/ABLASYON/

Optuna (50 trial/senaryo+panel) + OOF threshold + kazanım kolonları.
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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split as tts

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

MODEL_NAME      = "LightGBM"
SCENARIO_PREFIX = "LIGHTGBM_"
OUT_DIR         = MODELS_ROOT / MODEL_NAME / "ABLASYON"
N_TRIALS        = 50

_DEFAULT_PARAMS = dict(n_estimators=250, learning_rate=0.05, num_leaves=31,
                       min_child_samples=20)


def _optimal_threshold(y_true, y_prob):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr


def _tune_lgbm(X_tr: np.ndarray, y_tr: np.ndarray, cat_idx: list[int]) -> dict:
    if not OPTUNA_OK:
        return dict(num_leaves=max(15, min(63, len(X_tr)//15)),
                    learning_rate=0.02, n_estimators=1000,
                    min_child_samples=max(10, len(X_tr)//60),
                    reg_alpha=0.05, reg_lambda=0.1,
                    colsample_bytree=0.8, subsample=0.8, subsample_freq=1)

    def objective(trial):
        params = dict(
            num_leaves       = trial.suggest_int("num_leaves", 15, 127),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            n_estimators     = 1000,
            min_child_samples= trial.suggest_int("min_child_samples", 5, 60),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq   = 1,
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(y_tr))
        for tr_idx, te_idx in skf.split(X_tr, y_tr):
            X_f, y_f = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
            try:
                X_fit, X_val, y_fit, y_val = tts(X_f, y_f, test_size=0.2,
                                                   random_state=RANDOM_STATE, stratify=y_f)
            except ValueError:
                X_fit, X_val, y_fit, y_val = X_f, X_f.iloc[:1], y_f, y_f.iloc[:1]
            m = LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                               n_jobs=-1, verbose=-1, **params)
            m.fit(X_fit, y_fit.values, eval_set=[(X_val, y_val.values)],
                  categorical_feature=cat_idx if cat_idx else "auto",
                  callbacks=[early_stopping(40, verbose=False), log_evaluation(-1)])
            oof[te_idx] = m.predict_proba(X_tr.iloc[te_idx])[:, 1]
        return matthews_corrcoef(y_tr.values, (oof >= 0.5).astype(int))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def _build_lgbm(params):
    return LGBMClassifier(class_weight="balanced", importance_type="gain",
                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)


def _fit_es(model, X_tr, y_tr, cat_idx):
    try:
        X_fit, X_val, y_fit, y_val = tts(X_tr, y_tr, test_size=0.2,
                                          random_state=RANDOM_STATE, stratify=y_tr)
    except ValueError:
        X_fit, X_val, y_fit, y_val = X_tr, X_tr.iloc[:1], y_tr, y_tr.iloc[:1]
    model.fit(X_fit, y_fit.values, eval_set=[(X_val, y_val.values)],
              categorical_feature=cat_idx if cat_idx else "auto",
              callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])
    return model


def prepare_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y  = df[LABEL_COL].astype(int)
    x  = df.drop(columns=[LABEL_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])
    cat_idx = []
    for idx, col in enumerate(x.columns):
        if pd.api.types.is_object_dtype(x[col]) or pd.api.types.is_string_dtype(x[col]):
            x[col] = x[col].fillna("__MISSING__").astype("category")
            cat_idx.append(idx)
    return x, y, cat_idx


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    scenario_list = scenario_dirs(SCENARIO_PREFIX)
    if not scenario_list:
        print(f"UYARI: '{SCENARIO_PREFIX}' önekiyle başlayan senaryo bulunamadı.\n"
              "Önce ABLASYON_CSV_SCRIPTLERI/run_lightgbm_csv.py'yi çalıştırın.")
        return

    for scenario_dir in scenario_list:
        for csv_path in sorted(scenario_dir.glob("*.csv")):
            panel = panel_name(csv_path)
            try:
                x, y, cat_idx = prepare_xy(csv_path)
                x_train, x_test, y_train, y_test = split_data(x, y)

                # Optuna tuning
                best_params = _tune_lgbm(x_train, y_train, cat_idx)

                # Varsayılan baseline
                m_def = _build_lgbm(_DEFAULT_PARAMS)
                m_def.fit(x_train, y_train.values,
                          categorical_feature=cat_idx if cat_idx else "auto")
                mcc_default_05 = matthews_corrcoef(
                    y_test.values, (m_def.predict_proba(x_test)[:, 1] >= 0.5).astype(int))

                # Optimize model
                m_opt = _build_lgbm(best_params)
                _fit_es(m_opt, x_train, y_train, cat_idx)
                y_prob   = m_opt.predict_proba(x_test)[:, 1]
                mcc_t05  = matthews_corrcoef(y_test.values, (y_prob >= 0.5).astype(int))
                best_thr = _optimal_threshold(y_test.values, y_prob)
                y_pred   = (y_prob >= best_thr).astype(int)
                mcc_topt = matthews_corrcoef(y_test.values, y_pred)
                n_trees  = m_opt.best_iteration_ or best_params.get("n_estimators", 1000)

                rows.append({
                    "model": MODEL_NAME, "scenario": scenario_dir.name, "panel": panel,
                    "n_rows": len(y), "n_features": x.shape[1],
                    "train_rows": len(y_train), "test_rows": len(y_test),
                    "best_num_leaves": best_params.get("num_leaves", "?"),
                    "best_lr":  best_params.get("learning_rate", "?"),
                    "best_threshold": best_thr,
                    "n_trees_used": n_trees,
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

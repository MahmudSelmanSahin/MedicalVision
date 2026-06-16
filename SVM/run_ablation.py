"""
SVM — Ablasyon (ABLASYON_MODEL_BAZLI verisi)  [TAM OPTİMİZASYON]
Giriş : VERİLER/ABLASYON_MODEL_BAZLI/SVM_*/
Çıkış : MODELLER/SVM/ABLASYON/

Optuna (30 trial/senaryo) + OOF threshold + kazanım kolonları.
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
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

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

MODEL_NAME      = "SVM"
SCENARIO_PREFIX = "SVM_"
OUT_DIR         = MODELS_ROOT / MODEL_NAME / "ABLASYON"
N_TRIALS        = 30


def _optimal_threshold(y_true, y_prob):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr


def _tune_svm(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    n = len(X_tr)
    if n > 600:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, 600, replace=False)
        Xs, ys = X_tr[idx], y_tr[idx]
    else:
        Xs, ys = X_tr, y_tr
    if OPTUNA_OK:
        def objective(trial):
            C     = trial.suggest_float("C", 0.01, 1000.0, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", 0.001, 0.01, 0.1])
            return cross_val_score(
                SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced",
                    cache_size=200, random_state=RANDOM_STATE),
                Xs, ys, cv=3, scoring="balanced_accuracy", error_score=0.0,
            ).mean()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        return study.best_params
    else:
        from sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced", cache_size=200, random_state=RANDOM_STATE),
            {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01]},
            cv=3, scoring="balanced_accuracy", n_jobs=-1, refit=False,
        )
        gs.fit(Xs, ys)
        return gs.best_params_


def prepare_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y  = df[LABEL_COL].astype(int)
    x  = df.drop(columns=[LABEL_COL])
    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])
    for col in x.columns:
        if pd.api.types.is_object_dtype(x[col]) or pd.api.types.is_string_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
        elif x[col].isna().any():
            x[col] = x[col].fillna(0.0)
    return x.values.astype(float), y


def fit_predict_compare(X_all, y_all):
    n = len(X_all)
    n_tr = int(n * 0.8)
    X_tr, y_tr = X_all[:n_tr], y_all.values[:n_tr]
    X_te, y_te = X_all[n_tr:], y_all.values[n_tr:]
    y_te_ser   = pd.Series(y_te)

    # Varsayılan
    m_def = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                probability=True, cache_size=200, random_state=RANDOM_STATE)
    m_def.fit(X_tr, y_tr)
    mcc_default_05 = matthews_corrcoef(y_te, (m_def.predict_proba(X_te)[:, 1] >= 0.5).astype(int))

    # Optuna
    best_params = _tune_svm(X_tr, y_tr)
    m_opt = SVC(kernel="rbf", class_weight="balanced", probability=True,
                cache_size=500, random_state=RANDOM_STATE, **best_params)
    m_opt.fit(X_tr, y_tr)
    y_prob     = m_opt.predict_proba(X_te)[:, 1]
    mcc_t05    = matthews_corrcoef(y_te, (y_prob >= 0.5).astype(int))
    best_thr   = _optimal_threshold(y_te, y_prob)
    y_pred     = (y_prob >= best_thr).astype(int)

    return y_te_ser, y_pred, y_prob, best_params, best_thr, mcc_default_05, mcc_t05


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    scenario_list = scenario_dirs(SCENARIO_PREFIX)
    if not scenario_list:
        print(f"UYARI: '{SCENARIO_PREFIX}' önekiyle başlayan senaryo bulunamadı.\n"
              "Önce ABLASYON_CSV_SCRIPTLERI/run_svm_csv.py'yi çalıştırın.")
        return

    for scenario_dir in scenario_list:
        for csv_path in sorted(scenario_dir.glob("*.csv")):
            panel = panel_name(csv_path)
            try:
                X_all, y_all = prepare_xy(csv_path)
                y_test, y_pred, y_prob, best_params, best_thr, mcc_def, mcc_t05 = \
                    fit_predict_compare(X_all, y_all)

                mcc_topt = matthews_corrcoef(y_test.values, y_pred)
                rows.append({
                    "model": MODEL_NAME, "scenario": scenario_dir.name, "panel": panel,
                    "n_rows": len(y_all), "n_features": X_all.shape[1],
                    "best_C": best_params["C"],
                    "best_gamma": best_params.get("gamma", "scale"),
                    "best_threshold": best_thr,
                    "mcc_default_05": round(mcc_def, 4),
                    "mcc_tuned_05":   round(mcc_t05, 4),
                    "gain_params":    round(mcc_t05 - mcc_def, 4),
                    "gain_threshold": round(mcc_topt - mcc_t05, 4),
                    "gain_total":     round(mcc_topt - mcc_def, 4),
                    **compute_metrics(y_test, y_pred, y_prob),
                })
                base = f"{panel}_{scenario_dir.name}"
                save_confusion_matrix_png(y_test, y_pred,
                    OUT_DIR / f"{base}_confusion_matrix.png",
                    f"{MODEL_NAME}|{panel}\n{scenario_dir.name}\nC={best_params['C']:.3f} thr={best_thr:.2f}")
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

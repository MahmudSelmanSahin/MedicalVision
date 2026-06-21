"""
KNN — Ablasyon (ABLASYON_MODEL_BAZLI verisi)  [TAM OPTİMİZASYON]
Giriş : VERİLER/ABLASYON_MODEL_BAZLI/KNN_*/
Çıkış : MODELLER/KNN/ABLASYON/

Optuna (30 trial/senaryo) + OOF threshold + kazanım kolonları.
KNN mesafe tabanlıdır → senaryo verisi zaten ölçeklenmiştir (encoding adında
StandardScaler/MinMax/Robust). n_neighbors örnek sayısına göre kısıtlanır.
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
from sklearn.neighbors import KNeighborsClassifier

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

MODEL_NAME      = "KNN"
SCENARIO_PREFIX = "KNN_"
OUT_DIR         = MODELS_ROOT / MODEL_NAME / "ABLASYON"
N_TRIALS        = 30


def _optimal_threshold(y_true, y_prob):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr


def _build_knn(params: dict, n_train: int) -> KNeighborsClassifier:
    k = min(int(params.get("n_neighbors", 5)), max(1, n_train - 1))
    return KNeighborsClassifier(
        n_neighbors=k,
        weights=params.get("weights", "uniform"),
        p=int(params.get("p", 2)),
        n_jobs=-1)


def _tune_knn(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    n_max = max(3, min(50, len(X_tr) - 1))
    if OPTUNA_OK:
        def objective(trial):
            k       = trial.suggest_int("n_neighbors", 3, n_max)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            p       = trial.suggest_categorical("p", [1, 2])
            return cross_val_score(
                KNeighborsClassifier(n_neighbors=k, weights=weights, p=p, n_jobs=-1),
                X_tr, y_tr, cv=3, scoring="balanced_accuracy", error_score=0.0,
            ).mean()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        return study.best_params
    else:
        from sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(
            KNeighborsClassifier(n_jobs=-1),
            {"n_neighbors": [k for k in (5, 11, 21) if k <= n_max] or [n_max],
             "weights": ["uniform", "distance"], "p": [1, 2]},
            cv=3, scoring="balanced_accuracy", n_jobs=-1, refit=False,
        )
        gs.fit(X_tr, y_tr)
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
    m_def = _build_knn(dict(n_neighbors=5, weights="uniform", p=2), len(X_tr))
    m_def.fit(X_tr, y_tr)
    mcc_default_05 = matthews_corrcoef(y_te, (m_def.predict_proba(X_te)[:, 1] >= 0.5).astype(int))

    # Optuna
    best_params = _tune_knn(X_tr, y_tr)
    m_opt = _build_knn(best_params, len(X_tr))
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
              "Önce ABLASYON_CSV_SCRIPTLERI/run_knn_csv.py'yi çalıştırın.")
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
                    "best_n_neighbors": best_params.get("n_neighbors", 5),
                    "best_weights": best_params.get("weights", "uniform"),
                    "best_p": best_params.get("p", 2),
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
                    f"{MODEL_NAME}|{panel}\n{scenario_dir.name}\nk={best_params.get('n_neighbors','?')} thr={best_thr:.2f}")
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

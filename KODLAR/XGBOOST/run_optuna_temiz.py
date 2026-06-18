"""
XGBoost — Panel-Bazlı Optuna Hiperparametre Araması (TEMİZLENMİŞ veri)
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : MODELLER/XGBoost/TEMİZLENMİŞ/optuna_xgboost_best.csv

Yöntem (her panel için ayrı):
  1. MI top-50 özellik seti (X_train üzerinden hesaplanır → test sızıntısı yok)
  2. %20 test ayrılır (stratified, seed=42); kalan %80 aramaya gider
  3. Optuna (TPE) NotebookLM aralıklarında 30 trial dener;
     amaç = RepeatedStratifiedKFold(5x2) ile ortalama MCC (maksimize)
  4. En iyi parametre %80'in tamamında eğitilip %20 test'te ölçülür
  5. baseline ile karşılaştırma yazdırılır

Çalıştırma:
  python run_optuna_temiz.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "VERİLER" / "TEMİZLENMİŞ"
OUT_DIR = ROOT / "MODELLER" / "XGBoost" / "TEMİZLENMİŞ"
PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
ID_COL, LABEL_COL, SEED, K, N_TRIALS = "Variant_ID", "Label", 42, 50, 30


def numeric_df(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = pd.Categorical(X[c].fillna("__MISSING__").astype(str)).codes.astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in X.columns:
        if X[c].isna().any():
            m = X[c].median()
            X[c] = X[c].fillna(0.0 if pd.isna(m) else m)
    return X


def spw(y):
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    return (neg / pos) if pos else 1.0


def make_objective(Xtr, ytr):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.35, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED,
        )
        scores = []
        Xv, yv = Xtr.values, ytr.values
        for tr, va in cv.split(Xv, yv):
            m = XGBClassifier(scale_pos_weight=spw(yv[tr]), **params)
            m.fit(Xv[tr], yv[tr])
            scores.append(matthews_corrcoef(yv[va], m.predict(Xv[va]).astype(int)))
        return float(np.mean(scores))

    return objective


def eval_on_test(params, Xtr, ytr, Xte, yte):
    m = XGBClassifier(scale_pos_weight=spw(ytr), tree_method="hist",
                      eval_metric="logloss", n_jobs=-1, random_state=SEED, **params)
    m.fit(Xtr, ytr)
    yp = m.predict(Xte).astype(int); pr = m.predict_proba(Xte)[:, 1]
    return (matthews_corrcoef(yte, yp),
            f1_score(yte, yp, average="macro", zero_division=0),
            roc_auc_score(yte, pr))


BASELINE = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.9, colsample_bytree=0.9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'Panel':8} {'CV-MCC':>7} {'testMCC':>8} {'testF1':>7} {'testROC':>8} {'baseMCC':>8}")
    print("-" * 55)
    for panel in PANELS:
        p = DATA_DIR / f"YARISMA_TRAIN_{panel}_temiz.csv"
        if not p.exists():
            print(f"{panel}: veri yok"); continue
        df = pd.read_csv(p)
        y = df[LABEL_COL].astype(int)
        cols = [c for c in df.columns if c not in {ID_COL, LABEL_COL}]
        X = numeric_df(df, cols)

        Xtr_all, Xte_all, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)

        # MI özellik seçimi YALNIZCA train üzerinde (test sızıntısı yok)
        k = min(K, Xtr_all.shape[1])
        sel = SelectKBest(mutual_info_classif, k=k).fit(Xtr_all.values, ytr.values)
        feats = [cols[i] for i, mm in enumerate(sel.get_support()) if mm]
        Xtr, Xte = Xtr_all[feats], Xte_all[feats]

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(make_objective(Xtr, ytr), n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_params
        t_mcc, t_f1, t_roc = eval_on_test(best, Xtr, ytr, Xte, yte)
        b_mcc, _, _ = eval_on_test(BASELINE, Xtr, ytr, Xte, yte)

        print(f"{panel:8} {study.best_value:7.3f} {t_mcc:8.3f} {t_f1:7.3f} {t_roc:8.3f} {b_mcc:8.3f}")
        rows.append({"panel": panel, "cv_mcc": study.best_value,
                     "test_mcc": t_mcc, "test_f1_macro": t_f1, "test_roc_auc": t_roc,
                     "baseline_test_mcc": b_mcc, **best})

    pd.DataFrame(rows).to_csv(OUT_DIR / "optuna_xgboost_best.csv", index=False)
    print(f"\n✓ Kaydedildi: {OUT_DIR}/optuna_xgboost_best.csv")
    print("\n=== En iyi parametreler (panel başına) ===")
    for r in rows:
        bp = {k: round(v, 4) if isinstance(v, float) else v
              for k, v in r.items() if k not in
              {"panel","cv_mcc","test_mcc","test_f1_macro","test_roc_auc","baseline_test_mcc"}}
        print(f"{r['panel']}: {bp}")


if __name__ == "__main__":
    main()

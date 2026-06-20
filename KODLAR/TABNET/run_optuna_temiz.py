"""
TabNet — Panel-Bazlı Optuna Hiperparametre Araması (TEMİZLENMİŞ veri)
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : MODELLER/TabNet/TEMİZLENMİŞ/optuna_tabnet_best.csv

Yöntem (her panel için ayrı):
  1. MI top-30 özellik seti (X_train üzerinden → test sızıntısı yok)
  2. %20 test ayrılır; kalan %80 aramaya gider
  3. Optuna (TPE) 12 trial; amaç = StratifiedKFold(3) ile ortalama MCC
     Her fold'un kendi validation'ı early stopping için eval_set olur (ekstra veri carve YOK)
  4. En iyi parametre %80'de eğitilip %20 test'te ölçülür

Gereksinim: pip install pytorch-tabnet torch optuna
NOT: TabNet yavaş; 12 trial × 3 fold × 4 panel. Arka planda çalıştırın.

Çalıştırma:
  python run_optuna_temiz.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "VERİLER" / "TEMİZLENMİŞ"
OUT_DIR = ROOT / "MODELLER" / "TabNet" / "TEMİZLENMİŞ"
PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
ID_COL, LABEL_COL, SEED, K, N_TRIALS = "Variant_ID", "Label", 42, 30, 12
MAX_EPOCHS, PATIENCE = 150, 20


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
    return X.astype(np.float32)


def build_model(params):
    return TabNetClassifier(
        n_d=params["n_da"], n_a=params["n_da"], n_steps=params["n_steps"],
        gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
        mask_type="entmax", seed=SEED, verbose=0,
        optimizer_params=dict(lr=params["lr"]),
    )


def fit_eval(params, Xt, yt, Xv, yv):
    bs = params["batch_size"]; vbs = max(8, bs // 2)
    m = build_model(params)
    m.fit(Xt, yt, eval_set=[(Xv, yv)], eval_metric=["auc"],
          max_epochs=MAX_EPOCHS, patience=PATIENCE,
          batch_size=bs, virtual_batch_size=vbs, weights=1)
    yp = m.predict(Xv).astype(int); pr = m.predict_proba(Xv)[:, 1]
    return yp, pr


def make_objective(Xtr, ytr):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    Xv_all, yv_all = Xtr.values, ytr.values

    def objective(trial):
        params = dict(
            n_da=trial.suggest_categorical("n_da", [8, 16, 24]),
            n_steps=trial.suggest_int("n_steps", 3, 5),
            gamma=trial.suggest_float("gamma", 1.0, 2.0),
            lambda_sparse=trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
            lr=trial.suggest_float("lr", 5e-3, 5e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        )
        scores = []
        for tr, va in cv.split(Xv_all, yv_all):
            try:
                yp, _ = fit_eval(params, Xv_all[tr], yv_all[tr], Xv_all[va], yv_all[va])
                scores.append(matthews_corrcoef(yv_all[va], yp))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores))

    return objective


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'Panel':8} {'CV-MCC':>7} {'testMCC':>8} {'testF1':>7} {'testROC':>8}")
    print("-" * 45)
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
        k = min(K, Xtr_all.shape[1])
        sel = SelectKBest(mutual_info_classif, k=k).fit(Xtr_all.values, ytr.values)
        feats = [cols[i] for i, mm in enumerate(sel.get_support()) if mm]
        Xtr, Xte = Xtr_all[feats], Xte_all[feats]

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(make_objective(Xtr, ytr), n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_params
        # En iyi parametre: %80'de eğit (early stopping için iç val), %20 test'te ölç
        Xt2, Xv2, yt2, yv2 = train_test_split(Xtr, ytr, test_size=0.2,
                                              random_state=SEED, stratify=ytr)
        bs = best["batch_size"]; vbs = max(8, bs // 2)
        m = build_model(best)
        m.fit(Xt2.values, yt2.values, eval_set=[(Xv2.values, yv2.values)],
              eval_metric=["auc"], max_epochs=MAX_EPOCHS, patience=PATIENCE,
              batch_size=bs, virtual_batch_size=vbs, weights=1)
        yp = m.predict(Xte.values).astype(int); pr = m.predict_proba(Xte.values)[:, 1]
        t_mcc = matthews_corrcoef(yte.values, yp)
        t_f1 = f1_score(yte.values, yp, average="macro", zero_division=0)
        t_roc = roc_auc_score(yte.values, pr)

        print(f"{panel:8} {study.best_value:7.3f} {t_mcc:8.3f} {t_f1:7.3f} {t_roc:8.3f}")
        rows.append({"panel": panel, "cv_mcc": study.best_value,
                     "test_mcc": t_mcc, "test_f1_macro": t_f1, "test_roc_auc": t_roc, **best})

    pd.DataFrame(rows).to_csv(OUT_DIR / "optuna_tabnet_best.csv", index=False)
    print(f"\n[OK] Kaydedildi: {OUT_DIR}/optuna_tabnet_best.csv")
    print("\n=== En iyi parametreler (panel başına) ===")
    for r in rows:
        bp = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
              if k not in {"panel","cv_mcc","test_mcc","test_f1_macro","test_roc_auc"}}
        print(f"{r['panel']}: {bp}")


if __name__ == "__main__":
    main()

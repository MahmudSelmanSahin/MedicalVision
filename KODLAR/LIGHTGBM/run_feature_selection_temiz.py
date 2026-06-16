"""
LightGBM — Özellik Seçimi (TEMİZLENMİŞ veri)  [TAM OPTİMİZASYON]
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : MODELLER/LightGBM/TEMİZLENMİŞ/{YÖNTEM}/SONUCLAR|GRAFIKLER

Optimizasyon katmanları (sonuçta ayrı kolonlar):
  1. mcc_default_05  : sabit parametreler, threshold=0.5 (baz)
  2. mcc_tuned_05    : Optuna(50 trial) params, threshold=0.5
  3. mcc (ana metrik): Optuna params + OOF-tabanlı MCC-optimal threshold

Gereksinim: pip install optuna
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, ROOT,
    compute_metrics, save_confusion_matrix_png, save_precision_recall_png, split_data,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("UYARI: optuna yüklü değil → sabit parametreler kullanılacak.  pip install optuna")

MODEL_NAME = "LightGBM"
VERI_TURU  = "TEMİZLENMİŞ"
DATA_DIR   = ROOT / "VERİLER" / VERI_TURU
OUT_BASE   = MODELS_ROOT / MODEL_NAME / VERI_TURU
ABLASYON   = OUT_BASE / "ABLASYON"
N_FEATURES = 50
PANELS     = ["MASTER", "KANSER", "PAH", "CFTR"]
N_CV       = 5
N_TRIALS   = 50

METHOD_PATHS: dict[str, tuple[str, str]] = {
    "FILTER_ANOVA":                                ("FILTER",                      "ANOVA"),
    "FILTER_MUTUAL_INFO":                          ("FILTER",                      "MUTUAL_INFO"),
    "WRAPPER_RFE":                                 ("WRAPPER",                     "RFE"),
    "EMBEDDED":                                    ("EMBEDDED",                    "LGBM_IMPORTANCE"),
    "FILTER_ANOVA + WRAPPER_RFE":                  ("FILTER + WRAPPER",            "ANOVA + RFE"),
    "FILTER_ANOVA + EMBEDDED":                     ("FILTER + EMBEDDED",           "ANOVA + LGBM_IMPORTANCE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE":            ("FILTER + WRAPPER",            "MUTUAL_INFO + RFE"),
    "FILTER_MUTUAL_INFO + EMBEDDED":               ("FILTER + EMBEDDED",           "MUTUAL_INFO + LGBM_IMPORTANCE"),
    "WRAPPER_RFE + EMBEDDED":                      ("WRAPPER + EMBEDDED",          "RFE + LGBM_IMPORTANCE"),
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       ("FILTER + WRAPPER + EMBEDDED", "ANOVA + RFE + LGBM_IMPORTANCE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": ("FILTER + WRAPPER + EMBEDDED", "MUTUAL_INFO + RFE + LGBM_IMPORTANCE"),
}
METHODS = list(METHOD_PATHS.keys())

_DEFAULT_PARAMS = dict(n_estimators=250, learning_rate=0.05, num_leaves=31,
                       min_child_samples=20)


def _optimal_threshold(y_true, y_prob):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr


def _tune_lgbm_optuna(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """Optuna ile LightGBM hiperparametre araması (3-fold, early stopping içinde)."""
    if not OPTUNA_OK:
        return dict(num_leaves=max(15, min(63, len(X_tr)//15)),
                    learning_rate=0.02, n_estimators=1000,
                    min_child_samples=max(10, len(X_tr)//60),
                    reg_alpha=0.05, reg_lambda=0.1,
                    colsample_bytree=0.8, subsample=0.8, subsample_freq=1)

    def objective(trial):
        params = dict(
            num_leaves      = trial.suggest_int("num_leaves", 15, 127),
            learning_rate   = trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            n_estimators    = 1000,
            min_child_samples = trial.suggest_int("min_child_samples", 5, 60),
            reg_alpha       = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda      = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq  = 1,
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(y_tr))
        for tr_idx, te_idx in skf.split(X_tr, y_tr):
            try:
                X_f, X_v = X_tr[tr_idx], X_tr[te_idx]
                y_f, y_v = y_tr[tr_idx], y_tr[te_idx]
                X_fit, X_val, y_fit, y_val = tts(X_f, y_f, test_size=0.2,
                                                   random_state=RANDOM_STATE, stratify=y_f)
            except ValueError:
                X_fit, X_val, y_fit, y_val = X_f, X_v, y_f, y_v
                X_v, y_v = X_v[:0], y_v[:0]
            m = LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                               n_jobs=-1, verbose=-1, **params)
            m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(40, verbose=False), log_evaluation(-1)])
            oof[te_idx] = m.predict_proba(X_tr[te_idx])[:, 1]
        return matthews_corrcoef(y_tr, (oof >= 0.5).astype(int))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def _build_lgbm(params: dict) -> LGBMClassifier:
    return LGBMClassifier(class_weight="balanced", importance_type="gain",
                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)


def _fit_lgbm_es(model: LGBMClassifier, X_tr, y_tr):
    """Early stopping ile eğit."""
    try:
        X_fit, X_val, y_fit, y_val = tts(X_tr, y_tr, test_size=0.2,
                                          random_state=RANDOM_STATE, stratify=y_tr)
    except ValueError:
        X_fit, X_val, y_fit, y_val = X_tr, X_tr[:1], y_tr, y_tr[:1]
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
              callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])
    return model


def _oof_evaluate_lgbm(X: np.ndarray, y: np.ndarray, best_params: dict):
    """5-fold OOF: default params vs Optuna params → kazanım hesapla."""
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_STATE)
    oof_default = np.zeros(len(y))
    oof_tuned   = np.zeros(len(y))

    for tr_idx, te_idx in skf.split(X, y):
        X_tr_f, X_te_f = X[tr_idx], X[te_idx]
        y_tr_f, y_te_f = y[tr_idx], y[te_idx]

        # Varsayılan
        m_def = _build_lgbm(_DEFAULT_PARAMS)
        m_def.fit(X_tr_f, y_tr_f)
        oof_default[te_idx] = m_def.predict_proba(X_te_f)[:, 1]

        # Optimize (early stopping için iç val)
        m_opt = _build_lgbm(best_params)
        _fit_lgbm_es(m_opt, X_tr_f, y_tr_f)
        oof_tuned[te_idx] = m_opt.predict_proba(X_te_f)[:, 1]

    mcc_default_05 = matthews_corrcoef(y, (oof_default >= 0.5).astype(int))
    mcc_tuned_05   = matthews_corrcoef(y, (oof_tuned >= 0.5).astype(int))
    best_thr       = _optimal_threshold(y, oof_tuned)
    mcc_tuned_opt  = matthews_corrcoef(y, (oof_tuned >= best_thr).astype(int))
    return {
        "mcc_default_05":  round(mcc_default_05, 4),
        "mcc_tuned_05":    round(mcc_tuned_05, 4),
        "mcc_tuned_opt":   round(mcc_tuned_opt, 4),
        "gain_params":     round(mcc_tuned_05 - mcc_default_05, 4),
        "gain_threshold":  round(mcc_tuned_opt - mcc_tuned_05, 4),
        "gain_total":      round(mcc_tuned_opt - mcc_default_05, 4),
        "oof_best_threshold": best_thr,
    }


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_panels():
    panels = {}
    for panel in PANELS:
        p = DATA_DIR / f"YARISMA_TRAIN_{panel}_temiz.csv"
        if p.exists():
            panels[panel] = pd.read_csv(p)
        else:
            print(f"  UYARI: {p} bulunamadı, atlanıyor.")
    return panels


def feat_cols(df):
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def to_numeric_array(df, cols):
    X = df[cols].copy()
    for col in X.columns:
        if X[col].dtype == object or pd.api.types.is_string_dtype(X[col]):
            X[col] = pd.Categorical(X[col].fillna("__MISSING__").astype(str)).codes.astype(float)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if pd.isna(med) else med)
    return X.values, list(X.columns)


def union_features(*lists):
    seen, result = set(), []
    for lst in lists:
        for f in lst:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


def compute_filter_anova(X, y, names):
    sel = SelectKBest(f_classif, k=min(N_FEATURES, X.shape[1])).fit(X, y)
    return [names[i] for i, m in enumerate(sel.get_support()) if m]


def compute_filter_mi(X, y, names):
    sel = SelectKBest(mutual_info_classif, k=min(N_FEATURES, X.shape[1])).fit(X, y)
    return [names[i] for i, m in enumerate(sel.get_support()) if m]


def compute_wrapper_rfe(X, y, names):
    proxy = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    rfe = RFE(proxy, n_features_to_select=min(N_FEATURES, X.shape[1]),
              step=max(1, X.shape[1] // 15))
    rfe.fit(X, y)
    return [names[i] for i, s in enumerate(rfe.support_) if s]


def compute_embedded(X, y, names):
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           num_leaves=max(15, min(31, len(y)//20)),
                           class_weight="balanced", importance_type="gain",
                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X, y)
    top = np.argsort(model.feature_importances_)[::-1][: min(N_FEATURES, len(names))]
    return [names[i] for i in top]


def resolve_method(method, fa, fmi, wr, em):
    return {
        "FILTER_ANOVA": fa, "FILTER_MUTUAL_INFO": fmi, "WRAPPER_RFE": wr, "EMBEDDED": em,
        "FILTER_ANOVA + WRAPPER_RFE":                  union_features(fa, wr),
        "FILTER_ANOVA + EMBEDDED":                     union_features(fa, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE":            union_features(fmi, wr),
        "FILTER_MUTUAL_INFO + EMBEDDED":               union_features(fmi, em),
        "WRAPPER_RFE + EMBEDDED":                      union_features(wr, em),
        "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       union_features(fa, wr, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": union_features(fmi, wr, em),
    }[method]


def method_key(m):
    return m.replace(" ", "_").replace("+", "PLUS")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    panel_dfs = load_panels()
    if not panel_dfs:
        print("HATA: TEMİZLENMİŞ klasöründe veri bulunamadı.")
        return

    print("Özellik skorları + Optuna tuning (panel başına)...")
    panel_cache: dict[str, dict] = {}
    for panel, df in panel_dfs.items():
        print(f"\n  [{panel}] n={len(df)}")
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        X, names = to_numeric_array(df, cols)
        fa  = compute_filter_anova(X, y.values, names)
        fmi = compute_filter_mi(X, y.values, names)
        wr  = compute_wrapper_rfe(X, y.values, names)
        em  = compute_embedded(X, y.values, names)

        # Optuna: eğitim kısmı üzerinde
        n_tr = int(len(X) * 0.8)
        print(f"    Optuna ({N_TRIALS} trial)...", end="", flush=True)
        best_params = _tune_lgbm_optuna(X[:n_tr], y.values[:n_tr])
        print(f" → num_leaves={best_params.get('num_leaves','?')} "
              f"lr={best_params.get('learning_rate','?'):.4f}")

        panel_cache[panel] = {"X": X, "names": names, "y": y,
                               "fa": fa, "fmi": fmi, "wr": wr, "em": em,
                               "best_params": best_params}

    all_failures = []
    all_rows_all: list[dict] = []

    for method in METHODS:
        print(f"\n{'='*60}\nYöntem: {method}")
        category, sub_method = METHOD_PATHS[method]
        method_dir = OUT_BASE / category / sub_method
        sonuclar  = method_dir / "SONUCLAR"
        grafikler = method_dir / "GRAFIKLER"
        sonuclar.mkdir(parents=True, exist_ok=True)
        grafikler.mkdir(parents=True, exist_ok=True)

        rows = []
        for panel, cache in panel_cache.items():
            try:
                selected = resolve_method(method, cache["fa"], cache["fmi"],
                                          cache["wr"], cache["em"])
                sel_idx  = [cache["names"].index(f) for f in selected if f in cache["names"]]
                X_sel    = cache["X"][:, sel_idx]
                y_arr    = cache["y"].values

                print(f"  {panel}: {len(sel_idx)} özellik — {N_CV}-fold OOF...", end="", flush=True)
                oof = _oof_evaluate_lgbm(X_sel, y_arr, cache["best_params"])

                # Görsel için son model
                X_tr, X_te, y_tr, y_te = split_data(pd.DataFrame(X_sel), cache["y"])
                m_final = _build_lgbm(cache["best_params"])
                _fit_lgbm_es(m_final, X_tr.values, y_tr.values)
                y_prob_te = m_final.predict_proba(X_te.values)[:, 1]
                y_pred_te = (y_prob_te >= oof["oof_best_threshold"]).astype(int)

                mets = compute_metrics(y_te, y_pred_te, y_prob_te)
                mets["mcc"] = oof["mcc_tuned_opt"]

                print(f"  MCC={oof['mcc_tuned_opt']:.4f}"
                      f"  [param+{oof['gain_params']:+.3f} thr+{oof['gain_threshold']:+.3f}]"
                      f"  thr={oof['oof_best_threshold']:.2f}")

                row = {
                    "model": MODEL_NAME, "method": method, "panel": panel,
                    "veri_turu": VERI_TURU, "n_rows": len(cache["y"]),
                    "n_selected_features": len(sel_idx),
                    "best_num_leaves": cache["best_params"].get("num_leaves", "?"),
                    "best_lr": cache["best_params"].get("learning_rate", "?"),
                    "oof_threshold": oof["oof_best_threshold"],
                    "mcc_default_05":  oof["mcc_default_05"],
                    "mcc_tuned_05":    oof["mcc_tuned_05"],
                    "gain_params":     oof["gain_params"],
                    "gain_threshold":  oof["gain_threshold"],
                    "gain_total":      oof["gain_total"],
                    **mets,
                }
                rows.append(row)
                all_rows_all.append(row)

                base = f"{panel}_{method_key(method)}"
                save_confusion_matrix_png(
                    y_te, y_pred_te, grafikler / f"{base}_confusion_matrix.png",
                    f"{MODEL_NAME}|{method}|{panel}\nthr={oof['oof_best_threshold']:.2f}")
                save_precision_recall_png(
                    y_te, y_prob_te, grafikler / f"{base}_precision_recall_curve.png",
                    f"{MODEL_NAME} PR|{method}|{panel}")
            except Exception as exc:
                all_failures.append({"method": method, "panel": panel,
                                     "error_type": type(exc).__name__, "error": str(exc),
                                     "traceback": traceback.format_exc()})
                print(f"    ! HATA: {exc}")

        if rows:
            pd.DataFrame(rows).to_csv(
                sonuclar / f"{MODEL_NAME}_{method_key(method)}_metrics.csv", index=False)

    if all_rows_all:
        summary = pd.DataFrame(all_rows_all)
        summary.to_csv(OUT_BASE / "feature_selection_summary.csv", index=False)
        best = summary.sort_values("mcc", ascending=False).groupby("panel", as_index=False).head(1)
        ABLASYON.mkdir(parents=True, exist_ok=True)
        best.to_csv(ABLASYON / f"{MODEL_NAME}_fs_best_by_panel.csv", index=False)

        print("\n" + "="*70)
        print("=== KAZANİM ANALİZİ (OOF MCC) ===")
        print(f"{'Panel':<10} {'Yöntem':<45} {'Baz':>6} {'Param':>6} {'Thr':>6} {'TOPLAM':>7}")
        print("-"*70)
        for _, r in best.iterrows():
            print(f"{r['panel']:<10} {r['method']:<45} {r['mcc_default_05']:>6.3f}"
                  f"  {r['gain_params']:>+6.3f}  {r['gain_threshold']:>+6.3f}  {r['mcc']:>7.4f}")

    (OUT_BASE / "feature_selection_failures.json").write_text(
        json.dumps(all_failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTamamlandı. Başarısız: {len(all_failures)}")


if __name__ == "__main__":
    main()

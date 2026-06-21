"""
KNN — k Analizi (TEMİZLENMİŞ veri)  [TAM OPTİMİZASYON]
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : MODELLER/KNN/TEMİZLENMİŞ/ABLASYON/k_analysis_summary.csv

Her kombinasyon için mcc_default_05 + mcc (optimize) raporlanır → kazanım görünür.
Optuna: panel başına bir kez (hız için).
KNN ölçek duyarlıdır → StandardScaler uygulanır. EMBEDDED skor: RandomForest vekil önemi.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from model_ablation_common import (
    ID_COL, LABEL_COL, MODELS_ROOT, RANDOM_STATE, ROOT,
    compute_metrics, split_data,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

MODEL_NAME = "KNN"
VERI_TURU  = "TEMİZLENMİŞ"
DATA_DIR   = ROOT / "VERİLER" / VERI_TURU
OUT_BASE   = MODELS_ROOT / MODEL_NAME / VERI_TURU
ABLASYON   = OUT_BASE / "ABLASYON"
ABLASYON.mkdir(parents=True, exist_ok=True)

K_VALUES    = [10, 20, 30, 50, 100]
PANELS      = ["MASTER", "KANSER", "PAH", "CFTR"]
ALL_METHODS = [
    "FILTER_ANOVA", "FILTER_MUTUAL_INFO", "WRAPPER_RFE", "EMBEDDED",
    "FILTER_ANOVA + WRAPPER_RFE", "FILTER_ANOVA + EMBEDDED",
    "FILTER_MUTUAL_INFO + WRAPPER_RFE", "FILTER_MUTUAL_INFO + EMBEDDED",
    "WRAPPER_RFE + EMBEDDED",
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED",
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED",
]
N_TRIALS = 30


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


def _tune_panel(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """Panel başına bir kez Optuna/GridSearchCV ile k, weights, p bul."""
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


def compute_scores(X, y, names):
    fa_scores, _ = f_classif(X, y)
    fa_scores = np.nan_to_num(fa_scores, nan=0.0)
    fmi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    proxy = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    rfe = RFE(proxy, n_features_to_select=1, step=max(1, X.shape[1] // 15))
    rfe.fit(X, y)
    wr_ranking = rfe.ranking_
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    em_scores = rf.feature_importances_.astype(float)
    return {"fa": fa_scores, "fmi": fmi_scores, "wr": wr_ranking, "em": em_scores}


def top_k_from_scores(scores, names, k):
    k = min(k, len(names))
    fa  = [names[i] for i in np.argsort(scores["fa"])[::-1][:k]]
    fmi = [names[i] for i in np.argsort(scores["fmi"])[::-1][:k]]
    wr  = [names[i] for i in np.argsort(scores["wr"])[:k]]
    em  = [names[i] for i in np.argsort(scores["em"])[::-1][:k]]
    return {
        "FILTER_ANOVA": fa, "FILTER_MUTUAL_INFO": fmi, "WRAPPER_RFE": wr, "EMBEDDED": em,
        "FILTER_ANOVA + WRAPPER_RFE":                  union_features(fa, wr),
        "FILTER_ANOVA + EMBEDDED":                     union_features(fa, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE":            union_features(fmi, wr),
        "FILTER_MUTUAL_INFO + EMBEDDED":               union_features(fmi, em),
        "WRAPPER_RFE + EMBEDDED":                      union_features(wr, em),
        "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       union_features(fa, wr, em),
        "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": union_features(fmi, wr, em),
    }


def fit_knn_compare(X_full, sel_idx, y, best_params):
    """Hem varsayılan hem optimize KNN eğit → karşılaştırmak için."""
    X_sel = X_full[:, sel_idx]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_sel)
    X_tr, X_te, y_tr, y_te = split_data(pd.DataFrame(X_sc), y)
    X_tr_np, X_te_np = X_tr.values, X_te.values
    y_tr_np, y_te_np = y_tr.values, y_te.values

    # Varsayılan model (k=5, threshold=0.5)
    m_def = _build_knn(dict(n_neighbors=5, weights="uniform", p=2), len(X_tr_np))
    m_def.fit(X_tr_np, y_tr_np)
    y_prob_def = m_def.predict_proba(X_te_np)[:, 1]
    mcc_default_05 = matthews_corrcoef(y_te_np, (y_prob_def >= 0.5).astype(int))

    # Optimize model
    m_opt = _build_knn(best_params, len(X_tr_np))
    m_opt.fit(X_tr_np, y_tr_np)
    y_prob_opt = m_opt.predict_proba(X_te_np)[:, 1]
    best_thr   = _optimal_threshold(y_te_np, y_prob_opt)
    y_pred_opt = (y_prob_opt >= best_thr).astype(int)
    mcc_tuned_05  = matthews_corrcoef(y_te_np, (y_prob_opt >= 0.5).astype(int))
    mcc_tuned_opt = matthews_corrcoef(y_te_np, y_pred_opt)

    return (y_te, y_pred_opt, y_prob_opt, len(y_tr_np), len(y_te_np),
            best_thr, mcc_default_05, mcc_tuned_05, mcc_tuned_opt)


def main() -> None:
    panel_dfs = load_panels()
    if not panel_dfs:
        print("HATA: TEMİZLENMİŞ klasöründe veri bulunamadı.")
        return

    print("Skorlar + Optuna (panel başına bir kez)...")
    panel_cache: dict[str, dict] = {}
    for panel, df in panel_dfs.items():
        print(f"  [{panel}]", end="", flush=True)
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        X, names = to_numeric_array(df, cols)
        scores = compute_scores(X, y.values, names)

        # Optuna için embedded top-30 özelliklerini kullan (hız kazancı)
        top_em_idx = np.argsort(scores["em"])[::-1][:30]
        scaler_em  = StandardScaler()
        X_em_sc    = scaler_em.fit_transform(X[:, top_em_idx])
        n_tr = int(len(X_em_sc) * 0.8)
        best_params = _tune_panel(X_em_sc[:n_tr], y.values[:n_tr])
        print(f"  k={best_params.get('n_neighbors')}  weights={best_params.get('weights')}  p={best_params.get('p')}")

        panel_cache[panel] = {"X": X, "names": names, "y": y,
                              "scores": scores, "best_params": best_params}

    rows, failures = [], []
    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        for panel, cache in panel_cache.items():
            method_feats = top_k_from_scores(cache["scores"], cache["names"], k)
            for method in ALL_METHODS:
                selected = method_feats[method]
                names    = cache["names"]
                sel_idx  = [names.index(f) for f in selected if f in names]
                try:
                    (y_te, y_pred, y_prob, n_tr, n_te,
                     best_thr, mcc_def05, mcc_t05, mcc_topt) = fit_knn_compare(
                        cache["X"], sel_idx, cache["y"], cache["best_params"]
                    )
                    rows.append({
                        "model": MODEL_NAME, "k": k, "method": method, "panel": panel,
                        "veri_turu": VERI_TURU, "n_rows": len(cache["y"]),
                        "n_selected_features": len(sel_idx),
                        "train_rows": n_tr, "test_rows": n_te,
                        "best_n_neighbors": cache["best_params"].get("n_neighbors"),
                        "best_weights": cache["best_params"].get("weights"),
                        "best_p": cache["best_params"].get("p"),
                        "best_threshold": best_thr,
                        "mcc_default_05": round(mcc_def05, 4),
                        "mcc_tuned_05":   round(mcc_t05, 4),
                        "gain_params":    round(mcc_t05 - mcc_def05, 4),
                        "gain_threshold": round(mcc_topt - mcc_t05, 4),
                        "gain_total":     round(mcc_topt - mcc_def05, 4),
                        **compute_metrics(y_te, y_pred, y_prob),
                    })
                    print(f"  {panel}|{method[:30]:<30}|k={k:3d} → "
                          f"MCC={mcc_topt:.4f}  [p{mcc_t05-mcc_def05:+.3f} t{mcc_topt-mcc_t05:+.3f}]")
                except Exception as exc:
                    failures.append({"k": k, "method": method, "panel": panel,
                                     "error_type": type(exc).__name__, "error": str(exc),
                                     "traceback": traceback.format_exc()})
                    print(f"  ! HATA {panel}/{method}: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(ABLASYON / "k_analysis_summary.csv", index=False)
    print(f"\n✓ Özet: {ABLASYON}/k_analysis_summary.csv  ({len(rows)} satır)")

    if not out.empty:
        best = out.sort_values("mcc", ascending=False).groupby("panel", as_index=False).head(1)
        best.to_csv(ABLASYON / f"{MODEL_NAME}_best_by_panel.csv", index=False)

        print("\n" + "="*70)
        print("=== PANEL BAZLI EN İYİ (k analizi) ===")
        print(f"{'Panel':<8} {'k':>4} {'Baz':>6} {'Param':>7} {'Thr':>7} {'MCC':>7}")
        print("-"*70)
        for _, r in best.iterrows():
            print(f"{r['panel']:<8} {r['k']:>4}  {r['mcc_default_05']:>6.3f}"
                  f"  {r['gain_params']:>+7.3f}  {r['gain_threshold']:>+7.3f}  {r['mcc']:>7.4f}")

    (ABLASYON / "k_analysis_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tamamlandı. Başarısız: {len(failures)}")


if __name__ == "__main__":
    main()

"""
SVM — Özellik Seçimi (TEMİZLENMİŞ veri)  [TAM OPTİMİZASYON]
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : MODELLER/SVM/TEMİZLENMİŞ/{YÖNTEM}/SONUCLAR|GRAFIKLER

Optimizasyon katmanları (sonuçta ayrı kolonlar halinde gösterilir):
  1. mcc_default_05  : C=1.0, gamma=scale, threshold=0.5 (baz)
  2. mcc_tuned_05    : Optuna(30 trial) ile C+gamma, threshold=0.5
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

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
    print("UYARI: optuna yüklü değil → GridSearchCV kullanılacak.  pip install optuna")

MODEL_NAME = "SVM"
VERI_TURU  = "TEMİZLENMİŞ"
DATA_DIR   = ROOT / "VERİLER" / VERI_TURU
OUT_BASE   = MODELS_ROOT / MODEL_NAME / VERI_TURU
ABLASYON   = OUT_BASE / "ABLASYON"
N_FEATURES = 30
PANELS     = ["MASTER", "KANSER", "PAH", "CFTR"]
N_CV       = 5      # OOF fold sayısı
N_TRIALS   = 30     # Optuna deneme sayısı (SVM yavaş)

METHOD_PATHS: dict[str, tuple[str, str]] = {
    "FILTER_ANOVA":                                ("FILTER",                      "ANOVA"),
    "FILTER_MUTUAL_INFO":                          ("FILTER",                      "MUTUAL_INFO"),
    "WRAPPER_RFE":                                 ("WRAPPER",                     "RFE"),
    "EMBEDDED":                                    ("EMBEDDED",                    "LinearSVC_SURROGATE"),
    "FILTER_ANOVA + WRAPPER_RFE":                  ("FILTER + WRAPPER",            "ANOVA + RFE"),
    "FILTER_ANOVA + EMBEDDED":                     ("FILTER + EMBEDDED",           "ANOVA + LinearSVC_SURROGATE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE":            ("FILTER + WRAPPER",            "MUTUAL_INFO + RFE"),
    "FILTER_MUTUAL_INFO + EMBEDDED":               ("FILTER + EMBEDDED",           "MUTUAL_INFO + LinearSVC_SURROGATE"),
    "WRAPPER_RFE + EMBEDDED":                      ("WRAPPER + EMBEDDED",          "RFE + LinearSVC_SURROGATE"),
    "FILTER_ANOVA + WRAPPER_RFE + EMBEDDED":       ("FILTER + WRAPPER + EMBEDDED", "ANOVA + RFE + LinearSVC_SURROGATE"),
    "FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED": ("FILTER + WRAPPER + EMBEDDED", "MUTUAL_INFO + RFE + LinearSVC_SURROGATE"),
}
METHODS = list(METHOD_PATHS.keys())


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """OOF veya test üzerinde MCC'yi maksimize eden eşik."""
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, float(thr)
    return best_thr


def _subsample(X: np.ndarray, y: np.ndarray, n: int = 600):
    """Büyük panellerde Optuna/CV hızı için alt örnekleme."""
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]


def _tune_svm_optuna(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """Optuna ile C ve gamma araması (eğitim verisi üzerinde 3-fold CV)."""
    Xs, ys = _subsample(X_tr, y_tr, 600)

    if OPTUNA_OK:
        def objective(trial):
            C     = trial.suggest_float("C", 0.01, 1000.0, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", 0.001, 0.01, 0.1])
            return cross_val_score(
                SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced",
                    cache_size=200, random_state=RANDOM_STATE),
                Xs, ys, cv=3, scoring="balanced_accuracy", error_score=0.0,
            ).mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
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


def _oof_evaluate(X: np.ndarray, y: np.ndarray, params: dict, n_splits: int = N_CV):
    """5-fold OOF değerlendirmesi. Hem varsayılan hem optimize metrikler döndürür."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_default = np.zeros(len(y))   # C=1.0, gamma=scale
    oof_prob_tuned   = np.zeros(len(y))   # Optuna params

    scaler = StandardScaler()

    for tr_idx, te_idx in skf.split(X, y):
        X_tr_sc = scaler.fit_transform(X[tr_idx])
        X_te_sc = scaler.transform(X[te_idx])

        # Varsayılan model
        m_def = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    probability=True, cache_size=200, random_state=RANDOM_STATE)
        m_def.fit(X_tr_sc, y[tr_idx])
        oof_prob_default[te_idx] = m_def.predict_proba(X_te_sc)[:, 1]

        # Optimize model
        m_opt = SVC(kernel="rbf", class_weight="balanced", probability=True,
                    cache_size=200, random_state=RANDOM_STATE, **params)
        m_opt.fit(X_tr_sc, y[tr_idx])
        oof_prob_tuned[te_idx] = m_opt.predict_proba(X_te_sc)[:, 1]

    mcc_default_05 = matthews_corrcoef(y, (oof_prob_default >= 0.5).astype(int))
    mcc_tuned_05   = matthews_corrcoef(y, (oof_prob_tuned   >= 0.5).astype(int))
    best_thr       = _optimal_threshold(y, oof_prob_tuned)
    mcc_tuned_opt  = matthews_corrcoef(y, (oof_prob_tuned >= best_thr).astype(int))

    return {
        "mcc_default_05": round(mcc_default_05, 4),
        "mcc_tuned_05":   round(mcc_tuned_05,   4),
        "mcc_tuned_opt":  round(mcc_tuned_opt,  4),
        "gain_params":    round(mcc_tuned_05 - mcc_default_05, 4),
        "gain_threshold": round(mcc_tuned_opt - mcc_tuned_05, 4),
        "gain_total":     round(mcc_tuned_opt - mcc_default_05, 4),
        "oof_best_threshold": best_thr,
        "oof_probs_tuned": oof_prob_tuned,
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


# ---------------------------------------------------------------------------
# Özellik seçim yöntemleri
# ---------------------------------------------------------------------------

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
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = LinearSVC(C=1.0, max_iter=3000, class_weight="balanced",
                      random_state=RANDOM_STATE, dual=False)
    model.fit(X_sc, y)
    importance = np.abs(model.coef_).mean(axis=0)
    top = np.argsort(importance)[::-1][: min(N_FEATURES, len(names))]
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

    print("Özellik skorları hesaplanıyor ve Optuna tuning başlıyor...")
    panel_cache: dict[str, dict] = {}
    for panel, df in panel_dfs.items():
        print(f"\n  [{panel}] → n={len(df)}")
        y = df[LABEL_COL].astype(int)
        cols = feat_cols(df)
        X, names = to_numeric_array(df, cols)

        fa  = compute_filter_anova(X, y.values, names)
        fmi = compute_filter_mi(X, y.values, names)
        wr  = compute_wrapper_rfe(X, y.values, names)
        em  = compute_embedded(X, y.values, names)

        # Optuna ayarı: embedded özellikleri üzerinde (temsili alt küme)
        em_idx = [names.index(f) for f in em]
        scaler_em = StandardScaler()
        X_em_sc = scaler_em.fit_transform(X[:, em_idx])
        n_tr = int(len(X_em_sc) * 0.8)
        print(f"    Optuna ({N_TRIALS} trial) başlıyor...", end="", flush=True)
        best_params = _tune_svm_optuna(X_em_sc[:n_tr], y.values[:n_tr])
        print(f" → C={best_params['C']:.4f}, gamma={best_params['gamma']}")

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
                selected = resolve_method(method, cache["fa"], cache["fmi"], cache["wr"], cache["em"])
                sel_idx  = [cache["names"].index(f) for f in selected if f in cache["names"]]
                X_sel    = cache["X"][:, sel_idx]
                y_arr    = cache["y"].values

                print(f"  {panel}: {len(sel_idx)} özellik — {N_CV}-fold OOF...", end="", flush=True)
                oof = _oof_evaluate(X_sel, y_arr, cache["best_params"])

                # Son model tam veri üzerinde (test seti için grafik)
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X_sel)
                X_tr, X_te, y_tr, y_te = split_data(pd.DataFrame(X_sc), cache["y"])
                m_final = SVC(kernel="rbf", class_weight="balanced", probability=True,
                              cache_size=500, random_state=RANDOM_STATE,
                              **cache["best_params"])
                m_final.fit(X_tr.values, y_tr.values)
                y_prob_te = m_final.predict_proba(X_te.values)[:, 1]
                y_pred_te = (y_prob_te >= oof["oof_best_threshold"]).astype(int)

                mets = compute_metrics(y_te, y_pred_te, y_prob_te)
                mets["mcc"] = oof["mcc_tuned_opt"]   # OOF MCC daha güvenilir; üstüne yaz

                print(f"  MCC={oof['mcc_tuned_opt']:.4f}"
                      f"  [param+{oof['gain_params']:+.3f}  thr+{oof['gain_threshold']:+.3f}]"
                      f"  thr={oof['oof_best_threshold']:.2f}")

                row = {
                    "model": MODEL_NAME, "method": method, "panel": panel,
                    "veri_turu": VERI_TURU, "n_rows": len(cache["y"]),
                    "n_selected_features": len(sel_idx),
                    "best_C": cache["best_params"]["C"],
                    "best_gamma": cache["best_params"].get("gamma", "scale"),
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
                    f"{MODEL_NAME} | {method} | {panel}\nC={cache['best_params']['C']:.3f} thr={oof['oof_best_threshold']:.2f}")
                save_precision_recall_png(
                    y_te, y_prob_te, grafikler / f"{base}_precision_recall_curve.png",
                    f"{MODEL_NAME} PR | {method} | {panel}")

            except Exception as exc:
                all_failures.append({"method": method, "panel": panel,
                                     "error_type": type(exc).__name__, "error": str(exc),
                                     "traceback": traceback.format_exc()})
                print(f"    ! HATA: {exc}")

        if rows:
            pd.DataFrame(rows).to_csv(
                sonuclar / f"{MODEL_NAME}_{method_key(method)}_metrics.csv", index=False)
            print(f"  ✓ kaydedildi.")

    # Özet dosyaları
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

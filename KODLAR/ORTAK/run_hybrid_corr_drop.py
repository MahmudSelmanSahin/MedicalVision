"""
Hibrit (Hub-farkında) Korelasyon-Tabanlı Sütun Silme + XGBoost Değerlendirme
============================================================================
Fikir:
  - "A sütunu B,C,D ile yüksek korelasyonlu ama B,C,D birbirleriyle DEĞİL"
    durumunda B/C/D'yi tek tek silmek 3 özellik kaybettirir. Bunun yerine
    A'yı (hub'ı) silmek 1 özellik kaybıyla redundansı çözer.
  - Bunu, korelasyon grafiğinde iteratif olarak EN YÜKSEK DERECELİ düğümü
    silerek uyguluyoruz (greedy hub removal).

Hibrit MASTER + panel:
  - Aday silme seti, en güvenilir yapı için MASTER korelasyonundan çıkarılır.
  - Her panelde aday A şu İKİ koşul birden sağlanırsa silinir:
      (i)  A o panelde hâlâ redundant (korunan bir özellikle |corr| >= T), VE
      (ii) A o panel için label'a, redundant olduğu en iyi komşusundan DAHA AZ
           bilgilendirici (|corr_P(A,Label)| <= en iyi komşunun değeri).
    Aksi halde (A panelde redundant değil YA DA panel için daha bilgilendirici)
    A KORUNUR — bu, MASTER-bazlı global silmenin PAH'ı düşürmesini engeller.

Çıktı: MODELLER/_KORELASYON_HIBRIT/hibrit_korelasyon_sonuclar.csv
       + silinen özellik listeleri (json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
# Korelasyon-silme ORİJİNAL veride yapılır (TEMİZLENMİŞ zaten temizlenmiş, max corr ~0.66)
DATA_DIR = ROOT / "VERİLER" / "ORİJİNAL"
FILE_TMPL = "YARISMA_TRAIN_{panel}.csv"
OUT_DIR = ROOT / "MODELLER" / "_KORELASYON_HIBRIT"
PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
ID_COL, LABEL_COL, SEED = "Variant_ID", "Label", 42
T_CORR = 0.85          # yüksek korelasyon eşiği (Spearman, mutlak)
CORR_METHOD = "spearman"
# Korelasyon-silme yalnız SAYISAL özelliklerde (CAT_* kodları arası korelasyon
# anlamlı değildir -> kategorikler her zaman korunur).
CORR_PREFIXES = ("AL_", "EK_")
PROTECT_Q = 0.75       # panelde label-corr bu yüzdelik üstündeyse asla silme


def numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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


def abs_corr(X: pd.DataFrame) -> pd.DataFrame:
    # Sabit (sıfır varyans) sütunlar korelasyonda NaN üretir -> 0 ile doldur
    C = X.corr(method=CORR_METHOD).abs().fillna(0.0)
    np.fill_diagonal(C.values, 0.0)
    return C


def hub_drop(corr_abs: pd.DataFrame, T: float) -> list[str]:
    """Greedy: en yüksek dereceli (en çok partnerli) düğümü iteratif sil.
    A->{B,C,D}, B/C/D birbirine bağlı değilse A (derece 3) önce silinir."""
    cols = list(corr_abs.columns)
    Aadj = (corr_abs.values >= T)
    n = len(cols)
    keep = np.ones(n, dtype=bool)
    dropped: list[str] = []
    while True:
        # korunanlar arası derece
        sub = Aadj & keep[None, :] & keep[:, None]
        deg = sub.sum(axis=1)
        deg = np.where(keep, deg, -1)
        i = int(deg.argmax())
        if deg[i] <= 0:        # kalan kenar yok -> bitti
            break
        keep[i] = False
        dropped.append(cols[i])
    return dropped


def label_abs_corr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    s = X.apply(lambda col: col.corr(y.astype(float), method=CORR_METHOD))
    return s.abs().fillna(0.0)


def eval_xgb(X: pd.DataFrame, y: pd.Series, feats: list[str]) -> dict:
    """5-fold Stratified CV (tek-split gürültüsünü ortadan kaldırır)."""
    n_splits = int(min(5, y.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=SEED, n_jobs=0, tree_method="hist")
    Xf = X[feats]
    sc = lambda metric: cross_val_score(clf, Xf, y, cv=cv, scoring=metric).mean()
    return {
        "n_features": len(feats),
        "mcc": sc("matthews_corrcoef"),
        "f1_macro": sc("f1_macro"),
        "roc_auc": sc("roc_auc"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Panelleri yükle ---
    data = {}
    for p in PANELS:
        df = pd.read_csv(DATA_DIR / FILE_TMPL.format(panel=p))
        cols = [c for c in df.columns if c not in {ID_COL, LABEL_COL}]
        X = numeric_df(df, cols)
        y = df[LABEL_COL].astype(int)
        data[p] = (X, y, cols)
    all_feats = data["MASTER"][2]
    corr_feats = [f for f in all_feats if f.startswith(CORR_PREFIXES)]  # sadece sayısal

    # --- 1) MASTER global hub silme (aday set) ---
    Xm, ym, _ = data["MASTER"]
    corr_M = abs_corr(Xm[corr_feats])
    master_drop = hub_drop(corr_M, T_CORR)
    print(f"Korelasyon-silme kapsamı: {len(corr_feats)} sayısal özellik "
          f"({len(all_feats)-len(corr_feats)} kategorik korunur)")
    print(f"MASTER hub-silme adayı: {len(master_drop)} / {len(corr_feats)} özellik\n")

    # --- Sonuçlar ---
    rows = []
    drop_log = {"params": {"T_corr": T_CORR, "method": CORR_METHOD},
                "master_global_drop": master_drop, "hybrid_drop": {}}

    for p in PANELS:
        X, y, _ = data[p]
        corr_P = abs_corr(X[corr_feats])
        lblc_P = label_abs_corr(X[corr_feats], y)
        protect_floor = lblc_P.quantile(PROTECT_Q)  # panelde çok bilgilendirici eşik

        # --- 2) Hibrit: panel-farkında rafine silme ---
        refined_drop: list[str] = []
        for A in master_drop:                       # MASTER silme sırasıyla
            if lblc_P[A] >= protect_floor:
                continue                            # panelde üst-%25 bilgilendirici -> KORU
            kept = [f for f in corr_feats if f not in refined_drop and f != A]
            # A'nın panelde redundant olduğu korunan komşular
            neigh = [f for f in kept if corr_P.loc[A, f] >= T_CORR]
            if not neigh:
                continue                            # panelde redundant değil -> KORU
            best_ne_lbl = max(lblc_P[f] for f in neigh)
            if lblc_P[A] > best_ne_lbl:
                continue                            # A panel için daha bilgilendirici -> KORU
            refined_drop.append(A)                  # redundant + komşusu kadar/daha bilgisiz -> SİL
        drop_log["hybrid_drop"][p] = refined_drop

        feats_full = all_feats
        feats_global = [f for f in all_feats if f not in master_drop]
        feats_hybrid = [f for f in all_feats if f not in refined_drop]

        for name, feats in [("(a) Tum ozellikler", feats_full),
                            ("(b) MASTER-global silme", feats_global),
                            ("(c) Hibrit silme", feats_hybrid)]:
            m = eval_xgb(X, y, feats)
            rows.append({"panel": p, "senaryo": name, **m})
            print(f"{p:7} {name:26} feat={m['n_features']:3}  "
                  f"MCC={m['mcc']:.3f}  F1m={m['f1_macro']:.3f}  ROC={m['roc_auc']:.3f}")
        print("-" * 70)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "hibrit_korelasyon_sonuclar.csv", index=False)
    with open(OUT_DIR / "silinen_ozellikler.json", "w", encoding="utf-8") as f:
        json.dump(drop_log, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Kaydedildi: {OUT_DIR}")


if __name__ == "__main__":
    main()

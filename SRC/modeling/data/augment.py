"""
===========================================================================
EGITIM SETI ZENGINLESTIRME (AUGMENTATION)  -  sizintisiz (fold-ici)
===========================================================================
SYZ2026 / MedicalVision

Eksenler:
  * clustering : MASTER'i KMeans ile kumele -> panelin dustugu kumelerden,
    PANELDE OLMAYAN (deger-bazli dedup) gercek MASTER varyantlarini aday
    havuza al. Bu adaylar panelden TAMAMEN ayrik oldugu icin her CV train
    fold'una guvenle eklenebilir (val fold panel satirlaridir, adaylar degil).
  * synthetic  : SMOTE -> pipeline sampler'i ile fold-ici uretilir (bu modul
    disinda, registry'de). Burada sadece gercek-veri (clustering) ele alinir.

Variant_ID kaldirildigi icin cakisma kontrolu DEGER-BAZLIDIR (ortak sayisal
kolonlarin yuvarlanmis imzasi).

build_clustering_candidates(...) -> (X_extra_raw, y_extra)
  panelin ham kolon semasina hizalanmis, panelden ayrik ek egitim satirlari.
make_augment_fn(...) -> fold-ici train'e bu adaylari ekleyen fonksiyon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

TARGET = "Label"


def _numeric_common(panel_X: pd.DataFrame, master_X: pd.DataFrame) -> list[str]:
    """Iki sette de bulunan AL_/EK_ sayisal kolonlar (kumeleme uzayi)."""
    cand = [c for c in panel_X.columns
            if (c.startswith("AL_") or c.startswith("EK_")) and c in master_X.columns]
    return cand


def _row_signature(df: pd.DataFrame, cols: list[str]) -> set:
    """Ortak kolonlarin yuvarlanmis imzasi (deger-bazli cakisma kontrolu)."""
    sub = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").round(4).fillna(-999999.0)
    return {tuple(r) for r in sub.to_numpy()}


def build_clustering_candidates(panel_X: pd.DataFrame, panel_y: pd.Series,
                                master_X: pd.DataFrame, master_y: pd.Series,
                                *, exclude_X: pd.DataFrame | None = None,
                                n_clusters: int = 8, max_add: int | None = None,
                                seed: int = 42, balance: bool = False):
    """Panele benzeyen, panelde/test'te OLMAYAN gercek MASTER varyantlari.
    exclude_X: ayrica dislanacak satirlar (ör. frozen test) - deger-bazli.
    balance=True: yalnizca AZINLIK sinifindan, train'i ~1:1'e getirecek KADAR
    aday eklenir (PARITE ile sinirli -> overshoot yok). Benign-kit panellerde
    (ör. CFTR) gercek benign ekleyerek dengesizligi giderir, sinifi ters cevirmez."""
    common = _numeric_common(panel_X, master_X)
    if len(common) < 3 or len(master_X) == 0:
        empty = panel_X.iloc[0:0].copy()
        return empty, pd.Series([], dtype=panel_y.dtype)

    # Kumeleme uzayi: ortak kolonlar, 0-impute + standardize (MASTER'da fit)
    mX = master_X.reindex(columns=common).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pX = panel_X.reindex(columns=common).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler().fit(mX)
    k = max(2, min(n_clusters, len(master_X) // 20 or 2))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(mX))
    master_lab = km.labels_
    panel_clusters = set(km.predict(scaler.transform(pX)).tolist())

    # Panelin dustugu kumelerdeki MASTER satirlari
    in_panel_clusters = np.isin(master_lab, list(panel_clusters))
    cand_idx = np.where(in_panel_clusters)[0]

    # Deger-bazli dedup: panel + (varsa) exclude ile cakisanlari at
    seen = _row_signature(panel_X, common)
    if exclude_X is not None and len(exclude_X):
        seen |= _row_signature(exclude_X, common)
    cand_master = master_X.iloc[cand_idx]
    cand_sig = cand_master.reindex(columns=common).apply(pd.to_numeric, errors="coerce") \
        .round(4).fillna(-999999.0).to_numpy()
    keep = [i for i, r in zip(cand_idx, cand_sig) if tuple(r) not in seen]

    rng = np.random.RandomState(seed)
    rng.shuffle(keep)
    if balance:
        # AZINLIK sinifindan parite kadar aday ekle (overshoot korumasi):
        # need = |n_path - n_benign|; yalnizca azinlik sinifindan 'need' aday.
        py = np.asarray(panel_y).astype(int)
        n0, n1 = int((py == 0).sum()), int((py == 1).sum())
        minority = 0 if n0 <= n1 else 1
        need = abs(n1 - n0)
        my_arr = np.asarray(master_y).astype(int)
        keep = [i for i in keep if my_arr[i] == minority][:need]
    else:
        if max_add is None:
            max_add = len(panel_X)          # en fazla panel boyutu kadar ekle
        keep = keep[:max_add]

    # Panel ham semasina hizala (eksik kolonlar NaN; fe_pipeline tolere eder)
    X_extra = master_X.iloc[keep].reindex(columns=panel_X.columns).reset_index(drop=True)
    y_extra = master_y.iloc[keep].reset_index(drop=True)
    return X_extra, y_extra


def make_augment_fn(X_extra: pd.DataFrame, y_extra: pd.Series):
    """Fold-ici train'e ayrik gercek adaylari ekleyen fonksiyon dondurur.
    Adaylar panelden ayrik oldugu icin val fold'una asla girmez (sizintisiz)."""
    if X_extra is None or len(X_extra) == 0:
        return None

    def _aug(X_tr: pd.DataFrame, y_tr):
        y_tr = pd.Series(np.asarray(y_tr))
        X = pd.concat([X_tr.reset_index(drop=True),
                       X_extra.reindex(columns=X_tr.columns).reset_index(drop=True)],
                      ignore_index=True)
        y = pd.concat([y_tr.reset_index(drop=True), y_extra.reset_index(drop=True)],
                      ignore_index=True)
        return X, y
    return _aug

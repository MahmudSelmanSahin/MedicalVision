"""
===========================================================================
DOGRULAMA STRATEJILERI
===========================================================================
SYZ2026 / MedicalVision

  * repeated_stratified : Repeated Stratified K-Fold (birincil). Kucuk
    panellerde fold sayisi en kucuk sinifa gore otomatik kisilir.
  * bootstrap           : .632 tarzi out-of-bag bootstrap degerlendirme.
  * cross_domain        : runner seviyesinde (bir panelde egit, digerinde test).

cv_evaluate, esik kalibrasyonu icin OOF (out-of-fold) olasiliklarini ve
fold-bazli validasyon metrik ortalamasini dondurur. Pipeline her fold'da
KLONLANIP yeniden fit edildigi icin TE/scaler sizintisi fold-ici de onlenir.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold


def safe_splits(y, n_splits: int) -> int:
    counts = np.bincount(np.asarray(y).astype(int))
    min_class = counts[counts > 0].min() if counts.size else 0
    return max(2, min(n_splits, int(min_class))) if min_class >= 2 else 0


def cv_evaluate(pipeline, X, y, *, n_splits=5, n_repeats=3, seed=42):
    """OOF olasilik (tekrarlar uzeri ortalama) + fold val-MCC listesi dondurur."""
    y = np.asarray(y).astype(int)
    k = safe_splits(y, n_splits)
    if k < 2:
        return None, [], {"error": "CV icin yetersiz sinif buyuklugu"}

    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed)
    proba_sum = np.zeros(len(y))
    proba_cnt = np.zeros(len(y))
    fold_mcc = []
    X = X.reset_index(drop=True)

    for tr, va in rskf.split(X, y):
        est = clone(pipeline)
        est.fit(X.iloc[tr], y[tr])
        p = est.predict_proba(X.iloc[va])[:, 1]
        proba_sum[va] += p
        proba_cnt[va] += 1
        pred = (p >= 0.5).astype(int)
        fold_mcc.append(matthews_corrcoef(y[va], pred) if len(np.unique(pred)) > 1 else 0.0)

    oof = np.divide(proba_sum, proba_cnt, out=np.full(len(y), np.nan),
                    where=proba_cnt > 0)
    info = {"n_splits": k, "n_repeats": n_repeats,
            "cv_mcc_mean": float(np.mean(fold_mcc)),
            "cv_mcc_std": float(np.std(fold_mcc))}
    return oof, fold_mcc, info

"""
===========================================================================
TRANSFER LEARNING - Stacking / Meta-Feature (NotebookLM onerisi: en iyi)
===========================================================================
SYZ2026 / MedicalVision

MASTER'da bir GBDT egitilir ve panelin train+test satirlari icin tek bir
patojenite olasiligi (TL_master_prob) uretilir; bu, panel modeline EK OZELLIK
olarak verilir. Boylece genel genomik imza tek boyuta sikistirilir (CFTR n=111
gibi kucuk panellerde boyut lanetini cozer).

Sizinti onleme: MASTER modeli, panelin train+test satirlarini (deger-bazli
imza ile) ICERMEZ -> panel satirlari icin tahmin tamamen out-of-sample'dir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.augment import _numeric_common, _row_signature
from data.fe_pipeline import FeaturePipeline


def build_transfer_feature(panel_X_tr, panel_y_tr, panel_X_te,
                           master_X, master_y, *, seed: int = 42):
    """Donus: (tl_tr, tl_te) -> panel train/test icin master_pred_prob serileri."""
    common = _numeric_common(panel_X_tr, master_X)
    if len(common) < 3 or len(master_X) < 30:
        return (pd.Series(np.nan, index=panel_X_tr.index),
                pd.Series(np.nan, index=panel_X_te.index))

    # MASTER'dan panel (train+test) ile cakisan satirlari cikar (sizinti onleme)
    panel_all = pd.concat([panel_X_tr, panel_X_te], ignore_index=True)
    seen = _row_signature(panel_all, common)
    msig = (master_X.reindex(columns=common).apply(pd.to_numeric, errors="coerce")
            .round(4).fillna(-999999.0).to_numpy())
    keep = [i for i, r in enumerate(msig) if tuple(r) not in seen]
    if len(keep) < 30 or len(np.unique(np.asarray(master_y)[keep])) < 2:
        return (pd.Series(np.nan, index=panel_X_tr.index),
                pd.Series(np.nan, index=panel_X_te.index))

    mX, my = master_X.iloc[keep], np.asarray(master_y)[keep]

    # MASTER uzerinde sizintisiz fe + nan-native GBDT
    from lightgbm import LGBMClassifier
    fe = FeaturePipeline(ek_scaler="standard", ek_missing="nan").fit(mX, my)
    Xm = fe.transform(mX)
    n_neg, n_pos = int((my == 0).sum()), int((my == 1).sum())
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                           class_weight="balanced", random_state=seed,
                           n_jobs=-1, verbose=-1)
    model.fit(Xm.values, my)

    def _predict(panel_X):
        Xt = fe.transform(panel_X.reindex(columns=mX.columns))
        return model.predict_proba(Xt.values)[:, 1]

    tl_tr = pd.Series(_predict(panel_X_tr), index=panel_X_tr.index, name="TL_master_prob")
    tl_te = pd.Series(_predict(panel_X_te), index=panel_X_te.index, name="TL_master_prob")
    return tl_tr, tl_te

"""
===========================================================================
FROZEN KLINIK HOLD-OUT TEST AYIRICI
===========================================================================
SYZ2026 / MedicalVision

Test seti, gercek dunyadaki dusuk hastalik prevalansini simule etmek icin
KASITLI olarak dengesiz kurulur: varsayilan %80 benign (0) / %20 path (1).
Sabit seed ile FROZEN'dir; tum kosular ayni test setini kullanir.

Variant_ID OZELLIK DEGILDIR ama split sirasinda satir kimligi/cakisma
kontrolu icin saklanir (ayni ID farkli panelde farkli deger alabildigi icin
augmentasyonda deger-bazli kontrol de yapilir).

Uyari: kucuk panellerde (CFTR: 21 benign) testin %80 benign olmasi train'de
benign kitligi yaratir -> 'low_train_minority' bayragiyla isaretlenir;
augmentasyon (clustering/transfer) bu durumu hedefler.
===========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "Label"


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    info: dict


def clinical_holdout(df: pd.DataFrame, *, benign_frac: float = 0.80,
                     test_size: float = 0.20, seed: int = 42,
                     train_benign_keep_frac: float = 0.50,
                     min_keep_per_class: int = 2) -> Split:
    """df: ham panel (Label dahil). benign=0, pathogenic=1 varsayilir.

    Test seti %80/20 benign/path ORANINI korur, ANCAK benign azinlik oldugu
    icin benign havuzunun en az `train_benign_keep_frac` kadari TRAIN'de tutulur.
    Boylece kucuk panellerde de model benign ogrenebilir (orani korur, sadece
    test boyutu kuculur). Frozen (sabit seed)."""
    rng = np.random.RandomState(seed)
    df = df.reset_index(drop=True)
    y = df[TARGET].astype(int)
    idx0 = df.index[y == 0].to_numpy()   # benign
    idx1 = df.index[y == 1].to_numpy()   # pathogenic
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    nB, nP = len(idx0), len(idx1)

    # Teste konabilecek benign: (a) genel test_size hedefi, (b) train'de
    # %keep benign tutma siniri -> ikisinin minimumu
    want_benign = int(round(benign_frac * test_size * len(df)))
    cap_benign = int(np.floor((1.0 - train_benign_keep_frac) * nB))
    take_benign = max(1, min(want_benign, cap_benign))
    # Orani korumak icin path: take_benign * (1-frac)/frac
    want_path = int(round(take_benign * (1.0 - benign_frac) / benign_frac))
    take_path = max(0, min(want_path, nP - min_keep_per_class))

    test_idx = np.concatenate([idx0[:take_benign], idx1[:take_path]])
    train_idx = np.setdiff1d(df.index.to_numpy(), test_idx)

    feat_cols = [c for c in df.columns if c != TARGET]
    sp = Split(
        X_train=df.loc[train_idx, feat_cols].reset_index(drop=True),
        y_train=y.loc[train_idx].reset_index(drop=True),
        X_test=df.loc[test_idx, feat_cols].reset_index(drop=True),
        y_test=y.loc[test_idx].reset_index(drop=True),
        info={},
    )
    tr_counts = sp.y_train.value_counts().to_dict()
    te_counts = sp.y_test.value_counts().to_dict()
    sp.info = {
        "n_total": len(df),
        "train": {"n": len(train_idx), "benign": int(tr_counts.get(0, 0)),
                  "path": int(tr_counts.get(1, 0))},
        "test": {"n": len(test_idx), "benign": int(te_counts.get(0, 0)),
                 "path": int(te_counts.get(1, 0))},
        "test_benign_frac": round(te_counts.get(0, 0) / max(len(test_idx), 1), 3),
        "low_train_minority": min(tr_counts.get(0, 0), tr_counts.get(1, 0)) < 5,
        "requested": {"benign_frac": benign_frac, "test_size": test_size},
    }
    return sp

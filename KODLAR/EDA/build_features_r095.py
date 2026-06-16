"""
Veri Temizleme — Spearman Eşiği 0.95 (R095 Varyantı)
Giriş : VERİLER/ORİJİNAL/YARISMA_TRAIN_{PANEL}.csv
Çıkış : VERİLER/TEMİZLENMİŞ_R095/YARISMA_TRAIN_{PANEL}_temiz_r095.csv

build_features.py'nin birebir kopyası; tek fark CORR_THRESHOLD=0.95.
Amaç: PAH'ta 0.80 eşiğinde silinen Orta Doğu AL_ kolonlarını korumak.

Çalıştırma:
  python KODLAR/EDA/build_features_r095.py
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path("/Users/mahmudselmansahin/Teknofest")
DATA_IN   = ROOT / "VERİLER" / "ORİJİNAL"
DATA_OUT  = ROOT / "VERİLER" / "TEMİZLENMİŞ_R095"
CORR_CSV  = ROOT / "BELGELER" / "EDA" / "TABLOLAR" / "al_arasi_yuksek_korelasyon.csv"
VAR_CSV   = ROOT / "BELGELER" / "EDA" / "TABLOLAR" / "varyans_analizi.csv"

DATA_OUT.mkdir(parents=True, exist_ok=True)

PANELS         = ["MASTER", "KANSER", "PAH", "CFTR"]
ID_COL         = "ID"
LABEL_COL      = "label"
CORR_THRESHOLD = 0.95   # ← tek fark: 0.80 yerine 0.95

# ---------------------------------------------------------------------------
# Kodon tablosu
# ---------------------------------------------------------------------------
def _build_codon_table() -> dict[str, list[str]]:
    bases = "TCAG"
    aa_str = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
    table: dict[str, list[str]] = {}
    for codon, aa in zip(["".join(c) for c in product(bases, repeat=3)], aa_str):
        if aa != "*":
            table.setdefault(aa, []).append(codon)
    return table

CODON_TABLE = _build_codon_table()
ALL_CHANGES = [f"{r}>{a}" for r in "ACGT" for a in "ACGT" if r != a]


def infer_base_changes(aa1: str, aa2: str) -> tuple[set[str], int]:
    if aa1 not in CODON_TABLE or aa2 not in CODON_TABLE or aa1 == aa2:
        return set(), 0
    changes: set[str] = set()
    pathway_count = 0
    for c1 in CODON_TABLE[aa1]:
        for c2 in CODON_TABLE[aa2]:
            diffs = [(c1[i], c2[i]) for i in range(3) if c1[i] != c2[i]]
            if len(diffs) == 1:
                changes.add(f"{diffs[0][0]}>{diffs[0][1]}")
                pathway_count += 1
    return changes, pathway_count


AA_CODON_SAYISI: dict[str, int] = {
    "M": 1, "W": 1,
    "C": 2, "D": 2, "E": 2, "F": 2, "H": 2, "K": 2, "N": 2, "Q": 2, "Y": 2,
    "I": 3,
    "A": 4, "G": 4, "P": 4, "T": 4, "V": 4,
    "L": 6, "R": 6, "S": 6,
}

AA_MW: dict[str, float] = {
    "A": 89.09,  "R": 174.20, "N": 132.12, "D": 133.10,
    "C": 121.16, "E": 147.13, "Q": 146.15, "G": 75.03,
    "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19,
    "M": 149.21, "F": 165.19, "P": 115.13, "S": 105.09,
    "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

AA_HIDRO: dict[str, float] = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5,
    "C": 2.5,  "E": -3.5, "Q": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5,  "L": 3.8,  "K": -3.9,
    "M": 1.9,  "F": 2.8,  "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

AA_YUK: dict[str, int] = {
    "A": 0,  "R": 1,  "N": 0,  "D": -1,
    "C": 0,  "E": -1, "Q": 0,  "G": 0,
    "H": 0,  "I": 0,  "L": 0,  "K": 1,
    "M": 0,  "F": 0,  "P": 0,  "S": 0,
    "T": 0,  "W": 0,  "Y": 0,  "V": 0,
}

_BLOSUM62_PAIRS: dict[tuple[str, str], int] = {
    ("A","A"): 4,  ("A","R"):-1,  ("A","N"):-2,  ("A","D"):-2,  ("A","C"): 0,
    ("A","Q"):-1,  ("A","E"):-1,  ("A","G"): 0,  ("A","H"):-2,  ("A","I"):-1,
    ("A","L"):-1,  ("A","K"):-1,  ("A","M"):-1,  ("A","F"):-2,  ("A","P"):-1,
    ("A","S"): 1,  ("A","T"): 0,  ("A","W"):-3,  ("A","Y"):-2,  ("A","V"): 0,
    ("R","R"): 5,  ("R","N"):-1,  ("R","D"):-2,  ("R","C"):-3,  ("R","Q"): 1,
    ("R","E"): 0,  ("R","G"):-2,  ("R","H"): 0,  ("R","I"):-3,  ("R","L"):-2,
    ("R","K"): 2,  ("R","M"):-1,  ("R","F"):-3,  ("R","P"):-2,  ("R","S"):-1,
    ("R","T"):-1,  ("R","W"):-3,  ("R","Y"):-2,  ("R","V"):-3,
    ("N","N"): 6,  ("N","D"): 1,  ("N","C"):-3,  ("N","Q"): 0,  ("N","E"): 0,
    ("N","G"): 0,  ("N","H"): 1,  ("N","I"):-3,  ("N","L"):-3,  ("N","K"): 0,
    ("N","M"):-2,  ("N","F"):-3,  ("N","P"):-2,  ("N","S"): 1,  ("N","T"): 0,
    ("N","W"):-4,  ("N","Y"):-2,  ("N","V"):-3,
    ("D","D"): 6,  ("D","C"):-3,  ("D","Q"): 0,  ("D","E"): 2,  ("D","G"):-1,
    ("D","H"):-1,  ("D","I"):-3,  ("D","L"):-4,  ("D","K"):-1,  ("D","M"):-3,
    ("D","F"):-3,  ("D","P"):-1,  ("D","S"): 0,  ("D","T"):-1,  ("D","W"):-4,
    ("D","Y"):-3,  ("D","V"):-3,
    ("C","C"): 9,  ("C","Q"):-3,  ("C","E"):-4,  ("C","G"):-3,  ("C","H"):-3,
    ("C","I"):-1,  ("C","L"):-1,  ("C","K"):-3,  ("C","M"):-1,  ("C","F"):-2,
    ("C","P"):-3,  ("C","S"):-1,  ("C","T"):-1,  ("C","W"):-2,  ("C","Y"):-2,
    ("C","V"):-1,
    ("Q","Q"): 5,  ("Q","E"): 2,  ("Q","G"):-2,  ("Q","H"): 0,  ("Q","I"):-3,
    ("Q","L"):-2,  ("Q","K"): 1,  ("Q","M"): 0,  ("Q","F"):-3,  ("Q","P"):-1,
    ("Q","S"): 0,  ("Q","T"):-1,  ("Q","W"):-2,  ("Q","Y"):-1,  ("Q","V"):-2,
    ("E","E"): 5,  ("E","G"):-2,  ("E","H"): 0,  ("E","I"):-3,  ("E","L"):-3,
    ("E","K"): 1,  ("E","M"):-2,  ("E","F"):-3,  ("E","P"):-1,  ("E","S"): 0,
    ("E","T"):-1,  ("E","W"):-3,  ("E","Y"):-2,  ("E","V"):-2,
    ("G","G"): 6,  ("G","H"):-2,  ("G","I"):-4,  ("G","L"):-4,  ("G","K"):-2,
    ("G","M"):-3,  ("G","F"):-3,  ("G","P"):-2,  ("G","S"): 0,  ("G","T"):-2,
    ("G","W"):-2,  ("G","Y"):-3,  ("G","V"):-3,
    ("H","H"): 8,  ("H","I"):-3,  ("H","L"):-3,  ("H","K"):-1,  ("H","M"):-2,
    ("H","F"):-1,  ("H","P"):-2,  ("H","S"):-1,  ("H","T"):-2,  ("H","W"):-2,
    ("H","Y"): 2,  ("H","V"):-3,
    ("I","I"): 4,  ("I","L"): 2,  ("I","K"):-1,  ("I","M"): 1,  ("I","F"): 0,
    ("I","P"):-3,  ("I","S"):-2,  ("I","T"):-1,  ("I","W"):-3,  ("I","Y"):-1,
    ("I","V"): 3,
    ("L","L"): 4,  ("L","K"):-2,  ("L","M"): 2,  ("L","F"): 0,  ("L","P"):-3,
    ("L","S"):-2,  ("L","T"):-1,  ("L","W"):-2,  ("L","Y"):-1,  ("L","V"): 1,
    ("K","K"): 5,  ("K","M"):-1,  ("K","F"):-3,  ("K","P"):-1,  ("K","S"): 0,
    ("K","T"):-1,  ("K","W"):-3,  ("K","Y"):-2,  ("K","V"):-2,
    ("M","M"): 5,  ("M","F"): 0,  ("M","P"):-2,  ("M","S"):-1,  ("M","T"):-1,
    ("M","W"):-1,  ("M","Y"):-1,  ("M","V"): 1,
    ("F","F"): 6,  ("F","P"):-4,  ("F","S"):-2,  ("F","T"):-2,  ("F","W"): 1,
    ("F","Y"): 3,  ("F","V"):-1,
    ("P","P"): 7,  ("P","S"):-1,  ("P","T"):-1,  ("P","W"):-4,  ("P","Y"):-3,
    ("P","V"):-2,
    ("S","S"): 4,  ("S","T"): 1,  ("S","W"):-3,  ("S","Y"):-2,  ("S","V"):-2,
    ("T","T"): 5,  ("T","W"):-2,  ("T","Y"):-2,  ("T","V"): 0,
    ("W","W"): 11, ("W","Y"): 2,  ("W","V"):-3,
    ("Y","Y"): 7,  ("Y","V"):-1,
    ("V","V"): 4,
}

def blosum62(a1: str, a2: str) -> float | None:
    if a1 not in AA_MW or a2 not in AA_MW:
        return None
    key = (a1, a2) if (a1, a2) in _BLOSUM62_PAIRS else (a2, a1)
    return float(_BLOSUM62_PAIRS.get(key, np.nan))


def aa_features(row: pd.Series) -> dict:
    a1, a2 = row.get("AA_1"), row.get("AA_2")
    feats: dict = {
        "DELTA_HIDRO":       np.nan,
        "BLOSUM62":          np.nan,
        "SNV_PATHWAY_COUNT": np.nan,
        **{f"BAZ_{ch.replace('>','_')}": 0.0 for ch in ALL_CHANGES},
    }
    if not (isinstance(a1, str) and isinstance(a2, str)):
        return feats
    a1, a2 = a1.strip().upper(), a2.strip().upper()
    if a1 not in AA_MW or a2 not in AA_MW:
        return feats

    feats["DELTA_HIDRO"]       = AA_HIDRO[a2] - AA_HIDRO[a1]
    feats["BLOSUM62"]          = blosum62(a1, a2)
    changes, pathway_count     = infer_base_changes(a1, a2)
    feats["SNV_PATHWAY_COUNT"] = float(pathway_count)
    if len(changes) == 1:
        ch = next(iter(changes))
        key = f"BAZ_{ch.replace('>','_')}"
        if key in feats:
            feats[key] = 1.0
    return feats


def cat_pop_features(row: pd.Series) -> dict:
    val1 = row.get("CAT_1")
    is_exome = np.nan
    pop_nfe = 0.0
    pop_afr = 0.0
    if isinstance(val1, str):
        if "gnomADe" in val1:
            is_exome = 1.0
        elif "gnomADg" in val1:
            is_exome = 0.0
        v = val1.upper()
        if "NFE" in v:
            pop_nfe = 1.0
        if "AFR" in v:
            pop_afr = 1.0
    is_allofus = 1.0 if isinstance(row.get("CAT_2"), str) else 0.0
    val2 = row.get("CAT_2")
    if isinstance(val2, str) and "AFR" in val2.upper():
        pop_afr = 1.0
    return {"IS_EXOME": is_exome, "IS_ALLOFUS": is_allofus,
            "POP_NFE": pop_nfe, "POP_AFR": pop_afr}


def greedy_eleme(cols: list[str], corr_df: pd.DataFrame, threshold: float) -> set[str]:
    cols_set = set(cols)
    pairs = corr_df[
        (corr_df["feature1"].isin(cols_set)) &
        (corr_df["feature2"].isin(cols_set)) &
        (corr_df["spearman_r"].abs() >= threshold)
    ][["feature1", "feature2"]].values.tolist()

    adjacency: dict[str, set[str]] = {c: set() for c in cols_set}
    for f1, f2 in pairs:
        adjacency.setdefault(f1, set()).add(f2)
        adjacency.setdefault(f2, set()).add(f1)

    eliminated: set[str] = set()
    remaining = set(cols_set)
    while True:
        active = {c for c in remaining if adjacency.get(c, set()) & remaining - {c}}
        if not active:
            break
        target = max(active, key=lambda c: len(adjacency.get(c, set()) & remaining))
        eliminated.add(target)
        remaining.discard(target)
    return eliminated


def main() -> None:
    var_df = pd.read_csv(VAR_CSV)
    zero_var_cols = set(var_df[var_df["variance"] == 0]["column"].tolist())
    print(f"Sıfır varyans AL_ sütunları: {len(zero_var_cols)}")

    corr_df = pd.read_csv(CORR_CSV)
    print(f"Korelasyon tablosu: {len(corr_df)} çift (eşik={CORR_THRESHOLD})")

    for panel in PANELS:
        src = DATA_IN / f"YARISMA_TRAIN_{panel}.csv"
        if not src.exists():
            print(f"  [!] {panel} bulunamadı, atlanıyor")
            continue

        print(f"\n--- {panel} ---")
        df = pd.read_csv(src)
        print(f"  Giriş: {df.shape[0]} satır × {df.shape[1]} sütun")

        drop_zv = [c for c in zero_var_cols if c in df.columns]
        df.drop(columns=drop_zv, inplace=True)
        print(f"  Sıfır varyans silindi: {len(drop_zv)} sütun")

        if "CAT_6" in df.columns:
            df.drop(columns=["CAT_6"], inplace=True)
            print("  CAT_6 silindi")

        al_cols = [c for c in df.columns if c.startswith("AL_")]
        eliminated = greedy_eleme(al_cols, corr_df, CORR_THRESHOLD)
        df.drop(columns=list(eliminated), inplace=True)
        remaining_al = [c for c in df.columns if c.startswith("AL_")]
        print(f"  AL_ greedy eleme (r≥{CORR_THRESHOLD}): {len(eliminated)} silindi → {len(remaining_al)} AL_ kaldı")

        aa_feats = df.apply(aa_features, axis=1, result_type="expand")
        df = pd.concat([df, aa_feats], axis=1)

        pop_feats = df.apply(cat_pop_features, axis=1, result_type="expand")
        df = pd.concat([df, pop_feats], axis=1)

        dst = DATA_OUT / f"YARISMA_TRAIN_{panel}_temiz_r095.csv"
        df.to_csv(dst, index=False)
        print(f"  Çıkış: {df.shape[0]} satır × {df.shape[1]} sütun → {dst.name}")

    print("\nTamamlandı.")


if __name__ == "__main__":
    main()

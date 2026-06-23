"""
============================================================================
OZELLIK CIKARIMI (FEATURE EXTRACTION)  ->  DATA/Feature_extraction/
============================================================================
SYZ2026 / MedicalVision - Missense Genetik Varyant Siniflandirma

Uretilen ozellikler (spesifikasyon):
  1. is_exome      : CAT_1 'gnomADe*' -> 1, 'gnomADg*' -> 0, bos -> bos (NaN)
  2. CAT_1_te      : CAT_1 Target Encoding (boslar da ayri kategori olarak)
  3. CAT_2_te      : CAT_2 Target Encoding (boslar da ayri kategori olarak)
  4. is_segdup     : CAT_6 'segdup' iceriyorsa 1, aksi halde 0
  5. AL_185_present: AL_185 dolu ise 1, bos ise 0
  6. AA_1_te       : AA_1 Target Encoding
  7. AA_2_te       : AA_2 Target Encoding
  8. Delta_MW      : AA_2 - AA_1 molekuler agirlik farki
  9. Delta_PI      : AA_2 - AA_1 izoelektrik nokta farki
 10. Delta_HYDRO   : AA_2 - AA_1 hidrofobisite (Kyte-Doolittle) farki
 11. Delta_Charge  : AA_2 - AA_1 yan zincir yuk farki
 12. ek7_x_ek9     : EK_7 * EK_9
 13. ek2_ek3       : EK_2 - EK_3
 14. A, T, C, G    : CAT_3/CAT_4/CAT_5'te ilgili nukleotid varsa 1, yoksa 0
 15. has_archaic_delta: CAT_3, CAT_4, CAT_5 hepsi bos ise 0, degilse 1

Target Encoding sizinti olmamasi icin OUT-OF-FOLD (StratifiedKFold) +
smoothing ile yapilir.

Cikti: DATA/Feature_extraction/<VERISETI>.csv
Kullanim:
  python SRC/data_preprocessing/feature_extraction.py
  python SRC/data_preprocessing/feature_extraction.py --dataset MASTER
============================================================================
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import data_io as io

OUT_DIR = io.DATA_DIR / "Feature_extraction"

# --------------------------------------------------------------------------
# Aminoasit fiziko-kimyasal ozellik tablolari (20 standart aminoasit)
# --------------------------------------------------------------------------
AA_MW = {  # molekuler agirlik (g/mol, serbest aminoasit)
    "G": 75.07, "A": 89.09, "S": 105.09, "P": 115.13, "V": 117.15,
    "T": 119.12, "C": 121.16, "L": 131.17, "I": 131.17, "N": 132.12,
    "D": 133.10, "Q": 146.15, "K": 146.19, "E": 147.13, "M": 149.21,
    "H": 155.16, "F": 165.19, "R": 174.20, "Y": 181.19, "W": 204.23,
}
AA_PI = {  # izoelektrik nokta (pI)
    "A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07, "E": 3.22,
    "Q": 5.65, "G": 5.97, "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74,
    "M": 5.74, "F": 5.48, "P": 6.30, "S": 5.68, "T": 5.60, "W": 5.89,
    "Y": 5.66, "V": 5.96,
}
AA_HYDRO = {  # hidrofobisite (Kyte-Doolittle)
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
AA_CHARGE = {  # yan zincir yuku (pH ~7)
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0, "Q": 0, "E": -1, "G": 0,
    "H": 0.1, "I": 0, "L": 0, "K": 1, "M": 0, "F": 0, "P": 0, "S": 0,
    "T": 0, "W": 0, "Y": 0, "V": 0,
}


# --------------------------------------------------------------------------
# Target Encoding (out-of-fold, sizintisiz)
# --------------------------------------------------------------------------
def target_encode_oof(cat: pd.Series, y: pd.Series, n_splits: int = 5,
                      smoothing: float = 10.0, seed: int = 42) -> pd.Series:
    """Kategorik kolonu hedef ortalamasi ile sayisallastirir.
    Bos degerler '__MISSING__' kategorisi olarak ele alinir (kodlanir).
    Sizintiyi onlemek icin StratifiedKFold out-of-fold + smoothing kullanir.
    """
    cat = cat.astype("string").fillna("__MISSING__")
    y = y.astype(float)
    global_mean = y.mean()
    out = pd.Series(np.nan, index=cat.index, dtype=float)

    valid = y.notna()
    # En kucuk sinif kac ornek -> fold sayisini guvenli sec
    min_class = int(y[valid].value_counts().min()) if valid.any() else 0
    splits = max(2, min(n_splits, min_class)) if min_class >= 2 else 0

    if splits >= 2:
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        for tr, va in skf.split(cat[valid], y[valid]):
            idx = cat[valid].index
            tr_idx, va_idx = idx[tr], idx[va]
            stats = y.loc[tr_idx].groupby(cat.loc[tr_idx]).agg(["mean", "count"])
            enc = (stats["count"] * stats["mean"] + smoothing * global_mean) \
                / (stats["count"] + smoothing)
            out.loc[va_idx] = cat.loc[va_idx].map(enc).fillna(global_mean).values
    else:
        # Cok kucuk veri: smoothing'li tam-veri kodlamasi (fallback)
        stats = y[valid].groupby(cat[valid]).agg(["mean", "count"])
        enc = (stats["count"] * stats["mean"] + smoothing * global_mean) \
            / (stats["count"] + smoothing)
        out = cat.map(enc).fillna(global_mean)

    # Hedefi olmayan satirlar -> global ortalama
    out = out.fillna(global_mean)
    return out


# --------------------------------------------------------------------------
def aa_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Aminoasit kolonunu temizleyip buyuk harfe cevirir."""
    return io.col_or_na(df, col).astype("string").str.strip().str.upper()


def delta(a1: pd.Series, a2: pd.Series, table: dict) -> pd.Series:
    """AA_2 - AA_1 ozellik farki (bilinmeyen/bos aminoasit -> NaN)."""
    return a2.map(table).astype(float) - a1.map(table).astype(float)


def build_features(name: str) -> pd.DataFrame:
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    df = io.load_raw(name)
    feats = pd.DataFrame(index=df.index)
    feats[io.ID_COL] = df[io.ID_COL] if io.ID_COL in df.columns else pd.NA

    has_target = io.TARGET in df.columns
    y = df[io.TARGET] if has_target else None

    # 1) is_exome (CAT_1)
    c1 = io.col_or_na(df, "CAT_1").astype("string")
    is_exome = pd.Series(pd.NA, index=df.index, dtype="Int64")
    is_exome[c1.str.contains("gnomADe", case=False, na=False)] = 1
    is_exome[c1.str.contains("gnomADg", case=False, na=False)] = 0
    feats["is_exome"] = is_exome  # boslar NaN kalir

    # 2-3) CAT_1, CAT_2 Target Encoding (boslar dahil)
    if has_target:
        feats["CAT_1_te"] = target_encode_oof(io.col_or_na(df, "CAT_1"), y)
        feats["CAT_2_te"] = target_encode_oof(io.col_or_na(df, "CAT_2"), y)

    # 4) is_segdup (CAT_6)
    c6 = io.col_or_na(df, "CAT_6").astype("string")
    feats["is_segdup"] = c6.str.contains("segdup", case=False, na=False).astype(int)

    # 5) AL_185 dolu/bos -> 1/0
    feats["AL_185_present"] = io.col_or_na(df, "AL_185").notna().astype(int)

    # 6-7) AA Target Encoding
    aa1 = aa_series(df, "AA_1")
    aa2 = aa_series(df, "AA_2")
    if has_target:
        feats["AA_1_te"] = target_encode_oof(aa1, y)
        feats["AA_2_te"] = target_encode_oof(aa2, y)

    # 8-11) Aminoasit Delta ozellikleri (AA_2 - AA_1)
    feats["Delta_MW"] = delta(aa1, aa2, AA_MW)
    feats["Delta_PI"] = delta(aa1, aa2, AA_PI)
    feats["Delta_HYDRO"] = delta(aa1, aa2, AA_HYDRO)
    feats["Delta_Charge"] = delta(aa1, aa2, AA_CHARGE)

    # 12-13) EK etkilesim ozellikleri
    ek7 = pd.to_numeric(io.col_or_na(df, "EK_7"), errors="coerce")
    ek9 = pd.to_numeric(io.col_or_na(df, "EK_9"), errors="coerce")
    ek2 = pd.to_numeric(io.col_or_na(df, "EK_2"), errors="coerce")
    ek3 = pd.to_numeric(io.col_or_na(df, "EK_3"), errors="coerce")
    feats["ek7_x_ek9"] = ek7 * ek9
    feats["ek2_ek3"] = ek2 - ek3

    # 14) Nukleotid varligi (CAT_3/4/5 birlesik)
    cat345 = pd.concat(
        [io.col_or_na(df, c).astype("string").fillna("") for c in
         ("CAT_3", "CAT_4", "CAT_5")],
        axis=1,
    )
    joined = cat345.agg("".join, axis=1).str.upper()
    for nuc in ("A", "T", "C", "G"):
        feats[nuc] = joined.str.contains(nuc, regex=False).astype(int)

    # 15) has_archaic_delta: CAT_3/4/5 hepsi bos -> 0, degilse 1
    cat345_raw = pd.concat(
        [io.col_or_na(df, c) for c in ("CAT_3", "CAT_4", "CAT_5")], axis=1
    )
    feats["has_archaic_delta"] = cat345_raw.notna().any(axis=1).astype(int)

    # Hedefi en sona ekle
    if has_target:
        feats[io.TARGET] = y

    print(f"  Uretilen ozellik sayisi (ID/Label haric): "
          f"{feats.shape[1] - (2 if has_target else 1)}")
    return feats


def main():
    p = argparse.ArgumentParser(description="Ozellik cikarimi")
    p.add_argument("--dataset", choices=list(io.DATASETS), help="Tek veri seti")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [args.dataset] if args.dataset else list(io.DATASETS)
    for name in targets:
        try:
            feats = build_features(name)
            out_path = OUT_DIR / f"{name}.xlsx"
            feats.to_excel(out_path, index=False)
            print(f"  [yazildi] {out_path.relative_to(io.PROJECT_ROOT)}  "
                  f"(shape: {feats.shape})")
        except Exception as e:
            print(f"  [HATA] {name}: {e}")
    print(f"\nTamamlandi. Cikti: {OUT_DIR.relative_to(io.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

"""
Ortak veri okuma yardimcilari (SYZ2026 / MedicalVision).

Tum on isleme adimlari (imputation, feature_extraction) ham veriyi buradan
yukler. Kolon turleri ISME gore degil ICERIGE gore belirlenir:
CAT_* kolonlari kategorik (gnomADe_*, genotip vb.), AL_*/EK_* sayisaldir.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "DATA"
RAW_DIR = DATA_DIR / "Raw"

# Veri seti -> xlsx dosya adi
# (Veriler CSV'den XLSX'e cevrildi; boylece EK sutunlarindaki ondalik
#  nokta bozulmasi giderildi ve ayrac/ondalik ayarina gerek kalmadi.)
DATASETS = {
    "MASTER": "YARISMA_TRAIN_MASTER.xlsx",
    "CFTR": "YARISMA_TRAIN_CFTR.xlsx",
    "KANSER": "YARISMA_TRAIN_KANSER.xlsx",
    "PAH": "YARISMA_TRAIN_PAH.xlsx",
}

ID_COL = "Variant_ID"
TARGET = "Label"
NUMERIC_THRESHOLD = 0.5  # dolu degerlerin bu orani sayiya cevrilebiliyorsa -> sayisal
NA_TOKENS = ["", " ", "NA", "NaN", "nan", "null", "None"]


def load_raw(name: str) -> pd.DataFrame:
    """Ham XLSX'i yukler ve kolon turlerini icerige gore belirler
    (sayisal vs kategorik)."""
    fpath = RAW_DIR / DATASETS[name]
    if not fpath.exists():
        raise FileNotFoundError(f"Veri bulunamadi: {fpath}")

    df = pd.read_excel(fpath, na_values=NA_TOKENS, keep_default_na=True)
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("Int64")

    for col in df.columns:
        if col in (ID_COL, TARGET):
            continue
        raw = df[col]
        if pd.api.types.is_numeric_dtype(raw):
            continue
        s = raw.astype("string").str.strip().replace({"": pd.NA})
        n_nonnull = int(s.notna().sum())
        n_numeric = int(pd.to_numeric(s, errors="coerce").notna().sum())
        if n_nonnull > 0 and n_numeric / n_nonnull >= NUMERIC_THRESHOLD:
            df[col] = pd.to_numeric(s, errors="coerce")
        else:
            df[col] = s
    return df


def al_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("AL_")]


def ek_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("EK_")]


def col_or_na(df: pd.DataFrame, name: str) -> pd.Series:
    """Kolon varsa dondurur, yoksa tamami NaN olan seri dondurur."""
    if name in df.columns:
        return df[name]
    return pd.Series(pd.NA, index=df.index, name=name)

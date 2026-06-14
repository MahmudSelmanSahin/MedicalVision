from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TEKNOFEST_ROOT = Path("/Users/mahmudselmansahin/Teknofest")
DATA_ROOT = TEKNOFEST_ROOT / "VERİLER"
ORIGINAL_DIR = DATA_ROOT / "ORİJİNAL"
ID_COL = "Variant_ID"
LABEL_COL = "Label"
MISSING_TOKEN = "__MISSING__"
HIGH_MISSING_THRESHOLD = 0.90

PANEL_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}


VARIANT_FOLDERS = {
    "original": "ORİJİNAL",
    "drop_missing": "EKSİK VERİ SİLİNMİŞ",
    "imputed": "DOLDURULMUŞ",
    "normalized": "NORMALİZE",
    "normalized_imputed": "NORMALİZE + DOLDURULMUŞ",
    "normalized_drop_missing": "NORMALİZE + EKSİK VERİ SİLİNMİŞ",
    "imputed_drop_missing": "DOLDURULMUŞ + EKSİK VERİ SİLİNMİŞ",
    "normalized_drop_missing_imputed": "NORMALİZE + EKSİK VERİ SİLİNMİŞ + DOLDURULMUŞ",
    "synthetic": "SENTETİK VERİ ÜRETİLMİŞ",
    "synthetic_repeated": "SENTETİK VERİLERLE YUKARIDAKİ İŞLEMLER TEKRAR",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in {ID_COL, LABEL_COL}]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    prefixed = [col for col in df.columns if col.startswith("CAT_") or col.startswith("AA_")]
    object_cols = [
        col
        for col in feature_columns(df)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]
    return sorted(set(prefixed + object_cols), key=list(df.columns).index)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cats = set(categorical_columns(df))
    return [
        col
        for col in feature_columns(df)
        if col not in cats and pd.api.types.is_numeric_dtype(df[col])
    ]


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    cols: list[str] = []
    if ID_COL in df.columns:
        cols.append(ID_COL)
    cols.extend(col for col in df.columns if col not in {ID_COL, LABEL_COL})
    if LABEL_COL in df.columns:
        cols.append(LABEL_COL)
    return df.loc[:, cols].copy()


def drop_high_missing(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in feature_columns(df)
        if df[col].isna().mean() >= HIGH_MISSING_THRESHOLD
    ]
    return df.drop(columns=drop_cols)


def impute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cat_cols = categorical_columns(out)
    num_cols = numeric_columns(out)
    al_cols = [col for col in num_cols if col.startswith("AL_")]
    other_num_cols = [col for col in num_cols if col not in al_cols]

    if al_cols:
        out[al_cols] = out[al_cols].fillna(0)
    for col in other_num_cols:
        value = out[col].median()
        if pd.isna(value):
            value = 0.0
        out[col] = out[col].fillna(value)
    for col in cat_cols:
        out[col] = out[col].fillna(MISSING_TOKEN).astype(str)
    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = numeric_columns(out)
    al_cols = [col for col in num_cols if col.startswith("AL_")]
    if al_cols:
        out[al_cols] = np.log1p(out[al_cols])

    num_cols = numeric_columns(out)
    if num_cols:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])
    return out


def build_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dropped = drop_high_missing(df)
    return {
        "original": df.copy(),
        "drop_missing": dropped,
        "imputed": impute(df),
        "normalized": normalize(df),
        "normalized_imputed": normalize(impute(df)),
        "normalized_drop_missing": normalize(dropped),
        "imputed_drop_missing": impute(dropped),
        "normalized_drop_missing_imputed": normalize(impute(dropped)),
    }


def write_variant_csvs() -> dict[str, object]:
    report: dict[str, object] = {}
    for folder_name in VARIANT_FOLDERS.values():
        (DATA_ROOT / folder_name).mkdir(parents=True, exist_ok=True)

    for panel, filename in PANEL_FILES.items():
        source_path = ORIGINAL_DIR / filename
        df = pd.read_csv(source_path)
        variants = build_variants(df)

        for variant_key, variant_df in variants.items():
            folder = DATA_ROOT / VARIANT_FOLDERS[variant_key]
            output_path = folder / f"{panel}_{variant_key}.csv"
            ordered(variant_df).to_csv(output_path, index=False)

    for filename in PANEL_FILES.values():
        source_path = ORIGINAL_DIR / filename
        target_path = DATA_ROOT / VARIANT_FOLDERS["original"] / filename
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)

    for variant_key, folder_name in VARIANT_FOLDERS.items():
        folder = DATA_ROOT / folder_name
        csv_files = sorted(path.name for path in folder.glob("*.csv"))
        report[folder_name] = {
            "csv_count": len(csv_files),
            "csv_files": csv_files,
            "path": str(folder),
        }
    return report


def main() -> None:
    report = write_variant_csvs()
    report_path = DATA_ROOT / "VERI_KLASORLERI_RAPORU.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

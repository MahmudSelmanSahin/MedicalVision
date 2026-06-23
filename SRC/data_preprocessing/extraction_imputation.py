"""
============================================================================
BIRLESIK VERI (IMPUTATION + FEATURE EXTRACTION)  ->  DATA/Extraction-imputation/
============================================================================
SYZ2026 / MedicalVision - Missense Genetik Varyant Siniflandirma

Amac:
  Once imputation (AL MinMax + EK clip/scale), sonra feature_extraction
  (turetilen 18 ozellik) yapilmis ciktilari Variant_ID uzerinden
  birlestirip modele hazir tek dosya uretmek.

Girdi:
  - DATA/Imputation/<VERISETI>/<VERISETI>_ek-<scaler>_<senaryo>.csv
  - DATA/Feature_extraction/<VERISETI>.csv

Cikti (kolon sirasi: Variant_ID, AL_*, EK_*, <turetilen ozellikler>, Label):
  DATA/Extraction-imputation/<VERISETI>/<VERISETI>_ek-<scaler>_<senaryo>.csv

Her veri seti icin 4 imputation varyantinin (robust/standard x median/nan)
hepsi birlestirilir.

Kullanim:
  python SRC/data_preprocessing/extraction_imputation.py
  python SRC/data_preprocessing/extraction_imputation.py --dataset MASTER
============================================================================
"""
from __future__ import annotations

import argparse

import pandas as pd

import data_io as io

IMP_DIR = io.DATA_DIR / "Imputation"
FE_DIR = io.DATA_DIR / "Feature_extraction"
OUT_DIR = io.DATA_DIR / "Extraction-imputation"

VARIANTS = [f"ek-{s}_{m}" for s in ("robust", "standard") for m in ("median", "nan")]


def drop_duplicate_columns(df: pd.DataFrame, protect: list[str]):
    """Tipatip ayni (degerleri birebir esit) sutunlardan birini referans
    olarak birakir, digerlerini siler. NaN'lar esit kabul edilir.
    protect: silinmeyecek kolonlar (Variant_ID, Label)."""
    feats = [c for c in df.columns if c not in protect]
    seen: dict[bytes, str] = {}
    dropped: dict[str, str] = {}  # silinen -> referans
    for c in feats:
        # NaN'lari tutarli kodlayan hash imzasi
        key = pd.util.hash_pandas_object(df[c], index=False).values.tobytes()
        if key in seen and df[c].equals(df[seen[key]]):
            dropped[c] = seen[key]
        elif key not in seen:
            seen[key] = c
    kept = df.drop(columns=list(dropped))
    return kept, dropped


def merge_one(name: str, variant: str):
    imp_path = IMP_DIR / name / f"{name}_{variant}.xlsx"
    fe_path = FE_DIR / f"{name}.xlsx"
    if not imp_path.exists() or not fe_path.exists():
        print(f"  [atlandi] eksik girdi: {variant}")
        return None, None

    imp = pd.read_excel(imp_path)
    fe = pd.read_excel(fe_path)

    # Feature tarafindaki Label'i at (imputation'daki ile ayni) -> tek Label
    fe_no_label = fe.drop(columns=[io.TARGET], errors="ignore")

    merged = imp.merge(fe_no_label, on=io.ID_COL, how="inner", validate="1:1")

    # Kolon sirasi: Variant_ID, AL_*, EK_* (imputation), turetilenler, Label
    label = [io.TARGET] if io.TARGET in merged.columns else []
    ordered = [io.ID_COL] + [c for c in merged.columns
                             if c not in ([io.ID_COL] + label)] + label
    merged = merged[ordered]

    # Tipatip ayni sutunlari sil (Variant_ID ve Label korunur)
    merged, dropped = drop_duplicate_columns(merged, protect=[io.ID_COL] + label)
    return merged, dropped


def process_dataset(name: str):
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped_report = None
    for variant in VARIANTS:
        merged, dropped = merge_one(name, variant)
        if merged is None:
            continue
        out_path = out_dir / f"{name}_{variant}.xlsx"
        merged.to_excel(out_path, index=False)
        print(f"  [yazildi] {out_path.name}  (shape: {merged.shape}, "
              f"silinen ayni sutun: {len(dropped)})")
        dropped_report = dropped  # varyantlar arasi ayni; sonuncuyu sakla

    # Silinen "tipatip ayni" sutunlarin kaydi (referansiyla birlikte)
    if dropped_report:
        rep = pd.DataFrame(
            sorted(dropped_report.items()),
            columns=["silinen_sutun", "birakilan_referans"],
        )
        rep.to_excel(out_dir / f"{name}_silinen_ayni_sutunlar.xlsx", index=False)
        print(f"  [rapor]   {name}_silinen_ayni_sutunlar.xlsx ({len(rep)} sutun)")


def main():
    p = argparse.ArgumentParser(description="Imputation + feature birlestirme")
    p.add_argument("--dataset", choices=list(io.DATASETS), help="Tek veri seti")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [args.dataset] if args.dataset else list(io.DATASETS)
    for name in targets:
        try:
            process_dataset(name)
        except Exception as e:
            print(f"  [HATA] {name}: {e}")
    print(f"\nTamamlandi. Cikti: {OUT_DIR.relative_to(io.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

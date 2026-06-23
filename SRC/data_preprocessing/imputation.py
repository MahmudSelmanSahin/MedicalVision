"""
============================================================================
IMPUTATION / OLCEKLEME ADIMI  ->  DATA/Imputation/
============================================================================
SYZ2026 / MedicalVision - Missense Genetik Varyant Siniflandirma

Spesifikasyon:
  * AL_* kolonlari  : MinMaxScaler ile 0-1 araligina sikistirilir.
  * EK_* kolonlari  : ONCE Clipping (IQR baskilama) ile aykiri degerler
                      baskilanir, SONRA Robust veya StandardScaler ile
                      normalize edilir. Eksik degerler icin IKI senaryo:
                        - "median": eksikler medyan ile doldurulur
                        - "nan"   : eksikler NaN birakilir
                      (XGBoost/LightGBM/CatBoost NaN'i dogal isleyebilir.)

Notlar:
  - sklearn olcekleyiciler NaN'i fit sirasinda yok sayar, transform'da korur;
    bu sayede "nan" senaryosu sorunsuz calisir.
  - AL olcekleme her senaryoda aynidir (bir kez hesaplanir).

Cikti: DATA/Imputation/<VERISETI>/<VERISETI>_ek-<scaler>_<senaryo>.csv
       Kolonlar: Variant_ID, AL_* (minmax), EK_* (clip+scale), Label

Kullanim:
  python SRC/data_preprocessing/imputation.py
  python SRC/data_preprocessing/imputation.py --dataset MASTER \
         --ek-scaler robust --ek-missing nan
============================================================================
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import data_io as io

OUT_DIR = io.DATA_DIR / "Imputation"


def clip_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """IQR 1.5x kuralina gore aykiri degerleri alt/ust sinira baskilar
    (winsorization). NaN korunur (quantile NaN'i yok sayar)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return s
    return s.clip(lower=q1 - k * iqr, upper=q3 + k * iqr)


def scale_frame(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Robust veya Standard olcekleme. NaN korunur."""
    if frame.empty:
        return frame
    scaler = RobustScaler() if kind == "robust" else StandardScaler()
    arr = scaler.fit_transform(frame)
    return pd.DataFrame(arr, columns=frame.columns, index=frame.index)


def process_dataset(name: str, ek_scalers: list[str], missing_modes: list[str]):
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    df = io.load_raw(name)
    al = io.al_cols(df)
    ek = io.ek_cols(df)
    print(f"  AL kolon: {len(al)}, EK kolon: {len(ek)}")

    # --- AL: once eksikler 0 ile doldurulur, sonra MinMax 0-1 olceklenir ---
    if al:
        al_filled = df[al].fillna(0)
        al_scaled = pd.DataFrame(
            MinMaxScaler().fit_transform(al_filled), columns=al, index=df.index
        )
    else:
        al_scaled = pd.DataFrame(index=df.index)

    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for scaler_kind in ek_scalers:
        for mode in missing_modes:
            ek_df = df[ek].copy() if ek else pd.DataFrame(index=df.index)
            if ek:
                if mode == "median":
                    ek_df = ek_df.fillna(ek_df.median(numeric_only=True))
                # clip (her sutun icin) -> scale
                ek_df = ek_df.apply(clip_iqr)
                ek_df = scale_frame(ek_df, scaler_kind)

            parts = [df[[io.ID_COL]], al_scaled, ek_df]
            if io.TARGET in df.columns:
                parts.append(df[[io.TARGET]])
            out = pd.concat(parts, axis=1)

            fname = f"{name}_ek-{scaler_kind}_{mode}.xlsx"
            out.to_excel(out_dir / fname, index=False)
            n_nan = int(out[ek].isna().sum().sum()) if ek else 0
            print(f"  [yazildi] {fname}  (EK NaN: {n_nan}, shape: {out.shape})")


def main():
    p = argparse.ArgumentParser(description="Imputation / olcekleme")
    p.add_argument("--dataset", choices=list(io.DATASETS), help="Tek veri seti")
    p.add_argument("--ek-scaler", choices=["robust", "standard", "both"],
                   default="both", help="EK olcekleyici (vars: both)")
    p.add_argument("--ek-missing", choices=["median", "nan", "both"],
                   default="both", help="EK eksik deger senaryosu (vars: both)")
    args = p.parse_args()

    scalers = ["robust", "standard"] if args.ek_scaler == "both" else [args.ek_scaler]
    modes = ["median", "nan"] if args.ek_missing == "both" else [args.ek_missing]
    targets = [args.dataset] if args.dataset else list(io.DATASETS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in targets:
        try:
            process_dataset(name, scalers, modes)
        except Exception as e:
            print(f"  [HATA] {name}: {e}")
    print(f"\nTamamlandi. Cikti: {OUT_DIR.relative_to(io.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

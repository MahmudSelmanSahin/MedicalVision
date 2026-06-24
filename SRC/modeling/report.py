"""
===========================================================================
RAPOR - En iyi performans (model x panel) Excel ureticisi
===========================================================================
SYZ2026 / MedicalVision

Cikti: best_per_panel.xlsx
  * Sheet 'best_per_model_panel': her (model, panel) icin EN IYI kosu
    (senaryo + hiperparametre + tum metrikler). Siralama metrigi: MCC.
  * Sheet 'all_runs': tum kosular (basarili/atlanan/hatali).
  * Sheet 'best_per_panel': her panel icin tum modeller arasi en iyi.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RANK_METRIC = "mcc"
METRIC_COLS = ["threshold", "f1", "macro_f1", "mcc", "precision", "recall",
               "auc", "accuracy", "cv_mcc_mean", "cv_mcc_std"]


def build_best_per_panel(df: pd.DataFrame, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = df[df["status"] == "ok"].copy()
    abl_cols = [c for c in df.columns if c.startswith("abl_")]
    keep = (["panel", "model", "scenario", "hpo", "data_aug", "run_id", "best_hp"]
            + abl_cols + [c for c in METRIC_COLS if c in ok.columns])
    keep = [c for c in keep if c in ok.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        if not ok.empty and RANK_METRIC in ok.columns:
            idx = ok.groupby(["model", "panel"])[RANK_METRIC].idxmax()
            best_mp = ok.loc[idx, keep].sort_values(
                ["panel", RANK_METRIC], ascending=[True, False])
            best_mp.to_excel(xl, sheet_name="best_per_model_panel", index=False)

            idx2 = ok.groupby(["panel"])[RANK_METRIC].idxmax()
            ok.loc[idx2, keep].sort_values(RANK_METRIC, ascending=False) \
                .to_excel(xl, sheet_name="best_per_panel", index=False)
        else:
            pd.DataFrame({"info": ["basarili kosu yok"]}).to_excel(
                xl, sheet_name="best_per_model_panel", index=False)

        df.to_excel(xl, sheet_name="all_runs", index=False)

    print(f"[rapor] {out_path}  (basarili kosu: {len(ok)}/{len(df)})")
    return out_path

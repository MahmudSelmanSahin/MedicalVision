"""
===========================================================================
RAPOR - En iyi performans (model x panel) Excel ureticisi
===========================================================================
SYZ2026 / MedicalVision

Cikti: best_per_panel.xlsx
  * Sheet 'best_per_model_panel': her (model, panel) icin EN IYI kosu
    (senaryo + hiperparametre + tum metrikler).
  * Sheet 'best_per_panel': her panel icin tum modeller arasi en iyi.
  * Sheet 'ablasyon_ozeti': her eksenin izole (eslesmis) etkisi.
  * Sheet 'all_runs': tum kosular (basarili/atlanan/hatali).

EN IYI secimi OOF (cv_mcc_mean) ile yapilir; TEST metrigine gore SECILMEZ
(test'e gore secim = selection-bias). Test metrikleri tabloda yalnizca
nihai raporlama icin gosterilir. cv_mcc_mean yoksa mcc'ye dusulur.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RANK_METRIC = "macro_f1"       # tabloda gosterilen test metrigi
# Secim olcutu: SIZINTISIZ OOF metrikleri tercih sirasi. cv_macro_f1 varsa onu,
# yoksa cv_mcc_mean'i kullan; test metrigine ASLA dusme (selection-bias onlemi).
SELECT_PRIORITY = ["cv_macro_f1", "cv_mcc_mean"]
TIE_BREAK = "cv_mcc_mean"      # beraberlikte ikincil olcut
METRIC_COLS = ["threshold", "f1", "macro_f1", "mcc", "precision", "recall",
               "auc", "accuracy", "cv_macro_f1", "cv_mcc_mean", "cv_mcc_std"]
# Not: cv_macro_f1, runner'da OOF macro-F1 olarak kaydedilir; eski kosularda
# bulunmayabilir -> o durumda secim cv_mcc_mean ile yapilir (yine sizintisiz).

# Ablasyon ozeti icin: izole edilecek metrikler
SUMMARY_METRICS = ["mcc", "macro_f1", "auc"]


def _paired_delta(ok: pd.DataFrame, axis: str, val_on, val_off, label: str):
    """axis'i val_on vs val_off karsilastiran ESLESMIS delta (diger tum eksenler
    sabit). Donus: satir listesi {eksen, karsilastirma, panel, n, d_mcc, ...}."""
    id_cols = ["panel", "scenario", "model", "hpo", "data_aug"]
    id_cols += [c for c in ok.columns if c.startswith("abl_")]
    group_cols = [c for c in id_cols if c != axis and c in ok.columns]
    if axis not in ok.columns:
        return []
    rows = []
    for keys, g in ok.groupby(group_cols, dropna=False):
        on = g[g[axis] == val_on]
        off = g[g[axis] == val_off]
        if on.empty or off.empty:
            continue
        panel = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))["panel"]
        rec = {"eksen": label, "karsilastirma": f"{val_on} - {val_off}",
               "panel": panel, "n_cift": 1}
        for met in SUMMARY_METRICS:
            if met in g.columns:
                rec[f"delta_{met}"] = float(on[met].mean() - off[met].mean())
        rows.append(rec)
    return rows


def build_ablation_summary(ok: pd.DataFrame) -> pd.DataFrame:
    """Her ekseni izole eden ESLESMIS etki tablosu (panel + GENEL).
    'delta_*' pozitifse o yontem metrigi artirmis demektir."""
    rows = []
    # 1) on/off ablasyon bayraklari
    for c in [c for c in ok.columns if c.startswith("abl_")]:
        if {"on", "off"} & set(ok[c].dropna().unique()):
            rows += _paired_delta(ok, c, "on", "off", c.replace("abl_", ""))
    # 2) data_aug: her varyant vs original
    if "data_aug" in ok.columns:
        for v in [v for v in ok["data_aug"].dropna().unique() if v != "original"]:
            rows += _paired_delta(ok, "data_aug", v, "original", "data_aug")
    # 3) prior esik oncesi/sonrasi (dogrudan sutunlardan)
    if {"mcc_thr_prior", "mcc_thr_default"}.issubset(ok.columns):
        for panel, g in ok.groupby("panel"):
            rec = {"eksen": "prior_threshold", "karsilastirma": "prior - default",
                   "panel": panel, "n_cift": len(g),
                   "delta_mcc": float((g["mcc_thr_prior"] - g["mcc_thr_default"]).mean())}
            if {"macro_f1_thr_prior", "macro_f1_thr_default"}.issubset(ok.columns):
                rec["delta_macro_f1"] = float(
                    (g["macro_f1_thr_prior"] - g["macro_f1_thr_default"]).mean())
            rows.append(rec)

    if not rows:
        return pd.DataFrame({"info": ["ablasyon ozeti icin yeterli kosu yok"]})

    long = pd.DataFrame(rows)
    delta_cols = [c for c in long.columns if c.startswith("delta_")]
    # panel bazli ortalama + cift sayisi
    per_panel = (long.groupby(["eksen", "karsilastirma", "panel"], as_index=False)
                 .agg({**{c: "mean" for c in delta_cols}, "n_cift": "sum"}))
    # GENEL (tum paneller)
    overall = (long.groupby(["eksen", "karsilastirma"], as_index=False)
               .agg({**{c: "mean" for c in delta_cols}, "n_cift": "sum"}))
    overall["panel"] = "GENEL"
    out = pd.concat([overall, per_panel], ignore_index=True)
    for c in delta_cols:
        out[c] = out[c].round(4)
    cols = ["eksen", "karsilastirma", "panel", "n_cift"] + delta_cols
    return out[cols].sort_values(["eksen", "panel"]).reset_index(drop=True)


def build_best_per_panel(df: pd.DataFrame, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = df[df["status"] == "ok"].copy()
    abl_cols = [c for c in df.columns if c.startswith("abl_")]
    keep = (["panel", "model", "scenario", "hpo", "data_aug", "run_id", "best_hp"]
            + abl_cols + [c for c in METRIC_COLS if c in ok.columns])
    keep = [c for c in keep if c in ok.columns]

    # EN IYI'yi OOF ile sec (test'e gore DEGIL -> selection-bias yok).
    # Beraberlik TIE_BREAK ile bozulur (ör. cv_macro_f1 esitse cv_mcc_mean'e bak).
    # ONEMLI: bir OOF metrigi ancak TUM kosularda doluysa secime kullanilir;
    # aksi halde (ör. karisik matriste cv_macro_f1 sadece yeni kosularda varsa)
    # NaN'li modeller haksizca elenir -> her zaman dolu olan cv_mcc_mean'e dusulur.
    def _usable(c):
        return c in ok.columns and not ok.empty and ok[c].notna().mean() >= 0.99
    sel = next((c for c in SELECT_PRIORITY if _usable(c)),
               next((c for c in SELECT_PRIORITY if c in ok.columns), RANK_METRIC))
    tie = TIE_BREAK if TIE_BREAK in ok.columns else sel
    ok_sorted = ok.sort_values([sel, tie], ascending=[False, False])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        if not ok.empty and sel in ok.columns:
            best_mp = ok_sorted.drop_duplicates(["model", "panel"])[keep] \
                .sort_values(["panel", sel], ascending=[True, False])
            best_mp.to_excel(xl, sheet_name="best_per_model_panel", index=False)

            ok_sorted.drop_duplicates(["panel"])[keep] \
                .sort_values(sel, ascending=False) \
                .to_excel(xl, sheet_name="best_per_panel", index=False)
        else:
            pd.DataFrame({"info": ["basarili kosu yok"]}).to_excel(
                xl, sheet_name="best_per_model_panel", index=False)

        # Ablasyon ozeti: her eksenin izole (eslesmis) etkisi
        try:
            build_ablation_summary(ok).to_excel(
                xl, sheet_name="ablasyon_ozeti", index=False)
        except Exception as e:
            pd.DataFrame({"info": [f"ablasyon ozeti uretilemedi: {e}"]}).to_excel(
                xl, sheet_name="ablasyon_ozeti", index=False)

        df.to_excel(xl, sheet_name="all_runs", index=False)

    print(f"[rapor] {out_path}  (basarili kosu: {len(ok)}/{len(df)})")
    return out_path

"""
===========================================================================
PANEL-OZEL MULTIMODEL (ensemble)  -  matris BITTIKTEN sonra calistirilir
===========================================================================
SYZ2026 / MedicalVision

Iki NotebookLM kaynaginin onerisi: model SECIMI test'e gore DEGIL, OOF
(cv_mcc_mean) skoruna gore yapilir; frozen test yalnizca nihai olcum icindir.

Akis (her panel):
  1. all_runs (veya _checkpoint.jsonl) icindeki BASARILI taban-model kosulari.
  2. cv_mcc_mean'e gore her modelin EN IYI konfigini al, ust K modeli sec.
  3. Her secilen modeli kendi (senaryo + data_aug + ablasyon + hp) ile yeniden
     kur; OOF (train) ve frozen-test olasiliklarini uret.
  4. SOFT VOTING (olasilik ortalamasi) + STACKING (meta=LogReg, OOF uzerinde)
     ensemble'lari kur; esik prior-farkinda (test prior'i) secilir.
  5. Tek-en-iyi modelle karsilastir; sonuc + grafik yaz.

Kullanim (matris bitince):
  python SRC/modeling/multimodel.py --top_k 3
===========================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "SRC" / "data_preprocessing"))

from data.augment import make_augment_fn                  # noqa: E402
from eval.metrics import compute_metrics, optimize_threshold  # noqa: E402
from eval.plots import plot_confusion, plot_roc           # noqa: E402
from models.registry import build_pipeline                # noqa: E402
from validation.cv import cv_evaluate                     # noqa: E402
import runner as R                                         # noqa: E402

BASE_EXCLUDE = {"voting", "stacking"}


def load_results():
    """all_runs.xlsx varsa onu, yoksa _checkpoint.jsonl'i okur."""
    runs_dir = ROOT / "SRC" / "Result" / "runs"
    xlsx = runs_dir / "all_runs.xlsx"
    if xlsx.exists():
        return pd.read_excel(xlsx)
    ck = runs_dir / "_checkpoint.jsonl"
    recs = [json.loads(l) for l in ck.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(recs)


def _ablation_from_rec(rec) -> dict:
    return {k[4:]: rec[k] for k in rec.index if k.startswith("abl_")}


def rebuild(rec, common, cache):
    """Bir kosu kaydindan: (split, pipeline, augment_fn) yeniden kurar."""
    panel, scen, model, aug = rec["panel"], rec["scenario"], rec["model"], rec["data_aug"]
    ablation = _ablation_from_rec(rec)
    hp = json.loads(rec.get("best_hp") or "{}")
    sp = R.get_split(panel, common, cache)
    if aug == "transfer_learning":
        sp = R.get_transfer_split(panel, sp, common, cache)
    eff = dict(ablation)
    if "synthetic" in aug:
        eff["smote"] = "on"
    augment_fn = None
    if "clustering" in aug:
        Xe, ye = R.get_candidates(panel, sp, common, cache)
        augment_fn = make_augment_fn(Xe, ye)
    pipe = build_pipeline(model, scen, eff, sp.y_train, seed=common["seed"], hp=hp)
    return sp, pipe, augment_fn


def panel_ensemble(panel, df, common, cache, top_k=3):
    ok = df[(df["panel"] == panel) & (df["status"] == "ok")
            & (~df["model"].isin(BASE_EXCLUDE))].copy()
    if ok.empty or "cv_mcc_mean" not in ok.columns:
        return None
    idx = ok.groupby("model")["cv_mcc_mean"].idxmax()      # her modelin en iyisi
    top = ok.loc[idx].sort_values("cv_mcc_mean", ascending=False).head(top_k)

    oof_list, test_list, y_tr, y_te = [], [], None, None
    seed, bf = common["seed"], common["test"]["benign_frac"]
    for _, rec in top.iterrows():
        sp, pipe, aug = rebuild(rec, common, cache)
        oof, _, _ = cv_evaluate(pipe, sp.X_train, sp.y_train,
                                n_splits=common["cv"]["n_splits"], n_repeats=1,
                                seed=seed, augment_fn=aug)
        if oof is None:
            continue
        if aug is not None:
            Xa, ya = aug(sp.X_train, sp.y_train); pipe.fit(Xa, ya)
        else:
            pipe.fit(sp.X_train, sp.y_train)
        oof_list.append(oof)
        test_list.append(pipe.predict_proba(sp.X_test)[:, 1])
        y_tr, y_te = np.asarray(sp.y_train), np.asarray(sp.y_test)

    if len(oof_list) < 2:
        return None
    O = np.vstack(oof_list)            # (K, n_train)
    T = np.vstack(test_list)           # (K, n_test)
    m = ~np.isnan(O).any(axis=0)       # tum modellerde OOF dolu satirlar

    out = {"panel": panel, "secilen_modeller": ";".join(top["model"]),
           "k": len(oof_list)}

    # SOFT VOTING
    v_oof, v_test = O.mean(axis=0), T.mean(axis=0)
    thr = optimize_threshold(y_tr[m], v_oof[m], "mcc", bf)
    vm = compute_metrics(y_te, v_test, thr)
    out.update(voting_mcc=vm["mcc"], voting_macro_f1=vm["macro_f1"], voting_auc=vm["auc"])

    # STACKING (meta = LogReg, OOF uzerinde egitilir)
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    meta.fit(O[:, m].T, y_tr[m])
    s_oof = meta.predict_proba(O[:, m].T)[:, 1]
    s_test = meta.predict_proba(T.T)[:, 1]
    thr2 = optimize_threshold(y_tr[m], s_oof, "mcc", bf)
    sm = compute_metrics(y_te, s_test, thr2)
    out.update(stack_mcc=sm["mcc"], stack_macro_f1=sm["macro_f1"], stack_auc=sm["auc"])

    # tek-en-iyi (OOF) karsilastirma
    out["best_single_mcc"] = float(top.iloc[0]["mcc"])
    out["best_single_model"] = top.iloc[0]["model"]

    # grafik (stacking)
    gdir = ROOT / "SRC" / "Result" / "Graphics" / "MULTIMODEL"
    plot_confusion(y_te, (s_test >= thr2).astype(int), gdir / f"{panel}_stacking_cm.png",
                   f"{panel} stacking")
    plot_roc(y_te, s_test, gdir / f"{panel}_stacking_roc.png", f"{panel} stacking")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=3, help="panel basina birlestirilecek model")
    args = ap.parse_args()

    common, _, _ = R.load_config(None)
    df = load_results()
    cache = {}
    rows = []
    for panel in sorted(df["panel"].dropna().unique()):
        print(f"[multimodel] {panel} ...")
        res = panel_ensemble(panel, df, common, cache, top_k=args.top_k)
        if res:
            rows.append(res)
    out = ROOT / "SRC" / "Result" / "multimodel.xlsx"
    pd.DataFrame(rows).to_excel(out, index=False)
    print(f"[multimodel] yazildi: {out}")


if __name__ == "__main__":
    main()

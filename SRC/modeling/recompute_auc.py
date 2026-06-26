"""
===========================================================================
SIZINTISIZ cv_auc + cv_pr_auc RECOMPUTE  (HPO'suz, kayitli best_hp ile)
===========================================================================
SYZ2026 / MedicalVision

Model SECIMI icin eşikten BAGIMSIZ metrik (PathoPredictor + NotebookLM onerisi):
  cv_auc    = ROC-AUC(OOF)      -> genel ayirt edicilik (birincil secim olcutu)
  cv_pr_auc = PR-AUC(OOF)       -> azinlik sinifi tespiti (ek olcut)

execute_run'un AYNI yardimci fonksiyonlarini kullanir (get_split, transfer,
clustering augment, build_pipeline, cv_evaluate); tek fark: HPO yapilmaz,
all_runs'taki kayitli best_hp kullanilir. Boylece hizli (~HPO maliyeti yok)
ve execute_run ile birebir tutarli kalir. Test setine DOKUNMAZ.

Kullanim:
  python SRC/modeling/recompute_auc.py
===========================================================================
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "SRC" / "data_preprocessing"))

import runner as R                                         # noqa: E402
from models.registry import build_pipeline                # noqa: E402
from validation.cv import cv_evaluate                     # noqa: E402
from data.augment import make_augment_fn                  # noqa: E402

ABL_FLAGS = ["focal_loss", "feature_selection", "class_weight", "l2", "smote",
             "feature_subsample", "row_subsample", "early_stopping"]

# Isci-basina panel cache (transfer/clustering'i panel basina bir kez hesaplar)
_CACHE: dict = {}


def _recompute_one(row: dict, common: dict):
    rid = row["run_id"]
    try:
        data_aug = row["data_aug"]
        model, scenario = row["model"], row["scenario"]
        ablation = {f: row.get(f"abl_{f}", "off") for f in ABL_FLAGS}
        bh = row.get("best_hp")
        best_hp = json.loads(bh) if isinstance(bh, str) and bh.strip() else {}
        seed = common["seed"]

        sp = R.get_split(row["panel"], common, _CACHE)
        if data_aug == "transfer_learning":
            sp = R.get_transfer_split(row["panel"], sp, common, _CACHE)

        abl = dict(ablation)
        if "synthetic" in data_aug:
            abl["smote"] = "on"
        augment_fn = None
        if "clustering" in data_aug:
            X_extra, y_extra = R.get_candidates(row["panel"], sp, common, _CACHE)
            augment_fn = make_augment_fn(X_extra, y_extra)

        pipe = build_pipeline(model, scenario, abl, sp.y_train, seed=seed, hp=best_hp)
        oof, _, _ = cv_evaluate(
            pipe, sp.X_train, sp.y_train,
            n_splits=common["cv"]["n_splits"], n_repeats=common["cv"]["n_repeats"],
            seed=seed, augment_fn=augment_fn)
        if oof is None:
            return rid, None, None
        mask = ~np.isnan(oof)
        yv, oofm = np.asarray(sp.y_train)[mask], oof[mask]
        if len(np.unique(yv)) < 2:
            return rid, None, None
        return (rid, round(float(roc_auc_score(yv, oofm)), 4),
                round(float(average_precision_score(yv, oofm)), 4))
    except Exception as e:
        return rid, f"ERR:{type(e).__name__}", None


def main():
    common, _, _ = R.load_config("first_pass")
    runs_dir = ROOT / "SRC" / "Result" / "runs"
    df = pd.read_excel(runs_dir / "all_runs.xlsx")
    ok = df[df["status"] == "ok"].copy()
    rows = ok.to_dict("records")
    print(f"recompute hedefi: {len(rows)} ok kosu (HPO yok, sadece CV->OOF->AUC)")

    workers = int(common.get("parallel_runs", 8) or 8)
    cv_auc, cv_pr = {}, {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_recompute_one, r, common) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            rid, a, p = fut.result()
            cv_auc[rid], cv_pr[rid] = a, p
            if i % 25 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] {rid}  cv_auc={a} cv_pr_auc={p}")

    df["cv_auc"] = df["run_id"].map(lambda r: cv_auc.get(r))
    df["cv_pr_auc"] = df["run_id"].map(lambda r: cv_pr.get(r))

    # numerik olmayan (ERR) degerleri NaN yap
    for c in ("cv_auc", "cv_pr_auc"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_excel(runs_dir / "all_runs.xlsx", index=False)
    (runs_dir / "all_runs.json").write_text(
        df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTamamlandi ({time.time()-t0:.1f}s). cv_auc dolu: "
          f"{int(df['cv_auc'].notna().sum())}/{len(df[df['status']=='ok'])}")

    # best_per_panel'i cv_auc ile yeniden uret
    try:
        from report import build_best_per_panel
        build_best_per_panel(df, ROOT / "SRC" / "Result" / "best_per_panel.xlsx")
    except Exception as e:
        print(f"[rapor uyarisi] {e}")


if __name__ == "__main__":
    main()

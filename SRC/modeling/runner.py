"""
===========================================================================
RUNNER - Ablasyon kosularini uretir, calistirir, sonuclari toplar
===========================================================================
SYZ2026 / MedicalVision

Akis (her kosu):
  1. Uygulanabilirlik kontrolu (capabilities) -> gecersizse ATLA
  2. Frozen klinik hold-out split (panel basina cache)
  3. HPO (train) -> best hp
  4. CV (train) -> OOF olasilik -> karar esigi kalibrasyonu
  5. Tum train'de refit -> TEST'te tahmin -> metrikler (esikli)
  6. Grafikler (confusion, roc, pr) + metrik JSON kaydet

Kullanim:
  python SRC/modeling/runner.py                 # active_profile (config)
  python SRC/modeling/runner.py --profile full
  python SRC/modeling/runner.py --limit 4       # ilk N kosu (debug)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "SRC" / "data_preprocessing"))

import data_io as io                                   # noqa: E402
from capabilities import CAPABILITIES, applicable      # noqa: E402
from data.augment import build_clustering_candidates, make_augment_fn  # noqa: E402
from data.splits import Split, clinical_holdout         # noqa: E402
from data.transfer import build_transfer_feature        # noqa: E402
from eval.explain import final_feature_names, native_importance, shap_summary  # noqa: E402
from eval.metrics import compute_metrics, optimize_threshold  # noqa: E402
from eval.plots import plot_confusion, plot_pr, plot_roc      # noqa: E402
from hpo.search import run_hpo                          # noqa: E402
from models.registry import build_pipeline             # noqa: E402
from validation.cv import cv_evaluate                  # noqa: E402

ABLATION_FLAGS = ["focal_loss", "feature_selection", "class_weight", "l2",
                  "smote", "feature_subsample", "row_subsample", "early_stopping"]


def load_config(profile: str | None):
    cfg = yaml.safe_load((HERE / "config" / "experiment.yaml").read_text(encoding="utf-8"))
    prof = profile or cfg["active_profile"]
    return cfg["common"], cfg["profiles"][prof], prof


def _norm_state(v) -> str:
    """YAML 'off'/'on' kelimelerini boolean'a cevirebilir; stringe geri getir."""
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v).strip().lower()


def _ablation_configs(abl: dict, mode: str):
    """Ablasyon konfiglerini uretir.
    'full' : tum bayraklarin kartezyeni (2^N).
    'ofat' : baseline (hepsi off) + her bayragi TEK TEK acan konfigler (1+N).
             Standart ablasyon; her eksenin izole etkisini olcer (eslesmis)."""
    abl_keys = list(abl)
    if mode == "full":
        for combo in itertools.product(*[abl[k] for k in abl_keys]):
            yield dict(zip(abl_keys, combo))
    else:  # ofat
        base = {k: "off" for k in abl_keys}
        yield dict(base)
        for k in abl_keys:
            if "on" in abl[k]:
                c = dict(base); c[k] = "on"
                yield c


def enumerate_runs(p: dict, ablation_mode: str = "ofat"):
    """Profil eksenlerinin kartezyen carpimi -> kosu sozlukleri."""
    abl = {k: [_norm_state(x) for x in vals] for k, vals in p["ablation"].items()}
    for panel, scen, model, hpo, aug in itertools.product(
            p["panels"], p["scenarios"], p["models"], p["hpo"], p["data_aug"]):
        for combo in _ablation_configs(abl, ablation_mode):
            yield {
                "panel": panel, "scenario": scen, "model": model,
                "hpo": hpo, "data_aug": aug,
                "ablation": combo,
            }


def run_id(r: dict) -> str:
    on = "+".join(k for k, v in r["ablation"].items() if v == "on") or "base"
    return f"{r['panel']}__{r['scenario']}__{r['model']}__{r['hpo']}__{r['data_aug']}__{on}"


def get_split(panel: str, common: dict, cache: dict):
    if panel not in cache:
        df = io.load_raw(panel)
        df = df.drop(columns=[c for c in [io.ID_COL] if c in df.columns])  # Variant_ID at
        df = df[df[io.TARGET].notna()].reset_index(drop=True)
        sp = clinical_holdout(
            df, benign_frac=common["test"]["benign_frac"],
            test_size=common["test"]["size"],
            train_benign_keep_frac=common["test"].get("train_benign_keep_frac", 0.50),
            balanced=panel in common["test"].get("balanced_panels", []),
            seed=common["seed"])
        cache[panel] = sp
    return cache[panel]


def get_master_raw(cache: dict):
    """MASTER ham (Variant_ID'siz, etiketli) -> (mX, my). Bir kez yuklenir."""
    if "master_raw" not in cache:
        mdf = io.load_raw("MASTER")
        mdf = mdf.drop(columns=[c for c in [io.ID_COL] if c in mdf.columns])
        mdf = mdf[mdf[io.TARGET].notna()].reset_index(drop=True)
        cache["master_raw"] = (mdf.drop(columns=[io.TARGET]), mdf[io.TARGET].astype(int))
    return cache["master_raw"]


def get_candidates(panel: str, sp, common: dict, cache: dict):
    """Panel icin kumeleme aday havuzu (MASTER'dan, test'ten ayrik). Panel
    basina bir kez hesaplanir. MASTER'in kendisi icin bos dondurur."""
    key = f"cand::{panel}"
    if key not in cache:
        if panel == "MASTER":
            cache[key] = (sp.X_train.iloc[0:0].copy(), sp.y_train.iloc[0:0].copy())
        else:
            mX, my = get_master_raw(cache)
            cache[key] = build_clustering_candidates(
                sp.X_train, sp.y_train, mX, my,
                exclude_X=sp.X_test, seed=common["seed"])
    return cache[key]


def get_transfer_split(panel: str, sp, common: dict, cache: dict):
    """transfer_learning: panel train/test'e TL_master_prob ozelligi eklenmis
    yeni bir Split dondurur (cache'lenmis split'i MUTASYONA UGRATMAZ)."""
    key = f"tl::{panel}"
    if key not in cache:
        mX, my = get_master_raw(cache)
        tl_tr, tl_te = build_transfer_feature(
            sp.X_train, sp.y_train, sp.X_test, mX, my, seed=common["seed"])
        cache[key] = (tl_tr.to_numpy(), tl_te.to_numpy())
    tl_tr, tl_te = cache[key]
    Xtr = sp.X_train.copy(); Xtr["TL_master_prob"] = tl_tr
    Xte = sp.X_test.copy();  Xte["TL_master_prob"] = tl_te
    return Split(Xtr, sp.y_train, Xte, sp.y_test, sp.info)


def execute_run(r: dict, common: dict, split_cache: dict) -> dict:
    rid = run_id(r)
    rec = {**{k: r[k] for k in ("panel", "scenario", "model", "hpo", "data_aug")},
           **{f"abl_{k}": v for k, v in r["ablation"].items()},
           "run_id": rid, "status": "ok"}

    ok, why = applicable(r["model"], r["ablation"])
    if not ok:
        rec.update(status="skipped", reason=why)
        return rec

    if r["data_aug"] == "transfer_learning" and r["panel"] == "MASTER":
        rec.update(status="skipped", reason="MASTER icin transfer learning anlamsiz")
        return rec

    seed = common["seed"]
    sp = get_split(r["panel"], common, split_cache)
    rec.update(train_n=sp.info["train"]["n"], test_n=sp.info["test"]["n"],
               test_benign_frac=sp.info["test_benign_frac"],
               low_train_minority=sp.info["low_train_minority"])

    # transfer_learning: panel'e TL_master_prob meta-ozelligi ekle (sizintisiz)
    if r["data_aug"] == "transfer_learning":
        sp = get_transfer_split(r["panel"], sp, common, split_cache)

    # --- Augmentasyon kurulumu (fold-ici, sizintisiz) ---
    data_aug = r["data_aug"]
    ablation = dict(r["ablation"])
    if "synthetic" in data_aug:        # SMOTE -> pipeline sampler (fold-ici)
        ablation["smote"] = "on"
    augment_fn = None
    if "clustering" in data_aug:
        X_extra, y_extra = get_candidates(r["panel"], sp, common, split_cache)
        augment_fn = make_augment_fn(X_extra, y_extra)
        rec["aug_added"] = int(len(X_extra))
        if len(X_extra) == 0 and r["panel"] != "MASTER":
            rec["aug_warn"] = "kumeleme aday bulamadi"

    try:
        # 3) HPO (augmentasyon HPO icinde de uygulanir)
        best_hp, hpo_score, _ = run_hpo(
            r["model"], r["scenario"], ablation, sp.X_train, sp.y_train,
            method=r["hpo"], seed=seed, augment_fn=augment_fn,
            n_iter=common.get("hpo_n_iter", 15))
        # 4) CV + esik
        pipe = build_pipeline(r["model"], r["scenario"], ablation,
                              sp.y_train, seed=seed, hp=best_hp)
        oof, _, cv_info = cv_evaluate(
            pipe, sp.X_train, sp.y_train,
            n_splits=common["cv"]["n_splits"], n_repeats=common["cv"]["n_repeats"],
            seed=seed, augment_fn=augment_fn)
        if oof is None:
            rec.update(status="failed", reason=cv_info.get("error", "cv yok"))
            return rec
        mask = ~np.isnan(oof)
        yv, oofm = np.asarray(sp.y_train)[mask], oof[mask]
        # Esik, test setinin GERCEK benign oranina kalibre edilir (dengeli
        # panellerde ~0.5, klinik panellerde ~0.8) -> esik dogru operasyon
        # noktasinda secilir. test_benign_frac bilinen tasarim orani, sizinti yok.
        bf = sp.info["test_benign_frac"]
        metric = common["threshold_metric"]
        # iki esik: prior KAPALI (train dagilimi) ve prior ACIK (test prior'i)
        thr_default = optimize_threshold(yv, oofm, metric=metric, target_benign_frac=None)
        thr_prior = optimize_threshold(yv, oofm, metric=metric, target_benign_frac=bf)
        thr = thr_prior if common.get("prior_aware_threshold") else thr_default
        # Sizintisiz secim metrigi: OOF tahminlerinin secilen esikteki macro-F1'i.
        # Test'e dokunmaz -> model secimi bunun uzerinden yapilinca selection-bias
        # olmaz; ayrica karar metrigi (macro_f1) ile tutarli.
        from sklearn.metrics import f1_score as _f1m
        cv_macro_f1 = float(_f1m(yv, (oofm >= thr).astype(int),
                                 average="macro", zero_division=0))

        # 5) refit (augmentasyonlu) + test (orijinal frozen)
        if augment_fn is not None:
            Xtr_a, ytr_a = augment_fn(sp.X_train, sp.y_train)
            pipe.fit(Xtr_a, ytr_a)
        else:
            pipe.fit(sp.X_train, sp.y_train)
        test_proba = pipe.predict_proba(sp.X_test)[:, 1]
        m = compute_metrics(sp.y_test, test_proba, threshold=thr)

        # Rapor icin: prior ONCESI (default) vs SONRASI (prior) karsilastirmasi
        m_def = compute_metrics(sp.y_test, test_proba, threshold=thr_default)
        m_pri = compute_metrics(sp.y_test, test_proba, threshold=thr_prior)
        rec.update(thr_default=m_def["threshold"], mcc_thr_default=m_def["mcc"],
                   macro_f1_thr_default=m_def["macro_f1"],
                   thr_prior=m_pri["threshold"], mcc_thr_prior=m_pri["mcc"],
                   macro_f1_thr_prior=m_pri["macro_f1"])

        # 6) grafikler
        gdir = ROOT / "SRC" / "Result" / "Graphics" / r["model"].upper()
        y_pred = (test_proba >= thr).astype(int)
        plot_confusion(sp.y_test, y_pred, gdir / f"{rid}__cm.png", rid)
        plot_roc(sp.y_test, test_proba, gdir / f"{rid}__roc.png", rid)
        plot_pr(sp.y_test, test_proba, gdir / f"{rid}__pr.png", rid)

        # 7) aciklanabilirlik (feature importance + SHAP)
        if common.get("explain"):
            try:
                names = final_feature_names(pipe)
                imp = native_importance(pipe, names)
                if imp is not None:
                    idir = ROOT / "SRC" / "Result" / "runs" / "importance"
                    idir.mkdir(parents=True, exist_ok=True)
                    imp.to_excel(idir / f"{rid}.xlsx", index=False)
                    rec["top_importance"] = ";".join(imp["feature"].head(5))
                fam = CAPABILITIES[r["model"]].family
                top = shap_summary(pipe, sp.X_test, names, fam,
                                   gdir / f"{rid}__shap.png")
                if top:
                    rec["top_shap"] = ";".join(top)
            except Exception as e:
                rec["explain_warn"] = f"{type(e).__name__}: {e}"

        rec.update(status="ok", best_hp=json.dumps(best_hp), hpo_cv_mcc=round(hpo_score, 4),
                   cv_mcc_mean=round(cv_info["cv_mcc_mean"], 4),
                   cv_mcc_std=round(cv_info["cv_mcc_std"], 4),
                   cv_macro_f1=round(cv_macro_f1, 4), **m)
    except Exception as e:
        import traceback
        rec.update(status="failed", reason=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc().splitlines()[-1])
    return rec


_WORKER_CACHE: dict = {}   # her paralel iscide kalici (split/aday/transfer cache)


def _run_one(r, common):
    """Paralel isci girisi: isci-yerel cache ile tek kosu calistirir."""
    return execute_run(r, common, _WORKER_CACHE)


def _load_checkpoint(path: Path) -> dict:
    """Checkpoint JSONL'den tamamlanmis kosulari yukler {run_id: rec}."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["run_id"]] = rec
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", help="subset | first_pass | full (vars: config)")
    ap.add_argument("--limit", type=int, help="ilk N kosu (debug)")
    ap.add_argument("--fresh", action="store_true",
                    help="checkpoint'i yok say, bastan basla")
    ap.add_argument("--results-dir",
                    help="cikti klasoru override (vars: config results_dir). "
                         "Ayri profiller (catboost/svm) icin ayri klasor ver.")
    args = ap.parse_args()

    common, prof, prof_name = load_config(args.profile)
    runs = list(enumerate_runs(prof, common.get("ablation_mode", "ofat")))
    if args.limit:
        runs = runs[:args.limit]

    out_dir = ROOT / Path(args.results_dir or common["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "_checkpoint.jsonl"
    if args.fresh and ckpt.exists():
        ckpt.unlink()

    # Resume: tamamlanmis kosulari atla, kalanini calistir
    done = _load_checkpoint(ckpt)
    results = list(done.values())
    todo = [r for r in runs if run_id(r) not in done]
    print(f"Profil: {prof_name}  |  toplam: {len(runs)}  |  "
          f"tamamlanmis: {len(done)}  |  kalan: {len(todo)}")

    workers = int(common.get("parallel_runs", 1) or 1)
    t0 = time.time()

    def _log(i, rec):
        tag = rec["status"].upper()
        extra = (f"MCC={rec.get('mcc')} AUC={rec.get('auc')}"
                 if rec["status"] == "ok" else rec.get("reason", ""))
        print(f"[{i}/{len(todo)}] {tag:8} {rec['run_id']}  {extra}")

    with ckpt.open("a", encoding="utf-8") as cf:   # her kosuda incremental yaz
        if workers > 1 and len(todo) > 1:
            # Paralel: model_n_jobs=1 (asiri abonelik onlemi) -> isciler env'den okur
            os.environ["MODELING_N_JOBS"] = str(common.get("model_n_jobs", 1))
            print(f"Paralel yurutme: {workers} isci")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one, r, common) for r in todo]
                for i, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    results.append(rec)
                    cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cf.flush()
                    _log(i, rec)
        else:
            split_cache = {}
            for i, r in enumerate(todo, 1):
                rec = execute_run(r, common, split_cache)
                results.append(rec)
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cf.flush()
                _log(i, rec)

    df = pd.DataFrame(results)
    df.to_excel(out_dir / "all_runs.xlsx", index=False)
    (out_dir / "all_runs.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTamamlandi ({time.time()-t0:.1f}s). Sonuc: {out_dir/'all_runs.xlsx'}")

    # En iyi-per-panel Excel
    try:
        from report import build_best_per_panel
        build_best_per_panel(df, out_dir.parent / "best_per_panel.xlsx")
    except Exception as e:
        print(f"[rapor uyarisi] {e}")


if __name__ == "__main__":
    main()

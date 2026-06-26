"""
===========================================================================
ACIKLANABILIRLIK - sadece SECILEN 4 PANEL SAMPIYONU icin (hedefli)
===========================================================================
SYZ2026 / MedicalVision

best_per_panel.xlsx'teki 4 sampiyonu kayitli best_hp ile yeniden kurar,
augment'li train'e fit eder, sonra:
  * native_importance -> SRC/Result/explain/<rid>__importance.xlsx
  * SHAP ozet grafigi  -> SRC/Result/explain/<rid>__shap.png
execute_run'un AYNI augmentation/transfer kurulumunu kullanir.

Kullanim:
  python SRC/modeling/explain_champions.py
===========================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "SRC" / "data_preprocessing"))

import runner as R                                         # noqa: E402
from models.registry import build_pipeline                # noqa: E402
from capabilities import CAPABILITIES                     # noqa: E402
from data.augment import make_augment_fn                  # noqa: E402
from eval.explain import (final_feature_names,            # noqa: E402
                          native_importance, shap_summary)

ABL_FLAGS = ["focal_loss", "feature_selection", "class_weight", "l2", "smote",
             "feature_subsample", "row_subsample", "early_stopping"]
OUT = ROOT / "SRC" / "Result" / "explain"


def main():
    common, _, _ = R.load_config("first_pass")
    OUT.mkdir(parents=True, exist_ok=True)
    bpp = pd.read_excel(ROOT / "SRC" / "Result" / "best_per_panel.xlsx",
                        sheet_name="best_per_panel")
    cache: dict = {}
    summary = []

    for _, row in bpp.iterrows():
        panel, model, scenario = row["panel"], row["model"], row["scenario"]
        data_aug, rid = row["data_aug"], row["run_id"]
        ablation = {f: row.get(f"abl_{f}", "off") for f in ABL_FLAGS}
        bh = row.get("best_hp")
        best_hp = json.loads(bh) if isinstance(bh, str) and bh.strip() else {}
        seed = common["seed"]
        print(f"\n=== {panel}: {model} / {data_aug} ===")

        sp = R.get_split(panel, common, cache)
        if data_aug == "transfer_learning":
            sp = R.get_transfer_split(panel, sp, common, cache)
        abl = dict(ablation)
        if "synthetic" in data_aug:
            abl["smote"] = "on"
        augment_fn = None
        if "clustering" in data_aug:
            X_extra, y_extra = R.get_candidates(panel, sp, common, cache)
            augment_fn = make_augment_fn(X_extra, y_extra)

        pipe = build_pipeline(model, scenario, abl, sp.y_train, seed=seed, hp=best_hp)
        if augment_fn is not None:
            Xtr_a, ytr_a = augment_fn(sp.X_train, sp.y_train)
            pipe.fit(Xtr_a, ytr_a)
        else:
            pipe.fit(sp.X_train, sp.y_train)

        names = final_feature_names(pipe)
        imp = native_importance(pipe, names)
        top_imp = []
        if imp is not None:
            imp.to_excel(OUT / f"{rid}__importance.xlsx", index=False)
            top_imp = imp["feature"].head(8).tolist()
            print("  top onem:", ";".join(imp["feature"].head(5)))

        fam = CAPABILITIES[model].family
        top_shap = shap_summary(pipe, sp.X_test, names, fam,
                                OUT / f"{rid}__shap.png") or []
        if top_shap:
            print("  top SHAP:", ";".join(top_shap[:5]))
        else:
            print("  SHAP: uretilemedi (model/sebep)")

        summary.append({"panel": panel, "model": model, "data_aug": data_aug,
                        "top_importance": ";".join(top_imp[:5]),
                        "top_shap": ";".join(top_shap[:5])})

    pd.DataFrame(summary).to_excel(OUT / "_ozet.xlsx", index=False)
    print(f"\nTamamlandi. Cikti: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

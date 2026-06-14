"""
TEMİZLENMİŞ veri için ablasyon CSV üretimi
Giriş : VERİLER/TEMİZLENMİŞ/YARISMA_TRAIN_{PANEL}_temiz.csv
Çıkış : VERİLER/ABLASYON_TEMİZLENMİŞ/{SENARYO}/{PANEL}_*.csv

Orijinal ablation_csv_factory.py'nin aynı mantığını kullanır.
Fark: Panel dosyaları temizlenmiş (194 sütun), çıktı dizini farklı.
Ek: domain-spesifik 'al_zero' imputation (AL_→0, EK_→medyan).

Çalıştırma:
  python build_ablation_temiz.py
  python build_ablation_temiz.py --models catboost tabpfn
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Factory'den encoding fonksiyonlarını içe al
from ablation_csv_factory import (
    apply_one_hot_encoding,
    apply_ordinal_encoding,
    build_category_levels,
    cast_string_features,
    drop_high_missing_columns,
    scale_numeric,
)

ROOT      = Path("/Users/mahmudselmansahin/Teknofest")
DATA_IN   = ROOT / "VERİLER" / "TEMİZLENMİŞ"
DATA_OUT  = ROOT / "VERİLER" / "ABLASYON_TEMİZLENMİŞ"
ID_COL    = "Variant_ID"
LABEL_COL = "Label"

PANEL_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER_temiz.csv",
    "KANSER": "YARISMA_TRAIN_KANSER_temiz.csv",
    "PAH":    "YARISMA_TRAIN_PAH_temiz.csv",
    "CFTR":   "YARISMA_TRAIN_CFTR_temiz.csv",
}

# AL_ ve EK_ sütun grupları (domain-spesifik imputation için)
AL_PREFIX = "AL_"
EK_PREFIX = "EK_"
EK_EXTRAS = {"BLOSUM62", "DELTA_HIDRO", "SNV_PATHWAY_COUNT"}


@dataclass(frozen=True)
class Scenario:
    model: str
    name: str
    encoding: str = "none"        # none | ordinal | onehot
    imputer: str = "none"         # none | median | mean | knn | al_zero
    scaler: str = "none"          # none | standard | minmax | robust


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_panels() -> dict[str, pd.DataFrame]:
    panels = {}
    for panel, filename in PANEL_FILES.items():
        p = DATA_IN / filename
        if not p.exists():
            raise FileNotFoundError(f"Bulunamadı: {p}")
        panels[panel] = pd.read_csv(p)
    return panels


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df)
            if pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Domain-spesifik imputation
# ---------------------------------------------------------------------------

def impute_al_zero(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """AL_→0 (nadirlik sinyali), EK_+extras→medyan."""
    out = df.copy()
    num_cols = numeric_feature_columns(out)
    al_cols  = [c for c in num_cols if c.startswith(AL_PREFIX)]
    ek_cols  = [c for c in num_cols if c.startswith(EK_PREFIX) or c in EK_EXTRAS]

    out[al_cols] = out[al_cols].fillna(0.0)

    medians = {}
    for col in ek_cols:
        med = out[col].median()
        medians[col] = float(med) if not pd.isna(med) else 0.0
    out[ek_cols] = out[ek_cols].fillna(medians)

    return out, {"method": "al_zero", "al_cols": len(al_cols), "ek_cols": len(ek_cols)}


def impute_numeric(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict]:
    if method == "none":
        return df, {"method": "none"}
    if method == "al_zero":
        return impute_al_zero(df)

    out = df.copy()
    num_cols = numeric_feature_columns(out)

    if method in {"median", "mean"}:
        fill = {}
        for col in num_cols:
            v = out[col].median() if method == "median" else out[col].mean()
            fill[col] = float(v) if not pd.isna(v) else 0.0
        out[num_cols] = out[num_cols].fillna(fill)
        return out, {"method": method}

    if method == "knn":
        n = min(5, max(1, len(out) - 1))
        imputer = KNNImputer(n_neighbors=n, keep_empty_features=True)
        out[num_cols] = pd.DataFrame(
            imputer.fit_transform(out[num_cols]),
            columns=num_cols, index=out.index
        )
        return out, {"method": "knn", "n_neighbors": n}

    raise ValueError(f"Bilinmeyen imputer: {method}")


# ---------------------------------------------------------------------------
# Senaryo uygulama
# ---------------------------------------------------------------------------

ENCODING_LABELS = {
    "none":    "Kategorik_String",
    "ordinal": "Ordinal_Encoding",
    "onehot":  "OneHot_Encoding",
}
IMPUTER_LABELS = {
    "none":    "Eksik_Veri_Korundu",
    "median":  "Eksik_Veri_Medyan",
    "mean":    "Eksik_Veri_Ortalama",
    "knn":     "Eksik_Veri_KNNImputer",
    "al_zero": "AL_Sifir_EK_Medyan",
}
SCALER_LABELS = {
    "none":     None,
    "robust":   "RobustScaler",
    "minmax":   "MinMaxScaler",
    "standard": "StandardScaler",
}


def scenario_folder(scenario: Scenario) -> Path:
    parts = [scenario.model.upper(), ENCODING_LABELS[scenario.encoding]]
    sl = SCALER_LABELS[scenario.scaler]
    if sl:
        parts.append(sl)
    parts.append(IMPUTER_LABELS[scenario.imputer])
    return DATA_OUT / "_".join(parts)


def apply_scenario(
    df: pd.DataFrame,
    scenario: Scenario,
    category_levels: dict,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    manifest: dict = {
        "scenario": asdict(scenario),
        "input_shape": list(df.shape),
        "input_missing": int(df.isna().sum().sum()),
    }

    # Encoding
    if scenario.encoding == "none":
        out = cast_string_features(out)
    elif scenario.encoding == "ordinal":
        out, _ = apply_ordinal_encoding(out, category_levels)
    elif scenario.encoding == "onehot":
        out, _ = apply_one_hot_encoding(out, category_levels)

    # Imputation
    out, imp_details = impute_numeric(out, scenario.imputer)
    manifest["imputer"] = imp_details

    # Scaling
    out, scl_details = scale_numeric(out, scenario.scaler)
    manifest["scaler"] = scl_details

    # Sütun sırası
    ordered = [c for c in [ID_COL] + [c for c in out.columns if c not in {ID_COL, LABEL_COL}] + [LABEL_COL] if c in out.columns]
    out = out[ordered]

    manifest.update({
        "output_shape": list(out.shape),
        "output_missing": int(out.isna().sum().sum()),
    })
    return out, manifest


# ---------------------------------------------------------------------------
# Senaryo tanımları
# ---------------------------------------------------------------------------

def all_scenarios() -> list[Scenario]:
    s: list[Scenario] = []

    # CatBoost — kategorik string, normalizasyon yok
    for imputer in ["none", "median", "mean", "knn", "al_zero"]:
        s.append(Scenario("catboost", f"catboost_{imputer}", "none", imputer, "none"))

    # XGBoost — ordinal encoding, NaN native veya doldurma
    for imputer in ["none", "median", "knn", "al_zero"]:
        s.append(Scenario("xgboost", f"xgboost_{imputer}", "ordinal", imputer, "none"))

    # LightGBM — kategorik string, NaN native veya doldurma
    for imputer in ["none", "median", "al_zero"]:
        s.append(Scenario("lightgbm", f"lightgbm_{imputer}", "none", imputer, "none"))

    # TabPFN — ordinal encoding, normalizasyon yok / var
    for imputer in ["median", "knn", "al_zero"]:
        s.append(Scenario("tabpfn", f"tabpfn_{imputer}", "ordinal", imputer, "none"))
        s.append(Scenario("tabpfn", f"tabpfn_minmax_{imputer}", "ordinal", imputer, "minmax"))

    # TabNet — ordinal + scaling
    for scaler in ["robust", "minmax", "standard"]:
        for imputer in ["median", "knn", "al_zero"]:
            s.append(Scenario("tabnet", f"tabnet_{scaler}_{imputer}", "ordinal", imputer, scaler))

    # KNN — onehot + scaling
    for scaler in ["robust", "minmax", "standard"]:
        for imputer in ["median", "knn", "al_zero"]:
            s.append(Scenario("knn", f"knn_{scaler}_{imputer}", "onehot", imputer, scaler))

    # SVM — onehot + scaling
    for scaler in ["robust", "minmax", "standard"]:
        for imputer in ["median", "knn", "al_zero"]:
            s.append(Scenario("svm", f"svm_{scaler}_{imputer}", "onehot", imputer, scaler))

    # AdaBoost — onehot veya ordinal
    for encoding in ["onehot", "ordinal"]:
        for imputer in ["median", "knn", "al_zero"]:
            s.append(Scenario("adaboost", f"adaboost_{encoding}_{imputer}", encoding, imputer, "none"))

    return s


def scenarios_for_models(models: list[str]) -> list[Scenario]:
    wanted = {m.lower() for m in models}
    return [s for s in all_scenarios() if s.model.lower() in wanted]


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"])
    args = parser.parse_args()

    if len(args.models) == 1 and args.models[0].lower() == "all":
        scenarios = all_scenarios()
    else:
        scenarios = scenarios_for_models(args.models)

    print(f"TEMİZLENMİŞ veri yükleniyor ({DATA_IN})...")
    panels = load_panels()
    category_levels = build_category_levels(panels)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    total = 0

    for scenario in scenarios:
        folder = scenario_folder(scenario)
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n  {folder.name}")

        for panel, df in panels.items():
            out_df, manifest = apply_scenario(df, scenario, category_levels)
            csv_path = folder / f"{panel}_{scenario.name}.csv"
            out_df.to_csv(csv_path, index=False)
            total += 1
            missing_left = manifest["output_missing"]
            print(f"    {panel}: {out_df.shape} | eksik={missing_left}")

        (folder / "manifest.json").write_text(
            json.dumps({"scenario": asdict(scenario)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\n✓ Tamamlandı. Toplam {total} CSV → {DATA_OUT}")


if __name__ == "__main__":
    main()

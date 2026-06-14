from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


PANEL_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}

ID_COL = "Variant_ID"
LABEL_COL = "Label"
MISSING_TOKEN = "__MISSING__"
HIGH_MISSING_THRESHOLD = 0.90


@dataclass(frozen=True)
class Scenario:
    model: str
    name: str
    encoding: str = "none"
    imputer: str = "none"
    scaler: str = "none"
    drop_high_missing: bool = False
    preserve_native_missing: bool = False
    category_columns_for_manifest: bool = False


def resolve_input_dir(input_dir: str | None) -> Path:
    if input_dir:
        path = Path(input_dir).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Input directory not found: {path}")

    candidates = [
        Path("/Users/mahmudselmansahin/Teknofest/VERİLER/ORİJİNAL"),
        Path("/Users/mahmudselmansahin/Teknofest/VERİLER/ORİJİNAL"),
        Path("/Users/mahmudselmansahin/Teknofest/KODLAR"),
        Path.cwd(),
    ]
    for path in candidates:
        if (path / PANEL_FILES["MASTER"]).exists():
            return path

    root = Path("/Users/mahmudselmansahin/Teknofest")
    if root.exists():
        matches = list(root.rglob(PANEL_FILES["MASTER"]))
        if matches:
            return matches[0].parent

    raise FileNotFoundError("Could not locate original TEKNOFEST CSV files.")


def load_panels(input_dir: Path) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for panel, filename in PANEL_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing panel CSV: {path}")
        panels[panel] = pd.read_csv(path)
    return panels


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


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cats = set(categorical_columns(df))
    return [
        col
        for col in feature_columns(df)
        if col not in cats and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_category_levels(panels: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    all_cat_cols = sorted(
        {col for df in panels.values() for col in categorical_columns(df)},
        key=lambda col: list(next(iter(panels.values())).columns).index(col)
        if col in next(iter(panels.values())).columns
        else col,
    )
    levels: dict[str, list[str]] = {}
    for col in all_cat_cols:
        values: list[str] = []
        for df in panels.values():
            if col in df.columns:
                values.extend(df[col].fillna(MISSING_TOKEN).astype(str).tolist())
        levels[col] = sorted(set(values))
    return levels


def safe_token(value: object) -> str:
    token = str(value)
    token = re.sub(r"[^0-9A-Za-z_]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "blank"


def apply_ordinal_encoding(
    df: pd.DataFrame,
    category_levels: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    out = df.copy()
    mappings: dict[str, dict[str, int]] = {}
    for col in categorical_columns(out):
        levels = category_levels.get(col, [MISSING_TOKEN])
        mapping = {value: idx for idx, value in enumerate(levels)}
        out[col] = out[col].fillna(MISSING_TOKEN).astype(str).map(mapping).fillna(-1).astype(int)
        mappings[col] = mapping
    return out, mappings


def apply_one_hot_encoding(
    df: pd.DataFrame,
    category_levels: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = df.copy()
    cat_cols = categorical_columns(out)
    generated_columns: dict[str, list[str]] = {}
    encoded_parts: list[pd.DataFrame] = []

    for col in cat_cols:
        levels = category_levels.get(col, [MISSING_TOKEN])
        series = out[col].fillna(MISSING_TOKEN).astype(str)
        columns_for_col: list[str] = []
        encoded_data: dict[str, np.ndarray] = {}
        for level in levels:
            encoded_col = f"{col}__{safe_token(level)}"
            encoded_data[encoded_col] = (series == level).astype(np.int8).to_numpy()
            columns_for_col.append(encoded_col)
        encoded = pd.DataFrame(encoded_data, index=out.index)
        encoded_parts.append(encoded)
        generated_columns[col] = columns_for_col

    out = out.drop(columns=cat_cols)
    if encoded_parts:
        out = pd.concat([out] + encoded_parts, axis=1)
    return out, generated_columns


def impute_numeric(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if method == "none":
        return df, {"method": "none"}

    out = df.copy()
    num_cols = numeric_feature_columns(out)
    details: dict[str, object] = {"method": method, "columns": num_cols}
    if not num_cols:
        return out, details

    if method in {"median", "mean"}:
        fill_values: dict[str, float] = {}
        for col in num_cols:
            value = out[col].median() if method == "median" else out[col].mean()
            if pd.isna(value):
                value = 0.0
            fill_values[col] = float(value)
        out[num_cols] = out[num_cols].fillna(fill_values)
        details["all_missing_columns"] = [
            col for col in num_cols if df[col].isna().all()
        ]
        return out, details

    if method == "knn":
        n_neighbors = min(5, max(1, len(out) - 1))
        imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        imputed = imputer.fit_transform(out[num_cols])
        out[num_cols] = pd.DataFrame(imputed, columns=num_cols, index=out.index)
        details["n_neighbors"] = n_neighbors
        details["all_missing_columns"] = [
            col for col in num_cols if df[col].isna().all()
        ]
        return out, details

    raise ValueError(f"Unknown imputer: {method}")


def scale_numeric(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if method == "none":
        return df, {"method": "none"}

    scaler_by_method = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scaler_by_method:
        raise ValueError(f"Unknown scaler: {method}")

    out = df.copy()
    num_cols = numeric_feature_columns(out)
    if not num_cols:
        return out, {"method": method, "columns": []}

    scaler = scaler_by_method[method]()
    out[num_cols] = scaler.fit_transform(out[num_cols])
    return out, {"method": method, "columns": num_cols}


def cast_string_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_columns(out):
        out[col] = out[col].astype("string")
    return out


def drop_high_missing_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = feature_columns(df)
    dropped = [
        col
        for col in feature_cols
        if df[col].isna().mean() >= HIGH_MISSING_THRESHOLD
    ]
    return df.drop(columns=dropped), dropped


def apply_scenario(
    df: pd.DataFrame,
    scenario: Scenario,
    category_levels: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    working = df.copy()
    manifest: dict[str, object] = {
        "scenario": asdict(scenario),
        "input_shape": list(df.shape),
        "input_missing_values": int(df.isna().sum().sum()),
    }

    if scenario.drop_high_missing:
        working, dropped_columns = drop_high_missing_columns(working)
    else:
        dropped_columns = []
    manifest["dropped_columns"] = dropped_columns

    encoding_details: dict[str, object]
    if scenario.encoding == "none":
        working = cast_string_features(working)
        encoding_details = {
            "method": "none",
            "categorical_columns": categorical_columns(working),
        }
    elif scenario.encoding == "ordinal":
        working, mappings = apply_ordinal_encoding(working, category_levels)
        encoding_details = {
            "method": "ordinal",
            "categorical_columns": list(mappings.keys()),
            "mappings": mappings,
        }
    elif scenario.encoding == "onehot":
        working, generated = apply_one_hot_encoding(working, category_levels)
        encoding_details = {
            "method": "onehot",
            "source_columns": list(generated.keys()),
            "generated_columns": generated,
        }
    else:
        raise ValueError(f"Unknown encoding: {scenario.encoding}")
    manifest["encoding"] = encoding_details

    working, imputer_details = impute_numeric(working, scenario.imputer)
    manifest["imputer"] = imputer_details

    working, scaler_details = scale_numeric(working, scenario.scaler)
    manifest["scaler"] = scaler_details

    ordered_columns = []
    if ID_COL in working.columns:
        ordered_columns.append(ID_COL)
    ordered_columns.extend(
        col for col in working.columns if col not in {ID_COL, LABEL_COL}
    )
    if LABEL_COL in working.columns:
        ordered_columns.append(LABEL_COL)
    working = working.loc[:, ordered_columns].copy()

    manifest.update(
        {
            "output_shape": list(working.shape),
            "output_missing_values": int(working.isna().sum().sum()),
            "label_counts": working[LABEL_COL].value_counts(dropna=False).to_dict()
            if LABEL_COL in working.columns
            else {},
            "variant_id_present": ID_COL in working.columns,
            "label_present": LABEL_COL in working.columns,
            "category_columns": categorical_columns(working)
            if scenario.category_columns_for_manifest
            else [],
        }
    )
    return working, manifest


def readable_folder_parts(scenario: Scenario) -> list[str]:
    encoding_labels = {
        "none": "Kategorik_String",
        "ordinal": "Ordinal_Encoding",
        "onehot": "OneHot_Encoding",
    }
    imputer_labels = {
        "none": "Eksik_Veri_Korundu",
        "median": "Eksik_Veri_Medyan",
        "mean": "Eksik_Veri_Ortalama",
        "knn": "Eksik_Veri_KNNImputer",
    }
    scaler_labels = {
        "none": None,
        "robust": "RobustScaler",
        "minmax": "MinMaxScaler",
        "standard": "StandardScaler",
    }

    parts = [scenario.model.upper()]
    parts.append(encoding_labels[scenario.encoding])
    scaler_label = scaler_labels[scenario.scaler]
    if scaler_label:
        parts.append(scaler_label)
    parts.append(imputer_labels[scenario.imputer])
    if scenario.drop_high_missing:
        parts.append("Yuksek_Eksik_Sutun_Silinmis")
    return parts


def scenario_folder(output_dir: Path, scenario: Scenario) -> Path:
    return output_dir / "MODELLER" / Path(*readable_folder_parts(scenario))


def copy_original_csvs(input_dir: Path, output_dir: Path) -> None:
    original_dir = output_dir / "VERILER" / "ORIGINAL"
    original_dir.mkdir(parents=True, exist_ok=True)
    for filename in PANEL_FILES.values():
        shutil.copy2(input_dir / filename, original_dir / filename)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run_scenarios(
    scenarios: Iterable[Scenario],
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> list[Path]:
    resolved_input = resolve_input_dir(input_dir)
    resolved_output = (
        Path(output_dir).expanduser()
        if output_dir
        else Path.cwd() / "ABLASYON_CALISMASI"
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    panels = load_panels(resolved_input)
    category_levels = build_category_levels(panels)
    copy_original_csvs(resolved_input, resolved_output)
    written_csvs: list[Path] = []

    for scenario in scenarios:
        folder = scenario_folder(resolved_output, scenario)
        folder.mkdir(parents=True, exist_ok=True)
        scenario_manifest: dict[str, object] = {
            "model": scenario.model.upper(),
            "scenario": scenario.name,
            "folder_structure": "ABLASYON_CALISMASI/MODELLER/{MODEL}/{ENCODING}/{SCALER}/{EKSIK_VERI}",
            "input_dir": str(resolved_input),
            "output_dir": str(folder),
            "panels": {},
        }

        for panel, df in panels.items():
            transformed, manifest = apply_scenario(df, scenario, category_levels)
            output_csv = folder / f"{panel}_{scenario.name}.csv"
            transformed.to_csv(output_csv, index=False)
            scenario_manifest["panels"][panel] = {
                **manifest,
                "csv": str(output_csv),
            }
            written_csvs.append(output_csv)

        write_json(folder / "manifest.json", scenario_manifest)

    write_json(
        resolved_output / "category_levels.json",
        {
            "input_dir": str(resolved_input),
            "category_levels": category_levels,
        },
    )
    return written_csvs


def all_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = [
        Scenario(
            model="catboost",
            name="catboost_raw",
            encoding="none",
            imputer="none",
            preserve_native_missing=True,
        ),
        Scenario(
            model="catboost",
            name="catboost_drop_high_missing",
            encoding="none",
            imputer="none",
            drop_high_missing=True,
            preserve_native_missing=True,
        ),
        Scenario(
            model="catboost",
            name="catboost_median",
            encoding="none",
            imputer="median",
        ),
        Scenario(
            model="catboost",
            name="catboost_mean",
            encoding="none",
            imputer="mean",
        ),
        Scenario(
            model="catboost",
            name="catboost_knn",
            encoding="none",
            imputer="knn",
        ),
        Scenario(
            model="catboost",
            name="catboost_drop_high_missing_median",
            encoding="none",
            imputer="median",
            drop_high_missing=True,
        ),
        Scenario(
            model="catboost",
            name="catboost_drop_high_missing_mean",
            encoding="none",
            imputer="mean",
            drop_high_missing=True,
        ),
        Scenario(
            model="catboost",
            name="catboost_drop_high_missing_knn",
            encoding="none",
            imputer="knn",
            drop_high_missing=True,
        ),
        Scenario(
            model="xgboost",
            name="xgboost_ordinal_missing_native",
            encoding="ordinal",
            imputer="none",
            preserve_native_missing=True,
        ),
        Scenario(
            model="xgboost",
            name="xgboost_ordinal_median",
            encoding="ordinal",
            imputer="median",
        ),
        Scenario(
            model="lightgbm",
            name="lightgbm_category_missing_native",
            encoding="none",
            imputer="none",
            preserve_native_missing=True,
            category_columns_for_manifest=True,
        ),
        Scenario(
            model="lightgbm",
            name="lightgbm_category_median",
            encoding="none",
            imputer="median",
            category_columns_for_manifest=True,
        ),
    ]

    for model in ["knn", "svm"]:
        for scaler in ["robust", "minmax", "standard"]:
            for imputer in ["median", "mean", "knn"]:
                scenarios.append(
                    Scenario(
                        model=model,
                        name=f"{model}_onehot_{scaler}_{imputer}",
                        encoding="onehot",
                        imputer=imputer,
                        scaler=scaler,
                    )
                )

    for scaler in ["robust", "minmax", "standard"]:
        for imputer in ["median", "mean", "knn"]:
            scenarios.append(
                Scenario(
                    model="tabnet",
                    name=f"tabnet_ordinal_{scaler}_{imputer}",
                    encoding="ordinal",
                    imputer=imputer,
                    scaler=scaler,
                )
            )

    for imputer in ["median", "mean", "knn"]:
        scenarios.append(
            Scenario(
                model="tabpfn",
                name=f"tabpfn_ordinal_{imputer}",
                encoding="ordinal",
                imputer=imputer,
            )
        )
        scenarios.append(
            Scenario(
                model="tabpfn",
                name=f"tabpfn_drop_high_missing_ordinal_{imputer}",
                encoding="ordinal",
                imputer=imputer,
                drop_high_missing=True,
            )
        )

    for encoding in ["onehot", "ordinal"]:
        for imputer in ["median", "mean", "knn"]:
            scenarios.append(
                Scenario(
                    model="adaboost",
                    name=f"adaboost_{encoding}_{imputer}",
                    encoding=encoding,
                    imputer=imputer,
                )
            )

    return scenarios


def scenarios_for_models(models: Iterable[str]) -> list[Scenario]:
    wanted = {model.lower() for model in models}
    scenarios = [scenario for scenario in all_scenarios() if scenario.model.lower() in wanted]
    missing = wanted - {scenario.model.lower() for scenario in scenarios}
    if missing:
        raise ValueError(f"Unknown model name(s): {', '.join(sorted(missing))}")
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model-specific CSV ablation datasets without training models."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing original YARISMA_TRAIN_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where ABLASYON_VERILERI will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model families to generate: all, catboost, xgboost, lightgbm, knn, svm, tabnet, tabpfn, adaboost.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.models) == 1 and args.models[0].lower() == "all":
        scenarios = all_scenarios()
    else:
        scenarios = scenarios_for_models(args.models)
    written = run_scenarios(
        scenarios=scenarios,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(f"Generated {len(written)} CSV files.")


if __name__ == "__main__":
    main()

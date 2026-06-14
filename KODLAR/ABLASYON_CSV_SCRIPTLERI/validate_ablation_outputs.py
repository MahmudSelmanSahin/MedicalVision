from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ablation_csv_factory import ID_COL, LABEL_COL, PANEL_FILES, resolve_input_dir


NAN_FORBIDDEN_MODELS = {"KNN", "SVM", "TABNET", "TABPFN", "ADABOOST"}


def load_original_shapes(input_dir: Path) -> dict[str, int]:
    shapes: dict[str, int] = {}
    for panel, filename in PANEL_FILES.items():
        shapes[panel] = len(pd.read_csv(input_dir / filename, usecols=[LABEL_COL]))
    return shapes


def find_panel_name(csv_path: Path) -> str | None:
    for panel in PANEL_FILES:
        if csv_path.name.startswith(f"{panel}_"):
            return panel
    return None


def validate_outputs(input_dir: Path, output_dir: Path) -> dict[str, object]:
    original_rows = load_original_shapes(input_dir)
    errors: list[str] = []
    warnings: list[str] = []
    csv_count = 0
    onehot_reference: dict[str, list[str]] = {}

    model_root = output_dir / "MODELLER"
    search_root = model_root if model_root.exists() else output_dir

    for csv_path in sorted(search_root.rglob("*.csv")):
        csv_count += 1
        rel_parts = csv_path.relative_to(search_root).parts
        model = rel_parts[0].upper() if rel_parts else "UNKNOWN"
        scenario = "/".join(rel_parts[1:-1]) or csv_path.parent.name
        panel = find_panel_name(csv_path)
        if panel is None:
            errors.append(f"Cannot infer panel name from {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if ID_COL not in df.columns:
            errors.append(f"{csv_path}: missing {ID_COL}")
        if LABEL_COL not in df.columns:
            errors.append(f"{csv_path}: missing {LABEL_COL}")
        if len(df) != original_rows[panel]:
            errors.append(
                f"{csv_path}: row count {len(df)} != original {original_rows[panel]}"
            )
        if model in NAN_FORBIDDEN_MODELS and df.isna().sum().sum() > 0:
            errors.append(f"{csv_path}: NaN values remain for {model}")

        onehot_cols = sorted([col for col in df.columns if "__" in col])
        if onehot_cols:
            key = f"{model}/{scenario}"
            if key not in onehot_reference:
                onehot_reference[key] = onehot_cols
            elif onehot_reference[key] != onehot_cols:
                errors.append(f"{csv_path}: one-hot columns differ inside {key}")

    original_dir = output_dir / "VERILER" / "ORIGINAL"
    if original_dir.exists():
        original_count = len(list(original_dir.glob("*.csv")))
        if original_count != len(PANEL_FILES):
            warnings.append(
                f"Expected {len(PANEL_FILES)} original CSV copies, found {original_count}"
            )

    if csv_count == 0:
        errors.append(f"No model CSV files found under {search_root}")

    return {
        "csv_count": csv_count,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated ablation CSV files.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="ABLASYON_CALISMASI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_dir = Path(args.output_dir).expanduser()
    result = validate_outputs(input_dir, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

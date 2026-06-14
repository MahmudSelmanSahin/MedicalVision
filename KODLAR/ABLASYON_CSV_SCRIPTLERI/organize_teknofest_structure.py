from __future__ import annotations

import json
import shutil
from pathlib import Path


TEKNOFEST_ROOT = Path("/Users/mahmudselmansahin/Teknofest")
WORKSPACE_ROOT = Path("/Users/mahmudselmansahin/Documents/Codex/2026-05-21/files-mentioned-by-the-user-2026")

SCRIPT_FILES = [
    "ablation_csv_factory.py",
    "validate_ablation_outputs.py",
    "run_catboost_csv.py",
    "run_xgboost_csv.py",
    "run_lightgbm_csv.py",
    "run_knn_csv.py",
    "run_svm_csv.py",
    "run_tabnet_csv.py",
    "run_tabpfn_csv.py",
    "run_adaboost_csv.py",
    "run_all_csv_ablation.py",
]

MODEL_ALIASES = {
    "ADABOOST": "ADABOOST",
    "CATBOOST": "CATBOOST",
    "KNN": "KNN",
    "LIGHTGBM": "LIGHTGBM",
    "SVM": "SVM",
    "TABNET": "TABNET",
    "TABPFN": "TABPFN",
    "XGBOOST": "XGBOOST",
}

FEATURE_METHOD_DIRS = [
    "FILTER",
    "WRAPPER",
    "EMBEDDED",
    "FILTER + EMBEDDED",
    "FILTER + WRAPPER",
    "WRAPPER + EMBEDDED",
    "FILTER + WRAPPER + EMBEDDED",
]

DATA_VARIANT_SOURCES = {
    "ORİJİNAL": [
        TEKNOFEST_ROOT / "VERİLER" / "ORİJİNAL",
        TEKNOFEST_ROOT / "KODLAR",
    ],
    "EKSİK VERİ SİLİNMİŞ": [
        TEKNOFEST_ROOT / "KODLAR" / "Silinmiş Veri",
    ],
    "DOLDURULMUŞ": [
        TEKNOFEST_ROOT / "KODLAR" / "Doldurulmuş Veri",
    ],
    "NORMALİZE": [
        TEKNOFEST_ROOT / "KODLAR" / "Normalize Edilmiş Veri",
    ],
    "NORMALİZE + DOLDURULMUŞ": [
        TEKNOFEST_ROOT / "KODLAR" / "Nomalize_Doldur",
        TEKNOFEST_ROOT / "KODLAR" / "Doldur_Normalize",
    ],
    "NORMALİZE + EKSİK VERİ SİLİNMİŞ": [
        TEKNOFEST_ROOT / "KODLAR" / "Normalize_Sil",
    ],
    "DOLDURULMUŞ + EKSİK VERİ SİLİNMİŞ": [
        TEKNOFEST_ROOT / "KODLAR" / "Doldur_Sil",
    ],
    "NORMALİZE + EKSİK VERİ SİLİNMİŞ + DOLDURULMUŞ": [
        TEKNOFEST_ROOT / "KODLAR" / "Normalize_Doldur_Sil",
        TEKNOFEST_ROOT / "KODLAR" / "Doldur_Normalize_Sil",
    ],
    "SENTETİK VERİ ÜRETİLMİŞ": [],
    "SENTETİK VERİLERLE YUKARIDAKİ İŞLEMLER TEKRAR": [],
}


def copy_file(src: Path, dst: Path) -> None:
    if dst.exists() and src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_contents(src: Path, dst: Path, patterns: tuple[str, ...] = ("*.csv",)) -> int:
    if not src.exists():
        return 0
    count = 0
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for file_path in sorted(src.glob(pattern)):
            if file_path.is_file():
                copy_file(file_path, dst / file_path.name)
                count += 1
    return count


def ensure_model_tree() -> dict[str, object]:
    report: dict[str, object] = {}
    source_root = TEKNOFEST_ROOT / "KODLAR" / "ABLASYON_CALISMASI" / "MODELLER"
    model_root = TEKNOFEST_ROOT / "MODELLER"
    model_root.mkdir(parents=True, exist_ok=True)

    for source_model_dir in sorted(source_root.iterdir()):
        if not source_model_dir.is_dir():
            continue
        model_name = MODEL_ALIASES.get(source_model_dir.name.upper(), source_model_dir.name.upper())
        target_model_dir = model_root / model_name
        target_model_dir.mkdir(parents=True, exist_ok=True)

        for method in FEATURE_METHOD_DIRS:
            method_dir = target_model_dir / method
            (method_dir / "GRAFIKLER").mkdir(parents=True, exist_ok=True)
            (method_dir / "SONUCLAR").mkdir(parents=True, exist_ok=True)

        target_data_dir = target_model_dir / "VERILER"
        if target_data_dir.exists():
            shutil.rmtree(target_data_dir)
        shutil.copytree(source_model_dir, target_data_dir)

        csv_count = len(list(target_data_dir.rglob("*.csv")))
        report[model_name] = {
            "csv_count": csv_count,
            "data_dir": str(target_data_dir),
            "feature_method_dirs": FEATURE_METHOD_DIRS,
        }
    return report


def ensure_data_tree() -> dict[str, object]:
    report: dict[str, object] = {}
    data_root = TEKNOFEST_ROOT / "VERİLER"
    data_root.mkdir(parents=True, exist_ok=True)

    for folder_name, sources in DATA_VARIANT_SOURCES.items():
        target_dir = data_root / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for source in sources:
            copied += copy_tree_contents(source, target_dir)
        report[folder_name] = {
            "csv_count": len(list(target_dir.glob("*.csv"))),
            "copied_this_run": copied,
            "path": str(target_dir),
        }
    return report


def ensure_code_tree() -> dict[str, object]:
    scripts_dir = TEKNOFEST_ROOT / "KODLAR" / "ABLASYON_CSV_SCRIPTLERI"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for filename in SCRIPT_FILES:
        source_candidates = [
            TEKNOFEST_ROOT / "KODLAR" / filename,
            WORKSPACE_ROOT / filename,
        ]
        source = next((path for path in source_candidates if path.exists()), None)
        if source is None:
            continue
        copy_file(source, scripts_dir / filename)
        copied.append(filename)

    return {
        "scripts_dir": str(scripts_dir),
        "copied_scripts": copied,
    }


def main() -> None:
    report = {
        "root": str(TEKNOFEST_ROOT),
        "kodlar": ensure_code_tree(),
        "modeller": ensure_model_tree(),
        "veriler": ensure_data_tree(),
    }
    report_path = TEKNOFEST_ROOT / "KLASOR_YAPISI_RAPORU.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

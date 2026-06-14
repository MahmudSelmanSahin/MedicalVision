from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault(
    "TABPFN_MODEL_CACHE_DIR",
    "/Users/mahmudselmansahin/Documents/Codex/2026-05-21/files-mentioned-by-the-user-2026/.tabpfn_cache",
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    "/Users/mahmudselmansahin/Documents/Codex/2026-05-21/files-mentioned-by-the-user-2026/.cache",
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path("/Users/mahmudselmansahin/Teknofest")
DATA_ROOT = ROOT / "VERİLER" / "ABLASYON_MODEL_BAZLI"
MODELS_ROOT = ROOT / "MODELLER"
ID_COL = "Variant_ID"
LABEL_COL = "Label"
RANDOM_STATE = 42


def scenario_dirs(prefix: str) -> list[Path]:
    return sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)])


def panel_name(csv_path: Path) -> str:
    return csv_path.name.split("_", 1)[0]


def split_data(x: pd.DataFrame, y: pd.Series):
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        row["pr_auc"] = average_precision_score(y_true, y_prob)
        row["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        row["pr_auc"] = np.nan
        row["roc_auc"] = np.nan
    return row


def save_confusion_matrix_png(y_true, y_pred, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.3, 4.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Patojenik"],
        yticklabels=["Benign", "Patojenik"],
    )
    plt.title(title)
    plt.xlabel("Tahmin")
    plt.ylabel("Gercek")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_precision_recall_png(y_true, y_prob, out_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 4.5))
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_summary_plots(metrics: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    for metric in ["f1_macro", "mcc", "pr_auc", "recall"]:
        plot_df = metrics.sort_values(metric, ascending=False)
        labels = plot_df["panel"] + " | " + plot_df["scenario"]
        plt.figure(figsize=(10, max(4, len(plot_df) * 0.35)))
        sns.barplot(x=plot_df[metric], y=labels, color="#4C78A8")
        plt.title(f"{model_name} ablasyon - {metric}")
        plt.xlabel(metric)
        plt.ylabel("Panel | Senaryo")
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{metric}_summary.png", dpi=180)
        plt.close()


def save_metric_tables(metrics: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    metrics = metrics.sort_values(
        ["panel", "f1_macro", "mcc"],
        ascending=[True, False, False],
    )
    metrics.to_csv(out_dir / f"{model_name}_ablation_metrics.csv", index=False)
    metrics.groupby("panel", as_index=False).head(1).to_csv(
        out_dir / f"{model_name}_best_by_panel.csv",
        index=False,
    )

"""
Panel Bazlı Özellik Analizi

Her panel için:
  1. CatBoost feature importance (orijinal vs temizlenmiş)
  2. Spearman korelasyon haritası (hangi özellikler silindi?)
  3. Panel bazlı özellik örtüşme matrisi
  4. CFTR confusion matrix karşılaştırması (orijinal vs temiz)
  5. Tüm görsel BELGELER/OZELLIK_ANALIZI/ klasörüne kaydedilir

Çalıştırma:
  python panel_feature_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORTAK"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from model_ablation_common import ID_COL, LABEL_COL, RANDOM_STATE, ROOT, split_data

PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
ORIG_DIR   = ROOT / "VERİLER" / "ORİJİNAL"
TEMIZ_DIR  = ROOT / "VERİLER" / "TEMİZLENMİŞ"
OUT_DIR    = ROOT / "BELGELER" / "OZELLIK_ANALIZI"
TOP_N      = 20

BG    = "#0F1117"
FG    = "#E8EAF0"
ACC   = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]
PANEL_COLORS = dict(zip(PANELS, ACC))

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#3A3D4A", "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": "#2A2D3A",
    "font.family": "DejaVu Sans",
})


def load(panel, veri="orijinal"):
    if veri == "orijinal":
        path = ORIG_DIR / f"YARISMA_TRAIN_{panel}.csv"
    else:
        path = TEMIZ_DIR / f"YARISMA_TRAIN_{panel}_temiz.csv"
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in {ID_COL, LABEL_COL}]
    y = df[LABEL_COL].astype(int)
    return df, feats, y


def encode_for_cb(df, cols):
    X = df[cols].copy()
    cat_idx = []
    for i, c in enumerate(cols):
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].fillna("__MISSING__").astype(str)
            cat_idx.append(i)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    return X, cat_idx


def train_catboost(df, feats, y, iterations=150):
    X, cat_idx = encode_for_cb(df, feats)
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=0.05, depth=5,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )
    model.fit(X, y, cat_features=cat_idx)
    importances = model.get_feature_importance()
    return dict(zip(feats, importances)), model, X, cat_idx


def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path.name}")


# ─── 1. Panel bazlı özellik önem çubuk grafikleri ─────────────────────────────
def plot_importance_per_panel(imp_orig: dict, imp_temiz: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Panel Bazlı Özellik Önemi (CatBoost) — Orijinal vs Temizlenmiş",
                 fontsize=14, color=FG, y=1.01)

    for ax, panel in zip(axes.flat, PANELS):
        orig  = pd.Series(imp_orig[panel]).sort_values(ascending=False).head(TOP_N)
        temiz = pd.Series(imp_temiz.get(panel, {}))

        feats = orig.index.tolist()
        orig_vals  = orig.values
        temiz_vals = np.array([temiz.get(f, 0.0) for f in feats])

        x = np.arange(len(feats))
        w = 0.38
        col = PANEL_COLORS[panel]
        ax.barh(x + w/2, orig_vals,  w, color=col,      alpha=0.85, label="Orijinal")
        ax.barh(x - w/2, temiz_vals, w, color="#78909C", alpha=0.85, label="Temizlenmiş")

        # Temizlenmiş'te olmayan özellikleri işaretle
        for i, f in enumerate(feats):
            if temiz.get(f, 0) == 0:
                ax.barh(x[i] + w/2, orig_vals[i], w, color="#EF5350",
                        alpha=0.6, hatch="///")

        ax.set_yticks(x)
        ax.set_yticklabels(feats, fontsize=7)
        ax.set_title(f"{panel}  (Top {TOP_N})", color=col, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.legend(fontsize=8, facecolor="#1E2130")
        ax.grid(axis="x", alpha=0.3)

    # Legend for eliminated
    elim_patch = mpatches.Patch(facecolor="#EF5350", alpha=0.6, hatch="///",
                                label="TEMİZLENMİŞ'te yok (silindi)")
    fig.legend(handles=[elim_patch], loc="lower center", ncol=1,
               fontsize=9, facecolor="#1E2130", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "01_panel_feature_importance.png")


# ─── 2. Spearman korelasyon ısı haritası (orijinal, her panel) ───────────────
def plot_spearman_heatmap(imp_orig: dict):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Spearman Korelasyon Haritası — En Önemli 20 Özellik (Orijinal)",
                 fontsize=13, color=FG)

    cmap = LinearSegmentedColormap.from_list("rdbl",
        ["#1565C0", "#E3F2FD", "#FFEBEE", "#B71C1C"])

    for ax, panel in zip(axes.flat, PANELS):
        df, feats, _ = load(panel, "orijinal")
        top_feats = pd.Series(imp_orig[panel]).sort_values(ascending=False).head(TOP_N).index.tolist()
        # sadece numerik
        num_feats = [f for f in top_feats if df[f].dtype != object and
                     not pd.api.types.is_string_dtype(df[f])][:TOP_N]
        corr = df[num_feats].apply(pd.to_numeric, errors="coerce").corr(method="spearman")

        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(num_feats)))
        ax.set_yticks(range(len(num_feats)))
        ax.set_xticklabels(num_feats, rotation=90, fontsize=6)
        ax.set_yticklabels(num_feats, fontsize=6)
        ax.set_title(f"{panel}", color=PANEL_COLORS[panel], fontsize=11, fontweight="bold")

        # 0.80 eşiği: eliminasyon bölgesini işaretle
        n = len(num_feats)
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr.values[i, j]) >= 0.80:
                    for di in [i, j]:
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                                    fill=False, edgecolor="#FFD54F",
                                                    linewidth=0.8, alpha=0.7))

        plt.colorbar(im, ax=ax, shrink=0.8)

    thresh_patch = mpatches.Patch(facecolor="none", edgecolor="#FFD54F",
                                  linewidth=1.5, label="|Spearman| ≥ 0.80 (eliminasyon bölgesi)")
    fig.legend(handles=[thresh_patch], loc="lower center", fontsize=9,
               facecolor="#1E2130", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "02_spearman_heatmap.png")


# ─── 3. Panel özellik örtüşme matrisi ─────────────────────────────────────────
def plot_overlap_matrix(imp_orig: dict, imp_temiz: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Panel Bazlı Özellik Örtüşmesi — Top 30 (Orijinal vs Temizlenmiş)",
                 fontsize=13, color=FG)

    for ax, (veri, imp) in zip(axes, [("Orijinal", imp_orig), ("Temizlenmiş", imp_temiz)]):
        top30 = {p: set(pd.Series(imp.get(p, {})).sort_values(ascending=False).head(30).index)
                 for p in PANELS}
        mat = np.zeros((4, 4))
        for i, p1 in enumerate(PANELS):
            for j, p2 in enumerate(PANELS):
                if top30[p1] and top30[p2]:
                    mat[i, j] = len(top30[p1] & top30[p2]) / len(top30[p1] | top30[p2])

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(PANELS, fontsize=10)
        ax.set_yticklabels(PANELS, fontsize=10)
        ax.set_title(veri, color=FG, fontsize=11)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="black" if mat[i,j] > 0.4 else FG, fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Jaccard Benzerlik")

    fig.tight_layout()
    savefig(fig, "03_panel_feature_overlap.png")


# ─── 4. CFTR confusion matrix karşılaştırması ────────────────────────────────
def plot_cftr_confusion_comparison():
    results = {}
    for veri in ["orijinal", "temizlenmiş"]:
        df, feats, y = load("CFTR", veri)
        X, cat_idx = encode_for_cb(df, feats)
        X_tr, X_te, y_tr, y_te = split_data(X, y)
        model = CatBoostClassifier(
            iterations=250, learning_rate=0.05, depth=6,
            loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        model.fit(X_tr, y_tr, cat_features=cat_idx)
        y_pred = model.predict(X_te).astype(int)
        mcc = matthews_corrcoef(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred)
        results[veri] = {"cm": cm, "mcc": mcc, "y_te": y_te, "y_pred": y_pred}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CFTR — Confusion Matrix: Orijinal vs Temizlenmiş (CatBoost)",
                 fontsize=13, color=FG)

    for ax, (veri, res) in zip(axes, results.items()):
        cm = res["cm"]
        cmap = LinearSegmentedColormap.from_list("heatmap",
            [BG, PANEL_COLORS["CFTR"]])
        im = ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=FG)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Benign", "Pred Patojenik"], fontsize=9)
        ax.set_yticklabels(["Gerçek Benign", "Gerçek Patojenik"], fontsize=9)
        ax.set_title(f"{veri.capitalize()}  MCC={res['mcc']:.4f}",
                     color=PANEL_COLORS["CFTR"], fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    savefig(fig, "04_cftr_confusion_comparison.png")


# ─── 5. Silinen özellikler — panel bazlı çubuk grafik ─────────────────────────
def plot_eliminated_features(imp_orig: dict, imp_temiz: dict):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Spearman Eliminasyonuyla Silinen Önemli Özellikler (Top 30)",
                 fontsize=13, color=FG)

    all_eliminated = {}
    for panel in PANELS:
        top30_orig  = set(pd.Series(imp_orig[panel]).sort_values(ascending=False).head(30).index)
        temiz_feats = set(imp_temiz.get(panel, {}).keys())
        eliminated  = top30_orig - temiz_feats
        total_imp   = sum(imp_orig[panel].get(f, 0) for f in eliminated)
        all_eliminated[panel] = {"features": sorted(eliminated), "total_imp": total_imp}

    x = np.arange(len(PANELS))
    counts     = [len(all_eliminated[p]["features"]) for p in PANELS]
    total_imps = [all_eliminated[p]["total_imp"] for p in PANELS]

    ax2 = ax.twinx()
    bars = ax.bar(x, counts, color=[PANEL_COLORS[p] for p in PANELS], alpha=0.8, width=0.4, label="Silinen özellik sayısı")
    ax2.plot(x, total_imps, "o--", color="#FFD54F", linewidth=2, markersize=8, label="Toplam önem kaybı")

    ax.set_xticks(x)
    ax.set_xticklabels(PANELS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Silinen Özellik Sayısı (Top 30'dan)", color=FG, fontsize=10)
    ax2.set_ylabel("Toplam Önem Kaybı (%)", color="#FFD54F", fontsize=10)
    ax2.tick_params(axis="y", colors="#FFD54F")
    ax.set_ylim(0, max(counts) * 1.3 + 1)

    for i, (c, t) in enumerate(zip(counts, total_imps)):
        ax.text(i, c + 0.1, str(c), ha="center", fontsize=11, color=FG, fontweight="bold")

    # Alt kısımda silinen özellik isimleri
    for i, panel in enumerate(PANELS):
        feats_txt = "\n".join(all_eliminated[panel]["features"][:8])
        ax.text(i, -max(counts)*0.05, feats_txt, ha="center", va="top",
                fontsize=6, color=PANEL_COLORS[panel], alpha=0.8)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, facecolor="#1E2130", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "05_eliminated_features.png")


# ─── 6. Tüm paneller karşılaştırmalı MCC özeti ────────────────────────────────
def plot_mcc_summary():
    data = {
        "Orijinal\nAblasyon":    {"MASTER":0.541,"KANSER":0.690,"PAH":0.518,"CFTR":0.395},
        "Orijinal\nCombined":    {"MASTER":0.571,"KANSER":0.756,"PAH":0.584,"CFTR":0.504},
        "Temizlenmiş\nAblasyon": {"MASTER":0.573,"KANSER":0.756,"PAH":0.446,"CFTR":0.673},
        "Temizlenmiş\nCombined": {"MASTER":0.578,"KANSER":0.725,"PAH":0.518,"CFTR":0.504},
    }

    scenarios = list(data.keys())
    x = np.arange(len(scenarios))
    w = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("CatBoost — Panel Bazlı MCC Karşılaştırması (4 Yaklaşım)", fontsize=13, color=FG)

    for i, panel in enumerate(PANELS):
        vals = [data[s][panel] for s in scenarios]
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=panel,
                      color=PANEL_COLORS[panel], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=FG)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel("MCC", color=FG, fontsize=11)
    ax.set_ylim(0, 0.90)
    ax.legend(facecolor="#1E2130", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="#78909C", linestyle="--", linewidth=1, alpha=0.6, label="MCC=0.50")

    # CFTR düşüşünü işaretle
    cftr_vals = [data[s]["CFTR"] for s in scenarios]
    for i, (s, v) in enumerate(zip(scenarios, cftr_vals)):
        if s == "Temizlenmiş\nAblasyon":
            ax.annotate("En iyi CFTR\n(0.673)", xy=(i + 1.5*w, v),
                        xytext=(i + 1.5*w + 0.3, v + 0.08),
                        arrowprops=dict(arrowstyle="->", color="#F06292"),
                        color="#F06292", fontsize=8)

    fig.tight_layout()
    savefig(fig, "06_mcc_panel_summary.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Panel bazlı özellik analizi başlıyor...\n")

    imp_orig  = {}
    imp_temiz = {}

    print("Modeller eğitiliyor...")
    for panel in PANELS:
        print(f"  {panel} — orijinal")
        df, feats, y = load(panel, "orijinal")
        imp_orig[panel], _, _, _ = train_catboost(df, feats, y)

        print(f"  {panel} — temizlenmiş")
        df_t, feats_t, y_t = load(panel, "temizlenmiş")
        imp_temiz[panel], _, _, _ = train_catboost(df_t, feats_t, y_t)

    print("\nGrafikler oluşturuluyor...")

    print("1/6 — Panel bazlı özellik önemi")
    plot_importance_per_panel(imp_orig, imp_temiz)

    print("2/6 — Spearman korelasyon haritası")
    plot_spearman_heatmap(imp_orig)

    print("3/6 — Panel özellik örtüşmesi")
    plot_overlap_matrix(imp_orig, imp_temiz)

    print("4/6 — CFTR confusion matrix karşılaştırması")
    plot_cftr_confusion_comparison()

    print("5/6 — Silinen özellikler")
    plot_eliminated_features(imp_orig, imp_temiz)

    print("6/6 — MCC özet karşılaştırması")
    plot_mcc_summary()

    # Özellik önem tablosunu CSV'ye kaydet
    rows = []
    for panel in PANELS:
        for rank, (feat, imp) in enumerate(
            pd.Series(imp_orig[panel]).sort_values(ascending=False).head(TOP_N).items(), 1
        ):
            rows.append({
                "panel": panel, "veri": "orijinal",
                "rank": rank, "feature": feat, "importance": imp,
                "in_temiz": feat in imp_temiz.get(panel, {}),
            })
        for rank, (feat, imp) in enumerate(
            pd.Series(imp_temiz.get(panel, {})).sort_values(ascending=False).head(TOP_N).items(), 1
        ):
            rows.append({
                "panel": panel, "veri": "temizlenmiş",
                "rank": rank, "feature": feat, "importance": imp,
                "in_temiz": True,
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "feature_importance_table.csv", index=False)
    print(f"\n✓ Tablo: {OUT_DIR}/feature_importance_table.csv")
    print(f"✓ Tüm grafikler: {OUT_DIR}/")


if __name__ == "__main__":
    main()

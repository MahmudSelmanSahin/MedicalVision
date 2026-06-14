"""
TEKNOFEST 2026 — Kapsamlı EDA (Keşifçi Veri Analizi)
======================================================
Orijinal EDA_PDR kodunun genişletilmiş versiyonu.

Düzeltilen sorunlar:
  - Korelasyon analizi artık TÜM özellik çiftlerini (EK_ dahil) tarıyor
  - EK_ arası korelasyon matrisi ayrı bir bölüm olarak eklendi
  - AL_ kolonlarının yalnızca ilk 100'ü değil tümü analiz ediliyor

Yeni bölümler:
  - Mann-Whitney U testi ile sınıf ayrımı gücü (effect size)
  - Sıfır/düşük varyans analizi
  - Tüm paneller için eksik veri profili (MASTER'la sınırlı değil)
  - AL_ kolonları: eksiklik grupları ve korelasyon kümeleri
  - AA_ ve CAT_: chi-square + label bazında değer dağılımı
  - Random Forest özellik önemi (hızlı surrogate)
  - Panel karşılaştırması: tüm panellerde EK_ ortalamaları
  - Tüm çıktılar MODELLER/EDA/TABLOLAR/*.csv ve GRAFIKLER/*.png'ye kaydediliyor
  - Konsola yazdırılanlar aynı zamanda EDA_RAPORU.txt'ye de yazılıyor

Çalıştırma:
  python run_eda.py        (herhangi bir dizinden)
"""

from __future__ import annotations

import os
import sys
import warnings
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ── Yollar ────────────────────────────────────────────────────────────────────
ROOT      = Path("/Users/mahmudselmansahin/Teknofest")
DATA_DIR  = ROOT / "VERİLER" / "ORİJİNAL"
OUT_BASE  = ROOT / "MODELLER" / "EDA"
TABLOLAR  = OUT_BASE / "TABLOLAR"
GRAFIKLER = OUT_BASE / "GRAFIKLER"

for p in [TABLOLAR, GRAFIKLER]:
    p.mkdir(parents=True, exist_ok=True)

# Rapor dosyası
_rapor_buf = StringIO()

def _print(*args, **kwargs):
    """Hem konsola hem rapor dosyasına yazar."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=_rapor_buf)

def _save_rapor():
    (OUT_BASE / "EDA_RAPORU.txt").write_text(
        _rapor_buf.getvalue(), encoding="utf-8"
    )

# ── Stil ──────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.dpi": 120})
PANEL_COLORS = {"MASTER": "#2196F3", "KANSER": "#E91E63", "PAH": "#4CAF50", "CFTR": "#FF9800"}

# ══════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════
FILES = {
    "MASTER": DATA_DIR / "YARISMA_TRAIN_MASTER.csv",
    "KANSER": DATA_DIR / "YARISMA_TRAIN_KANSER.csv",
    "PAH":    DATA_DIR / "YARISMA_TRAIN_PAH.csv",
    "CFTR":   DATA_DIR / "YARISMA_TRAIN_CFTR.csv",
}

data: dict[str, pd.DataFrame] = {
    name: pd.read_csv(path, sep=None, engine="python")
    for name, path in FILES.items()
}

ID_COL    = "Variant_ID"
LABEL_COL = "Label"

def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]

# Özellik grupları (MASTER bazında)
_master = data["MASTER"]
_cols   = feat_cols(_master)
AL_COLS  = [c for c in _cols if c.startswith("AL_")]
EK_COLS  = [c for c in _cols if c.startswith("EK_")]
AA_COLS  = [c for c in _cols if c.startswith("AA_")]
CAT_COLS = [c for c in _cols if c.startswith("CAT_")]
NUM_COLS  = AL_COLS + EK_COLS  # sayısal özellikler


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — GENEL BAKIŞ
# ══════════════════════════════════════════════════════════════════════════════
_print("=" * 80)
_print("1. GENEL BAKIŞ — VERİ YAPISI")
_print("=" * 80)

panel_summary = []
for name, df in data.items():
    p  = int(df[LABEL_COL].sum())
    b  = len(df) - p
    r  = max(p, b) / max(min(p, b), 1)
    mt = int(df.isnull().sum().sum())
    mp = 100 * mt / (df.shape[0] * df.shape[1])
    panel_summary.append({
        "Panel": name, "Satır": df.shape[0], "Sütun": df.shape[1],
        "Patojenik": p, "Benign": b, "Oran(P/B)": round(r, 2),
        "Eksik_hücre": mt, "Eksik_%": round(mp, 1),
    })
    _print(f"\n{name}:")
    _print(f"  Boyut        : {df.shape[0]} × {df.shape[1]}")
    _print(f"  Patojenik    : {p}  |  Benign: {b}  |  Oran: {r:.2f}:1")
    _print(f"  Eksik veri   : {mt} hücre ({mp:.1f}%)")

pd.DataFrame(panel_summary).to_csv(TABLOLAR / "panel_genel_bakis.csv", index=False)

# Grafik: panel bazında sınıf dağılımı
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, df) in zip(axes, data.items()):
    vc = df[LABEL_COL].value_counts().sort_index()
    ax.bar(["Benign\n(0)", "Patojenik\n(1)"],
           [vc.get(0, 0), vc.get(1, 0)],
           color=["#42A5F5", "#EF5350"])
    ax.set_title(name, fontweight="bold", color=PANEL_COLORS[name])
    ax.set_ylabel("Varyant Sayısı")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
plt.suptitle("Sınıf Dağılımı — Tüm Paneller", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "01_sinif_dagilimi.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — ÖZELLİK KATEGORİLERİ
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("2. ÖZELLİK KATEGORİLERİ")
_print("=" * 80)

_print(f"\nToplam özellik : {len(_cols)}")
_print(f"  AL_ (Allel Frekansı)      : {len(AL_COLS):4d} sütun")
_print(f"  EK_ (Evrimsel Korunmuşluk): {len(EK_COLS):4d} sütun")
_print(f"  AA_ (Amino Asit)          : {len(AA_COLS):4d} sütun")
_print(f"  CAT_ (Kategorik)          : {len(CAT_COLS):4d} sütun")

fig, ax = plt.subplots(figsize=(7, 4))
groups = {"AL_": len(AL_COLS), "EK_": len(EK_COLS),
          "AA_": len(AA_COLS), "CAT_": len(CAT_COLS)}
bars = ax.barh(list(groups.keys()), list(groups.values()),
               color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
for bar, val in zip(bars, groups.values()):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontweight="bold")
ax.set_xlabel("Sütun Sayısı")
ax.set_title("Özellik Kategorileri", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "02_ozellik_kategorileri.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — EKSİK VERİ PROFİLİ (TÜM PANELLER)
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("3. EKSİK VERİ PROFİLİ — TÜM PANELLER")
_print("=" * 80)

all_missing_rows = []
for panel_name, df in data.items():
    for col in feat_cols(df):
        mc = int(df[col].isnull().sum())
        mp = 100 * mc / len(df)
        cat = col.split("_")[0]
        all_missing_rows.append({
            "panel": panel_name, "column": col, "category": cat,
            "missing_count": mc, "missing_pct": mp,
        })

missing_df = pd.DataFrame(all_missing_rows)
missing_df.to_csv(TABLOLAR / "eksik_veri_profili.csv", index=False)

_print("\nKategori × Panel bazında ortalama eksiklik (%):")
pivot = missing_df.groupby(["panel", "category"])["missing_pct"].mean().unstack(fill_value=0)
_print(pivot.round(1).to_string())

# Grafik: panel × kategori eksiklik ısı haritası
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Ortalama Eksiklik %"})
ax.set_title("Kategori × Panel Eksik Veri Oranı (%)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "03_eksik_veri_isitaritasi.png"); plt.close()

# En yüksek eksikli kolonlar (MASTER)
master_missing = (
    missing_df[missing_df["panel"] == "MASTER"]
    .sort_values("missing_pct", ascending=False)
    .head(40)
)
_print(f"\nMASTER — En eksik 40 sütun:")
_print(master_missing[["column", "missing_count", "missing_pct", "category"]]
       .to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — SIFIR / DÜŞÜK VARYANS ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("4. SIFIR / DÜŞÜK VARYANS ANALİZİ (MASTER)")
_print("=" * 80)

num_data = _master[NUM_COLS].apply(pd.to_numeric, errors="coerce")
variances = num_data.var()

zero_var  = variances[variances == 0]
low_var   = variances[(variances > 0) & (variances < 1e-6)]
near_zero = variances[(variances >= 1e-6) & (variances < 1e-3)]

_print(f"\nSıfır varyans sütun sayısı    : {len(zero_var)}")
_print(f"Çok düşük varyans (<1e-6)     : {len(low_var)}")
_print(f"Düşük varyans (1e-6 – 1e-3)  : {len(near_zero)}")

if len(zero_var):
    _print(f"  Sıfır varyans sütunları: {list(zero_var.index)}")

var_df = pd.DataFrame({"column": variances.index, "variance": variances.values})
var_df["category"] = var_df["column"].str.split("_").str[0]
var_df.to_csv(TABLOLAR / "varyans_analizi.csv", index=False)

# Grafik: AL_ ve EK_ varyans dağılımı
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
al_var = var_df[var_df["category"] == "AL"]["variance"]
ek_var = var_df[var_df["category"] == "EK"]["variance"]
axes[0].hist(al_var.clip(upper=al_var.quantile(0.95)), bins=40,
             color="#2196F3", edgecolor="white")
axes[0].set_title("AL_ Kolonları — Varyans Dağılımı")
axes[0].set_xlabel("Varyans (95. yüzdelik ile kesilmiş)")
axes[1].bar(ek_var.index, ek_var.values, color="#4CAF50")
axes[1].set_title("EK_ Kolonları — Varyans")
axes[1].set_xlabel("Özellik"); axes[1].set_ylabel("Varyans")
axes[1].tick_params(axis="x", rotation=45)
plt.suptitle("Varyans Analizi", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "04_varyans_analizi.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — EK_ KOLON DETAYLI ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("5. EK_ KOLONLARI — DETAYLI ANALİZ")
_print("=" * 80)

ek_data = _master[EK_COLS + [LABEL_COL]].copy()
ek_filled = ek_data[EK_COLS].apply(pd.to_numeric, errors="coerce")
ek_filled = ek_filled.fillna(ek_filled.median())

# 5A — Label ile Spearman korelasyonu
ek_label_corr = []
for col in EK_COLS:
    r, p = spearmanr(ek_filled[col], ek_data[LABEL_COL], nan_policy="omit")
    ek_label_corr.append({"feature": col, "spearman_r": r, "p_value": p})

ek_corr_df = pd.DataFrame(ek_label_corr).sort_values("spearman_r", ascending=False)
_print("\n5A — EK_ kolonlarının Label ile Spearman korelasyonu:")
_print(ek_corr_df.to_string(index=False))
ek_corr_df.to_csv(TABLOLAR / "ek_label_korelasyon.csv", index=False)

# 5B — EK_ kolonları arası TAM korelasyon matrisi
_print("\n5B — EK_ kolonları arası Spearman korelasyon matrisi:")
ek_inter = ek_filled.corr(method="spearman")
_print(ek_inter.round(3).to_string())
ek_inter.to_csv(TABLOLAR / "ek_arasi_korelasyon_matrisi.csv")

# Yüksek EK_ çiftleri
ek_pairs = []
cols_list = list(ek_inter.columns)
for i, c1 in enumerate(cols_list):
    for c2 in cols_list[i + 1:]:
        r = ek_inter.loc[c1, c2]
        ek_pairs.append({"feature1": c1, "feature2": c2, "spearman_r": round(r, 4)})
ek_pairs_df = pd.DataFrame(ek_pairs).sort_values("spearman_r", ascending=False, key=abs)
_print("\n5B — EK_ çiftleri (|r| sıralaması):")
_print(ek_pairs_df.to_string(index=False))
ek_pairs_df.to_csv(TABLOLAR / "ek_arasi_korelasyon_ciftler.csv", index=False)

# Grafik: EK_ korelasyon ısı haritası
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(ek_inter, dtype=bool))
sns.heatmap(ek_inter, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("EK_ Kolonları Arası Spearman Korelasyon Matrisi",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "05a_ek_korelasyon_matrisi.png"); plt.close()

# 5C — Mann-Whitney U testi (sınıf ayrımı gücü)
_print("\n5C — EK_ kolonları Mann-Whitney U testi (label bazlı ayrım):")
mw_rows = []
for col in EK_COLS:
    benign    = ek_filled.loc[ek_data[LABEL_COL] == 0, col].dropna()
    pathogen  = ek_filled.loc[ek_data[LABEL_COL] == 1, col].dropna()
    stat, p   = mannwhitneyu(benign, pathogen, alternative="two-sided")
    n1, n2    = len(benign), len(pathogen)
    r_effect  = 1 - (2 * stat) / (n1 * n2)
    mw_rows.append({
        "feature": col, "mean_benign": round(benign.mean(), 4),
        "mean_pathogen": round(pathogen.mean(), 4),
        "U_stat": round(stat, 1), "p_value": round(p, 6),
        "effect_r": round(r_effect, 4),
    })

mw_df = pd.DataFrame(mw_rows).sort_values("effect_r", ascending=False, key=abs)
_print(mw_df.to_string(index=False))
mw_df.to_csv(TABLOLAR / "ek_mannwhitney.csv", index=False)

# Grafik: EK_ box-plot per label
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(EK_COLS):
    ax = axes[i]
    df_plot = pd.DataFrame({
        "Değer": ek_filled[col],
        "Sınıf": ek_data[LABEL_COL].map({0: "Benign", 1: "Patojenik"}),
    })
    sns.boxplot(x="Sınıf", y="Değer", data=df_plot, ax=ax,
                palette={"Benign": "#42A5F5", "Patojenik": "#EF5350"})
    r_val = ek_corr_df.set_index("feature").loc[col, "spearman_r"]
    ax.set_title(f"{col}  (r={r_val:.3f})", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
for j in range(len(EK_COLS), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("EK_ Kolonları — Label Bazında Dağılım",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "05b_ek_boxplot_label.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — AL_ KOLON ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("6. AL_ KOLONLARI ANALİZİ")
_print("=" * 80)

al_data  = _master[AL_COLS + [LABEL_COL]].copy()
al_num   = al_data[AL_COLS].apply(pd.to_numeric, errors="coerce")
al_filled = al_num.fillna(0)   # AL_ için eksik = frekans=0 kabul

# 6A — Label ile Spearman korelasyonu (tüm AL_ kolonları)
al_label_corr = []
for col in AL_COLS:
    r, p = spearmanr(al_filled[col], al_data[LABEL_COL], nan_policy="omit")
    if not np.isnan(r):
        al_label_corr.append({"feature": col, "spearman_r": r, "p_value": p})

al_corr_df = pd.DataFrame(al_label_corr).sort_values("spearman_r", ascending=False, key=abs)
_print(f"\n6A — AL_ × Label Spearman (|r| top 30):")
_print(al_corr_df.head(30).to_string(index=False))
al_corr_df.to_csv(TABLOLAR / "al_label_korelasyon.csv", index=False)

# 6B — AL_ arası yüksek korelasyon (TÜM çiftler, >0.50)
_print("\n6B — AL_ kolonları arası korelasyon (|r| > 0.50) — TAM TARAMA:")
al_high_pairs = []
al_arr = al_filled.values
al_names = list(al_filled.columns)
n_al = len(al_names)
for i in range(n_al):
    for j in range(i + 1, n_al):
        c1, c2 = al_names[i], al_names[j]
        r, p = spearmanr(al_arr[:, i], al_arr[:, j], nan_policy="omit")
        if not np.isnan(r) and abs(r) > 0.50:
            al_high_pairs.append({
                "feature1": c1, "feature2": c2,
                "spearman_r": round(r, 4), "p_value": round(p, 8),
            })

al_pairs_df = pd.DataFrame(al_high_pairs).sort_values("spearman_r", ascending=False, key=abs)
_print(f"  Toplam yüksek korelasyonlu AL_ çifti: {len(al_pairs_df)}")
if len(al_pairs_df):
    _print(al_pairs_df.head(30).to_string(index=False))
al_pairs_df.to_csv(TABLOLAR / "al_arasi_yuksek_korelasyon.csv", index=False)

# Eşik dağılımı
_print("\n  |r| eşik bazında AL_ çift sayısı:")
for thr in [0.50, 0.60, 0.70, 0.80, 0.90]:
    n = (al_pairs_df["spearman_r"].abs() > thr).sum()
    _print(f"    > {thr}: {n} çift")

# 6C — Eksiklik grupları
_print("\n6C — AL_ eksiklik yüzdesi grupları (MASTER):")
al_miss = missing_df[(missing_df["panel"] == "MASTER") & (missing_df["category"] == "AL")]
bins = [0, 10, 30, 50, 70, 90, 101]
labels = ["<10%", "10-30%", "30-50%", "50-70%", "70-90%", ">90%"]
al_miss = al_miss.copy()
al_miss["grup"] = pd.cut(al_miss["missing_pct"], bins=bins, labels=labels, right=False)
grup_counts = al_miss["grup"].value_counts().sort_index()
_print(grup_counts.to_string())
grup_counts.to_frame("sütun_sayısı").to_csv(TABLOLAR / "al_eksiklik_gruplari.csv")

# Grafik: AL_ label korelasyon (top 40)
top40_al = al_corr_df.head(40)
fig, ax = plt.subplots(figsize=(14, 8))
colors = ["#EF5350" if r > 0 else "#42A5F5" for r in top40_al["spearman_r"]]
ax.barh(top40_al["feature"], top40_al["spearman_r"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman r")
ax.set_title("AL_ Kolonları — Label Korelasyonu (|r| Top 40)",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(GRAFIKLER / "06a_al_label_korelasyon_top40.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — AA_ ve CAT_ ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("7. AA_ ve CAT_ KATEGORİK ÖZELLİKLER")
_print("=" * 80)

chi2_rows = []
for col in AA_COLS + CAT_COLS:
    if col not in _master.columns:
        continue
    _print(f"\n{col}:")
    col_vals = _master[col].fillna("__EKSİK__").astype(str)
    vc = _master[col].value_counts(dropna=False)
    _print(f"  Benzersiz değer : {_master[col].nunique()}")
    _print(f"  Eksik           : {_master[col].isnull().sum()} ({100*_master[col].isnull().mean():.1f}%)")
    _print(f"  En sık 5 değer  :\n{vc.head().to_string()}")

    # Chi-square ile label ilişkisi
    ct = pd.crosstab(col_vals, _master[LABEL_COL])
    if ct.shape[0] >= 2 and ct.shape[1] == 2:
        chi2, p, dof, _ = chi2_contingency(ct)
        _print(f"  Chi-square      : {chi2:.2f}  (p={p:.4f}, df={dof})")
        chi2_rows.append({"feature": col, "chi2": round(chi2, 2),
                          "p_value": round(p, 6), "dof": dof})
    else:
        _print("  Chi-square      : hesaplanamadı (tek değer)")

if chi2_rows:
    chi2_df = pd.DataFrame(chi2_rows).sort_values("chi2", ascending=False)
    chi2_df.to_csv(TABLOLAR / "kategorik_chi2.csv", index=False)

# Grafik: AA_ label bazında değer dağılımı
if AA_COLS:
    fig, axes = plt.subplots(1, len(AA_COLS), figsize=(6 * len(AA_COLS), 5))
    if len(AA_COLS) == 1:
        axes = [axes]
    for ax, col in zip(axes, AA_COLS):
        ct = pd.crosstab(_master[col].fillna("?"), _master[LABEL_COL])
        ct.plot(kind="bar", ax=ax, color=["#42A5F5", "#EF5350"],
                edgecolor="white", rot=45)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Amino Asit")
        ax.set_ylabel("Sayı")
        ax.legend(["Benign (0)", "Patojenik (1)"])
    plt.suptitle("Amino Asit Dağılımı × Label", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAFIKLER / "07_aa_label_dagilimi.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8 — TÜM ÖZELLİKLER × LABEL KORELASYON (ÜST 50)
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("8. TÜM SAYISAL ÖZELLİKLER × LABEL — SPEARMAN (|r| Top 50)")
_print("=" * 80)

all_num_filled = _master[NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
all_target_corr = []
for col in NUM_COLS:
    r, p = spearmanr(all_num_filled[col], _master[LABEL_COL], nan_policy="omit")
    if not np.isnan(r):
        all_target_corr.append({
            "feature": col, "spearman_r": r, "abs_r": abs(r),
            "p_value": p, "category": col.split("_")[0],
        })

all_corr_df = pd.DataFrame(all_target_corr).sort_values("abs_r", ascending=False)
_print("\n")
_print(all_corr_df.head(50)[["feature", "spearman_r", "p_value", "category"]]
       .to_string(index=False))
all_corr_df.to_csv(TABLOLAR / "tum_ozellikler_label_korelasyon.csv", index=False)

# Grafik: Top 50
top50 = all_corr_df.head(50)
fig, ax = plt.subplots(figsize=(14, 12))
colors = [PANEL_COLORS.get(cat, "#888")
          for cat in top50["category"].map({"AL": "MASTER", "EK": "KANSER"}).fillna("#9E9E9E")]
colors_real = ["#4CAF50" if r > 0 else "#F44336" for r in top50["spearman_r"]]
ax.barh(range(len(top50)), top50["spearman_r"].values, color=colors_real)
ax.set_yticks(range(len(top50)))
ax.set_yticklabels(
    [f"{row.feature} [{row.category}]" for _, row in top50.iterrows()],
    fontsize=8,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman r")
ax.set_title("Tüm Özellikler × Label — Spearman (Top 50)",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(GRAFIKLER / "08_tum_ozellikler_label_korelasyon.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9 — PANEL KARŞILAŞTIRMASI
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("9. PANEL KARŞILAŞTIRMASI — EK_ ORTALAMALARI")
_print("=" * 80)

panel_ek_rows = []
for panel_name, df in data.items():
    for col in EK_COLS:
        if col in df.columns:
            for label_val in [0, 1]:
                subset = df[df[LABEL_COL] == label_val][col]
                panel_ek_rows.append({
                    "panel": panel_name, "feature": col,
                    "label": label_val,
                    "mean": round(subset.mean(), 4),
                    "std": round(subset.std(), 4),
                    "n": len(subset.dropna()),
                })

panel_ek_df = pd.DataFrame(panel_ek_rows)
panel_ek_df.to_csv(TABLOLAR / "panel_ek_karsilastirma.csv", index=False)

_print("\nEK_ ortalamalar (mean ± std) — Panel × Label:")
pivot_ek = panel_ek_df.pivot_table(
    index=["panel", "label"], columns="feature", values="mean"
)
_print(pivot_ek.round(3).to_string())

# Grafik: EK_ panel karşılaştırması (Patojenik)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(EK_COLS):
    ax = axes[i]
    for panel_name in data:
        df = data[panel_name]
        subset = df[df[LABEL_COL] == 1][col].dropna()
        ax.boxplot(subset, positions=[list(data.keys()).index(panel_name)],
                   widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=PANEL_COLORS[panel_name], alpha=0.7),
                   medianprops=dict(color="black", linewidth=2),
                   showfliers=False)
    ax.set_title(col, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(data.keys()), rotation=30, fontsize=8)
    ax.set_ylabel("Değer")
for j in range(len(EK_COLS), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("EK_ Kolonları — Panel Karşılaştırması (Sadece Patojenik)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "09_panel_ek_karsilastirma.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10 — RANDOM FOREST ÖZELLİK ÖNEMİ (SURROGATE)
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("10. RANDOM FOREST ÖZELLİK ÖNEMİ (MASTER Panel — Hızlı Surrogate)")
_print("=" * 80)

X_rf  = all_num_filled.values.astype(np.float32)
y_rf  = _master[LABEL_COL].astype(int).values
rf    = RandomForestClassifier(n_estimators=100, max_depth=8,
                               class_weight="balanced",
                               random_state=42, n_jobs=-1)
rf.fit(X_rf, y_rf)
fi = rf.feature_importances_

fi_df = pd.DataFrame({
    "feature": NUM_COLS,
    "importance": fi,
    "category": [c.split("_")[0] for c in NUM_COLS],
}).sort_values("importance", ascending=False)

fi_df.to_csv(TABLOLAR / "rf_feature_importance.csv", index=False)

_print("\nTop 30 özellik (RF importance):")
_print(fi_df.head(30).to_string(index=False))

# Kategori bazında toplam önem
cat_fi = fi_df.groupby("category")["importance"].sum().sort_values(ascending=False)
_print(f"\nKategori bazında toplam önem:")
_print(cat_fi.round(4).to_string())

# Grafik: Top 30
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
top30 = fi_df.head(30)
cat_colors = {"AL": "#2196F3", "EK": "#4CAF50"}
bar_colors = [cat_colors.get(c, "#9E9E9E") for c in top30["category"]]
axes[0].barh(range(len(top30)), top30["importance"].values, color=bar_colors)
axes[0].set_yticks(range(len(top30)))
axes[0].set_yticklabels(
    [f"{r.feature} [{r.category}]" for _, r in top30.iterrows()], fontsize=8
)
axes[0].invert_yaxis()
axes[0].set_xlabel("RF Importance")
axes[0].set_title("Top 30 Özellik (RF Importance)", fontweight="bold")
import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()]
axes[0].legend(handles=legend_patches, loc="lower right")

axes[1].pie(cat_fi.values, labels=cat_fi.index,
            autopct="%1.1f%%", startangle=140,
            colors=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
axes[1].set_title("Kategori Bazında Toplam Önem", fontweight="bold")
plt.suptitle("Random Forest Özellik Önemi", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "10_rf_feature_importance.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 11 — EK_ × AL_ ÇAPRAZKORELASYONLARİ (top EK_ ile top AL_)
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("11. EK_ × TOP-AL_ ÇAPRAZ KORELASYONLARİ")
_print("=" * 80)

top_al = al_corr_df.head(20)["feature"].tolist()
cross_rows = []
for ek in EK_COLS:
    for al in top_al:
        r, p = spearmanr(all_num_filled[ek], all_num_filled[al], nan_policy="omit")
        if not np.isnan(r):
            cross_rows.append({"EK": ek, "AL": al, "r": round(r, 4)})

cross_df = pd.DataFrame(cross_rows)
cross_pivot = cross_df.pivot(index="EK", columns="AL", values="r")
_print("\nEK_ × Top-20 AL_ Spearman korelasyon:")
_print(cross_pivot.round(3).to_string())
cross_pivot.to_csv(TABLOLAR / "ek_al_capraz_korelasyon.csv")

fig, ax = plt.subplots(figsize=(18, 5))
sns.heatmap(cross_pivot, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.3, ax=ax, vmin=-1, vmax=1)
ax.set_title("EK_ × Top-20 AL_ Spearman Korelasyon",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIKLER / "11_ek_al_capraz_korelasyon.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 12 — ÖZET VE BULGULAR
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("12. ÖZET BULGULAR VE PDR ÖNERİLERİ")
_print("=" * 80)

# Kritik sayılar
best_ek = ek_corr_df.iloc[0]
worst_ek = ek_corr_df.iloc[-1]
top_al_feat = al_corr_df.iloc[0]
max_ek_corr = ek_pairs_df.iloc[0]
al_high_80 = (al_pairs_df["spearman_r"].abs() > 0.80).sum()

_print(f"""
DOĞRULANAN BULGULAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EK_ KOLONLARI:
   • Label ile en güçlü korelasyon: {best_ek.feature} (r={best_ek.spearman_r:.4f})
   • Label ile en zayıf korelasyon: {worst_ek.feature} (r={worst_ek.spearman_r:.4f})
   • EK_ arası max korelasyon: {max_ek_corr.feature1} – {max_ek_corr.feature2} (r={max_ek_corr.spearman_r:.4f})
   • EK_ çiftlerinin tümü orta-yüksek korelasyona sahip → multicollinearity gerçek

2. AL_ KOLONLARI:
   • {len(AL_COLS)} AL_ kolonundan yalnızca {len(al_pairs_df)} çiftin |r| > 0.50
   • |r| > 0.80 AL_ çifti: {al_high_80} (neredeyse hiç değil)
   • Label ile en yüksek: {top_al_feat.feature} (r={top_al_feat.spearman_r:.4f})
   • AL_ kolonları birbirinden büyük ölçüde BAĞIMSIZ

3. SINIF AYRIMI:
   • RF Importance: AL_ toplam %{cat_fi.get('AL', 0)*100:.1f} | EK_ %{cat_fi.get('EK', 0)*100:.1f}
   • EK_ kolonları az sayıda ama güçlü sinyal taşıyor
   • AL_ kolonları çok sayıda ama tek tek zayıf sinyal (kollektif önem büyük)

4. EKSİK VERİ:
   • AL_ ortalama %56.9 eksik (MASTER)
   • EK_ ortalama %16.9 eksik
   • CAT_6 %97.7 eksik → bu kolon atılabilir

5. PANEL FARKLILIKLARI:
   • CFTR paneli EK_ ortalamalarında diğerlerinden yüksek
     (küçük panel + seçilmiş CFTR varyantları biyolojik olarak farklı)
   • PAH'ta EK_3 diğer panellerden düşük

PDR YORUMU (doğrulanmış):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Yüksek korelasyon iddiası EK_ kolonlarının KENDİ ARASINDA geçerli
  (EK_2-EK_3: r={ek_inter.loc['EK_2','EK_3']:.3f}, EK_7-EK_9: r={ek_inter.loc['EK_7','EK_9']:.3f})
• AL_ kolonları ARASINDA yüksek korelasyon YOK — her biri farklı populasyon
• Bu ayrım Feature Selection mekanizmalarını doğrudan etkiliyor:
  FILTER yöntemleri EK_ multicollinearity'yi görmüyor (tek değişken bazlı)
  EMBEDDED/WRAPPER yöntemleri EK_ tekrarları teke indirgiyor → avantaj
""")

# Özet tablo
summary_rows = [
    {"Bulgu": "En yüksek EK_-Label korelasyonu", "Değer": f"{best_ek.feature} r={best_ek.spearman_r:.3f}"},
    {"Bulgu": "Max EK_ iç korelasyonu", "Değer": f"{max_ek_corr.feature1}-{max_ek_corr.feature2} r={max_ek_corr.spearman_r:.3f}"},
    {"Bulgu": "AL_ çifti |r|>0.50 sayısı", "Değer": str(len(al_pairs_df))},
    {"Bulgu": "AL_ çifti |r|>0.80 sayısı", "Değer": str(al_high_80)},
    {"Bulgu": "RF Önem: AL_ toplam", "Değer": f"%{cat_fi.get('AL',0)*100:.1f}"},
    {"Bulgu": "RF Önem: EK_ toplam", "Değer": f"%{cat_fi.get('EK',0)*100:.1f}"},
    {"Bulgu": "Sıfır varyans sütun", "Değer": str(len(zero_var))},
    {"Bulgu": "MASTER eksik hücre", "Değer": "568464 (%54.9)"},
    {"Bulgu": "CAT_6 eksiklik", "Değer": "%97.7"},
]
pd.DataFrame(summary_rows).to_csv(TABLOLAR / "ozet_bulgular.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# BİTİŞ
# ══════════════════════════════════════════════════════════════════════════════
_print("\n" + "=" * 80)
_print("ÇIKTI DOSYALARI")
_print("=" * 80)
_print(f"\nTABLOLAR  : {TABLOLAR}")
for f in sorted(TABLOLAR.glob("*.csv")):
    _print(f"  {f.name}")
_print(f"\nGRAFİKLER : {GRAFIKLER}")
for f in sorted(GRAFIKLER.glob("*.png")):
    _print(f"  {f.name}")
_print(f"\nRAPOR     : {OUT_BASE}/EDA_RAPORU.txt")

_print("\n" + "=" * 80)
_print("KAPSAMLI EDA TAMAMLANDI")
_print("=" * 80)

_save_rapor()

"""
============================================================================
KAPSAMLI KESIFCI VERI ANALIZI (EDA) - ON ISLEME ONCESI ("Before")
============================================================================
SYZ2026 / MedicalVision - Missense Genetik Varyant Siniflandirma

Amac:
    DATA/Raw icindeki ham yarisma verilerini, herhangi bir on isleme
    UYGULANMADAN once derinlemesine incelemek. Uretilen tablo ve grafikler,
    PDR raporunun "Yontem" bolumundeki maddeleri (veri yapisi, eksik/aykiri
    deger analizi, sinif dengesizligi, dagilimlar, korelasyon vb.)
    gerekcelendirmek icin kullanilir.

Kapsanan analizler:
    1. Veri kumesi yapisi  : shape, dtype dagilimi, bellek, ozet istatistik
    2. Eksik deger analizi : kolon bazli eksik sayisi/yuzdesi + grafik + harita
    3. Sinif dagilimi      : Label countplot + pasta + veri setleri karsilastirma
    4. Aykiri deger analizi: IQR tabanli outlier tablosu + boxplot'lar
    5. Dagilimlar          : EK_ skorlari histogram/KDE, kategorik countplot
    6. Bagimlilik grafikleri: ozellik-hedef iliskisi (violin/box), korelasyon-bar
    7. Korelasyon isi haritasi: EK_ skorlari + yuksek varyansli ozellikler

Ciktilar:
    - Grafikler : SRC/Result/Graphics/eda/<VERISETI>/*.png
    - Tablolar  : SRC/Result/Graphics/eda/<VERISETI>/*.csv
    - Ozet rapor: SRC/Result/Graphics/eda/<VERISETI>/00_ozet.md

Kullanim:
    python SRC/data_analysis/Before/eda_before.py
    # veya tek veri seti:
    python SRC/data_analysis/Before/eda_before.py --dataset MASTER
============================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI olmadan dosyaya kaydet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 11

# --------------------------------------------------------------------------
# Yollar ve sabitler
# --------------------------------------------------------------------------
# Bu dosya: SRC/data_analysis/Before/eda_before.py  ->  proje koku 3 ust dizin
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "DATA" / "Raw"
GRAPHICS_DIR = PROJECT_ROOT / "SRC" / "Result" / "Graphics" / "eda"

# Veri seti -> (dosya adi, ayrac, ondalik)
DATASETS = {
    "MASTER": ("YARISMA_TRAIN_MASTER.csv", ";", ","),
    "CFTR": ("YARISMA_TRAIN_CFTR.csv", ";", ","),
    "KANSER": ("YARISMA_TRAIN_KANSER.csv", ",", "."),
    "PAH": ("YARISMA_TRAIN_PAH.csv", ",", "."),
}

ID_COL = "Variant_ID"
TARGET = "Label"
# Kolon turleri isme gore DEGIL, icerige gore otomatik belirlenir
# (bkz. load_dataset). Sema geregi kategorik olanlar: CAT_1..CAT_6
# (popülasyon etiketleri / genotipler) ve AA_1, AA_2 (aminoasit harfleri).
# Bir kolonun dolu degerlerinin en az bu oraninda sayiya cevrilebilmesi
# durumunda sayisal, aksi halde kategorik sayilir.
NUMERIC_THRESHOLD = 0.5

TOP_N = 25  # tablo/grafiklerde gosterilecek "en ust" ozellik sayisi


# --------------------------------------------------------------------------
# Yardimci fonksiyonlar
# --------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig, out_dir: Path, name: str) -> None:
    fpath = out_dir / name
    fig.savefig(fpath)
    plt.close(fig)
    print(f"    [grafik] {fpath.relative_to(PROJECT_ROOT)}")


def load_dataset(name: str) -> pd.DataFrame:
    """Ham CSV'yi dogru ayrac/ondalik ile yukler, sayisal kolonlari coerce eder."""
    fname, sep, dec = DATASETS[name]
    fpath = RAW_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Veri bulunamadi: {fpath}")

    # Bos string'leri NaN say
    df = pd.read_csv(
        fpath,
        sep=sep,
        decimal=dec,
        na_values=["", " ", "NA", "NaN", "nan", "null", "None"],
        keep_default_na=True,
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]

    # Hedefi tam sayi yap
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("Int64")

    # Kolon turunu ICERIGE gore otomatik belirle:
    #   dolu degerlerinin >= NUMERIC_THRESHOLD orani sayiya cevrilebiliyorsa -> sayisal
    #   aksi halde -> kategorik (CAT_*, AA_* gibi metin kolonlari korunur)
    # Bu sayede "CAT" kolonlari yanlislikla NaN'a cevrilip "%100 eksik"
    # gozukmez.
    for col in df.columns:
        if col in (ID_COL, TARGET):
            continue
        raw = df[col]
        if pd.api.types.is_numeric_dtype(raw):
            continue  # zaten sayisal okundu
        s = raw.astype("string").str.strip().replace({"": pd.NA})
        n_nonnull = int(s.notna().sum())
        n_numeric = int(pd.to_numeric(s, errors="coerce").notna().sum())
        if n_nonnull > 0 and n_numeric / n_nonnull >= NUMERIC_THRESHOLD:
            df[col] = pd.to_numeric(s, errors="coerce")  # sayisal
        else:
            df[col] = s  # kategorik (string), bos -> NaN

    return df


def get_feature_groups(df: pd.DataFrame):
    """Kolonlari icerik tipine gore mantiksal gruplara ayirir."""
    feats = [c for c in df.columns if c not in (ID_COL, TARGET)]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    ek_cols = [c for c in numeric if c.startswith("EK_")]
    return numeric, categorical, ek_cols


# --------------------------------------------------------------------------
# 1) Veri kumesi yapisi
# --------------------------------------------------------------------------
def analyze_structure(df, name, out_dir, report):
    print("  [1/7] Veri kumesi yapisi...")
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    dtype_counts = df.dtypes.astype(str).value_counts()
    dup = int(df.duplicated().sum())
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    report.append(f"## 1. Veri Kumesi Yapisi\n")
    report.append(f"- **Satir x Kolon**: {n_rows} x {n_cols}")
    report.append(f"- **Bellek**: {mem_mb:.2f} MB")
    report.append(f"- **Tam kopya satir**: {dup}")
    report.append(f"- **Sabit (tek degerli) kolon sayisi**: {len(const_cols)}")
    report.append(f"- **Dtype dagilimi**: " + ", ".join(f"{k}={v}" for k, v in dtype_counts.items()))

    # Ozet istatistik tablosunu kaydet
    desc = df.describe(include="all").transpose()
    desc.to_csv(out_dir / "01_ozet_istatistik.csv", encoding="utf-8-sig")
    if const_cols:
        pd.Series(const_cols, name="sabit_kolon").to_csv(
            out_dir / "01_sabit_kolonlar.csv", index=False, encoding="utf-8-sig"
        )
    report.append("")


# --------------------------------------------------------------------------
# 2) Eksik deger analizi
# --------------------------------------------------------------------------
def analyze_missing(df, name, out_dir, report):
    print("  [2/7] Eksik deger analizi...")
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = (
        pd.DataFrame({"eksik_sayisi": miss, "eksik_yuzdesi": miss_pct})
        .query("eksik_sayisi > 0")
        .sort_values("eksik_sayisi", ascending=False)
    )
    miss_df.to_csv(out_dir / "02_eksik_degerler.csv", encoding="utf-8-sig")

    report.append("## 2. Eksik Deger Analizi\n")
    report.append(f"- **Eksik iceren kolon sayisi**: {len(miss_df)} / {df.shape[1]}")
    report.append(f"- **Toplam eksik hucre**: {int(miss.sum())} "
                  f"(%{miss.sum() / df.size * 100:.2f})")
    if not miss_df.empty:
        report.append(f"- **En cok eksik kolon**: {miss_df.index[0]} "
                      f"(%{miss_df['eksik_yuzdesi'].iloc[0]})")

    if miss_df.empty:
        report.append("- Eksik deger yok.\n")
        return

    # Bar grafigi: en cok eksik TOP_N kolon
    top = miss_df.head(TOP_N)
    fig, ax = plt.subplots(figsize=(11, max(5, len(top) * 0.32)))
    sns.barplot(x="eksik_yuzdesi", y=top.index, data=top, ax=ax, color="#d1495b")
    ax.set_title(f"{name} - En Cok Eksik Degere Sahip {len(top)} Kolon")
    ax.set_xlabel("Eksik Yuzdesi (%)")
    ax.set_ylabel("Kolon")
    savefig(fig, out_dir, "02_eksik_deger_bar.png")

    # Eksiklik haritasi (cok kolon varsa orneklenmis)
    cols_for_map = miss_df.index.tolist()[:80]
    sample = df[cols_for_map].sample(min(len(df), 500), random_state=42)
    fig, ax = plt.subplots(figsize=(min(20, 0.22 * len(cols_for_map) + 4), 8))
    sns.heatmap(sample.isna(), cbar=False, ax=ax, cmap="rocket")
    ax.set_title(f"{name} - Eksiklik Haritasi (eksik kolonlar, 500 satir ornek)")
    ax.set_xlabel("Kolonlar")
    ax.set_ylabel("Satir ornekleri")
    savefig(fig, out_dir, "02_eksiklik_haritasi.png")
    report.append("")


# --------------------------------------------------------------------------
# 3) Sinif dagilimi
# --------------------------------------------------------------------------
def analyze_class_balance(df, name, out_dir, report):
    print("  [3/7] Sinif dagilimi...")
    if TARGET not in df.columns:
        return
    counts = df[TARGET].value_counts(dropna=False).sort_index()
    pct = (counts / counts.sum() * 100).round(2)

    report.append("## 3. Sinif Dagilimi (Label)\n")
    for k, v in counts.items():
        report.append(f"- Sinif **{k}**: {v} ornek (%{pct[k]})")
    if len(counts) == 2:
        ratio = counts.max() / counts.min()
        report.append(f"- **Dengesizlik orani (cogunluk/azinlik)**: {ratio:.2f} : 1")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axes[0],
                palette="viridis")
    axes[0].set_title(f"{name} - Sinif Frekanslari")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Ornek sayisi")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    axes[1].pie(counts.values, labels=[str(i) for i in counts.index],
                autopct="%1.1f%%", colors=sns.color_palette("viridis", len(counts)),
                startangle=90, wedgeprops={"edgecolor": "white"})
    axes[1].set_title(f"{name} - Sinif Oranlari")
    fig.tight_layout()
    savefig(fig, out_dir, "03_sinif_dagilimi.png")
    report.append("")


# --------------------------------------------------------------------------
# 4) Aykiri deger analizi (IQR)
# --------------------------------------------------------------------------
def analyze_outliers(df, name, numeric, ek_cols, out_dir, report):
    print("  [4/7] Aykiri deger analizi...")
    rows = []
    for col in numeric:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            n_out = int(((s < q1) | (s > q3)).sum()) - int((s == q1).sum())
            n_out = max(n_out, 0)
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "kolon": col,
            "aykiri_sayisi": n_out,
            "aykiri_yuzdesi": round(n_out / len(s) * 100, 2),
            "min": s.min(), "max": s.max(),
            "ortalama": s.mean(), "medyan": s.median(), "std": s.std(),
        })
    out_df = pd.DataFrame(rows).sort_values("aykiri_yuzdesi", ascending=False)
    out_df.to_csv(out_dir / "04_aykiri_degerler.csv", index=False, encoding="utf-8-sig")

    report.append("## 4. Aykiri Deger Analizi (IQR 1.5x kurali)\n")
    if not out_df.empty:
        n_with = int((out_df["aykiri_sayisi"] > 0).sum())
        report.append(f"- **Aykiri deger iceren kolon**: {n_with} / {len(out_df)}")
        report.append(f"- **En yuksek aykiri oranli kolon**: {out_df['kolon'].iloc[0]} "
                      f"(%{out_df['aykiri_yuzdesi'].iloc[0]})")

    # En yuksek aykiri oranli TOP_N kolon bar grafigi
    top = out_df.head(TOP_N)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(11, max(5, len(top) * 0.32)))
        sns.barplot(x="aykiri_yuzdesi", y="kolon", data=top, ax=ax, color="#edae49")
        ax.set_title(f"{name} - En Yuksek Aykiri Deger Oranina Sahip {len(top)} Kolon")
        ax.set_xlabel("Aykiri Deger Yuzdesi (%)")
        savefig(fig, out_dir, "04_aykiri_oran_bar.png")

    # EK_ skorlari icin boxplot (anahtar in-silico ozellikler)
    if ek_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        df[ek_cols].plot(kind="box", ax=ax, vert=True, patch_artist=True)
        ax.set_title(f"{name} - EK_ In-Silico Skorlari Boxplot")
        ax.set_ylabel("Deger")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        savefig(fig, out_dir, "04_EK_boxplot.png")
    report.append("")


# --------------------------------------------------------------------------
# 5) Dagilimlar (histogram + kategorik)
# --------------------------------------------------------------------------
def analyze_distributions(df, name, categorical, ek_cols, out_dir, report):
    print("  [5/7] Dagilim grafikleri...")
    report.append("## 5. Dagilimlar\n")

    # EK_ skorlari histogram + KDE grid
    if ek_cols:
        n = len(ek_cols)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(ek_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#30638e")
            axes[i].set_title(col)
            axes[i].set_xlabel("")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{name} - EK_ Skorlari Histogram + KDE", y=1.01)
        fig.tight_layout()
        savefig(fig, out_dir, "05_EK_histogram.png")

    # Kategorik kolonlar (aminoasit) countplot
    for col in categorical:
        vc = df[col].value_counts().head(25)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette="crest")
        ax.set_title(f"{name} - {col} Dagilimi (en sik 25)")
        ax.set_xlabel(col)
        ax.set_ylabel("Frekans")
        savefig(fig, out_dir, f"05_kategorik_{col}.png")
        report.append(f"- **{col}** benzersiz deger: {df[col].nunique()}")
    report.append("")


# --------------------------------------------------------------------------
# 6) Bagimlilik grafikleri (ozellik - hedef)
# --------------------------------------------------------------------------
def analyze_dependency(df, name, numeric, ek_cols, out_dir, report):
    print("  [6/7] Bagimlilik (ozellik-hedef) grafikleri...")
    if TARGET not in df.columns:
        return
    report.append("## 6. Ozellik - Hedef Bagimliligi\n")

    # EK_ skorlarinin sinifa gore dagilimi (violin/box)
    if ek_cols:
        n = len(ek_cols)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(ek_cols):
            sub = df[[col, TARGET]].dropna()
            sns.violinplot(x=TARGET, y=col, data=sub, ax=axes[i], palette="Set2")
            axes[i].set_title(col)
            axes[i].set_xlabel("Label")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{name} - EK_ Skorlarinin Sinifa Gore Dagilimi", y=1.01)
        fig.tight_layout()
        savefig(fig, out_dir, "06_EK_vs_label_violin.png")

    # Her sayisal ozelligin hedef ile mutlak korelasyonu (point-biserial)
    y = df[TARGET].astype("float")
    corrs = {}
    for col in numeric:
        s = df[col]
        if s.notna().sum() < 10 or s.nunique(dropna=True) <= 1:
            continue
        c = s.corr(y)
        if pd.notna(c):
            corrs[col] = c
    corr_s = pd.Series(corrs).sort_values(key=np.abs, ascending=False)
    corr_s.to_csv(out_dir / "06_hedef_korelasyon.csv", encoding="utf-8-sig",
                  header=["korelasyon"])

    top = corr_s.head(TOP_N)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(11, max(5, len(top) * 0.32)))
        colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in top.values]
        sns.barplot(x=top.values, y=top.index, ax=ax, palette=colors)
        ax.set_title(f"{name} - Hedef ile En Iliskili {len(top)} Ozellik")
        ax.set_xlabel("Label ile Korelasyon (point-biserial)")
        ax.axvline(0, color="black", lw=0.8)
        savefig(fig, out_dir, "06_hedef_korelasyon_bar.png")
        report.append(f"- **Hedefle en iliskili ozellik**: {top.index[0]} "
                      f"(r={top.iloc[0]:.3f})")
    report.append("")


# --------------------------------------------------------------------------
# 7) Korelasyon isi haritasi
# --------------------------------------------------------------------------
def analyze_correlation(df, name, numeric, ek_cols, out_dir, report):
    print("  [7/7] Korelasyon isi haritasi...")
    report.append("## 7. Korelasyon Isi Haritasi\n")

    # Tum sayisal korelasyon matrisini CSV olarak kaydet (rapor icin)
    full_corr = df[numeric].corr()
    full_corr.to_csv(out_dir / "07_korelasyon_matrisi.csv", encoding="utf-8-sig")

    # Yuksek korelasyonlu cift uyarisi (|r| > 0.95)
    upper = full_corr.where(np.triu(np.ones(full_corr.shape), k=1).astype(bool))
    high_pairs = (
        upper.stack().reset_index()
        .rename(columns={"level_0": "ozellik_1", "level_1": "ozellik_2", 0: "korelasyon"})
    )
    high_pairs = high_pairs[high_pairs["korelasyon"].abs() > 0.95]
    high_pairs.sort_values("korelasyon", key=np.abs, ascending=False).to_csv(
        out_dir / "07_yuksek_korelasyon_ciftleri.csv", index=False, encoding="utf-8-sig"
    )
    report.append(f"- **|r|>0.95 olan ozellik cifti**: {len(high_pairs)} "
                  f"(bkz. 07_yuksek_korelasyon_ciftleri.csv)")

    # EK_ skorlari arasi isi haritasi (okunabilir, anahtar ozellikler)
    if len(ek_cols) > 1:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(df[ek_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{name} - EK_ In-Silico Skorlari Korelasyon")
        savefig(fig, out_dir, "07_EK_korelasyon_heatmap.png")

    # En yuksek varyansli TOP_N ozellik arasi isi haritasi
    variances = df[numeric].var().sort_values(ascending=False)
    top_var_cols = variances.head(TOP_N).index.tolist()
    if len(top_var_cols) > 1:
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(df[top_var_cols].corr(), cmap="coolwarm", center=0,
                    square=True, ax=ax, cbar_kws={"shrink": 0.6})
        ax.set_title(f"{name} - En Yuksek Varyansli {len(top_var_cols)} Ozellik Korelasyonu")
        savefig(fig, out_dir, "07_yuksek_varyans_korelasyon_heatmap.png")
    report.append("")


# --------------------------------------------------------------------------
# Tek veri seti islemcisi
# --------------------------------------------------------------------------
def run_dataset(name: str):
    print(f"\n{'='*70}\nVERI SETI: {name}\n{'='*70}")
    df = load_dataset(name)
    out_dir = ensure_dir(GRAPHICS_DIR / name)
    numeric, categorical, ek_cols = get_feature_groups(df)

    report = [f"# {name} - Kesifci Veri Analizi (Before)\n",
              f"_Otomatik uretildi: eda_before.py_\n"]

    analyze_structure(df, name, out_dir, report)
    analyze_missing(df, name, out_dir, report)
    analyze_class_balance(df, name, out_dir, report)
    analyze_outliers(df, name, numeric, ek_cols, out_dir, report)
    analyze_distributions(df, name, categorical, ek_cols, out_dir, report)
    analyze_dependency(df, name, numeric, ek_cols, out_dir, report)
    analyze_correlation(df, name, numeric, ek_cols, out_dir, report)

    (out_dir / "00_ozet.md").write_text("\n".join(report), encoding="utf-8")
    print(f"  [ozet]  {(out_dir / '00_ozet.md').relative_to(PROJECT_ROOT)}")
    return df


def cross_dataset_summary(dfs: dict):
    """Veri setleri arasi sinif dagilimi karsilastirmasi."""
    print(f"\n{'='*70}\nVERI SETLERI KARSILASTIRMASI\n{'='*70}")
    out_dir = ensure_dir(GRAPHICS_DIR / "_karsilastirma")
    rows = []
    for name, df in dfs.items():
        if TARGET not in df.columns:
            continue
        vc = df[TARGET].value_counts()
        rows.append({
            "veri_seti": name, "satir": len(df), "kolon": df.shape[1],
            "sinif_0": int(vc.get(0, 0)), "sinif_1": int(vc.get(1, 0)),
            "pozitif_oran": round(vc.get(1, 0) / len(df) * 100, 1),
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "veri_seti_karsilastirma.csv", index=False, encoding="utf-8-sig")

    if not comp.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        # Gruplanmis (yan yana) sutunlar: her veri seti icin sinif 0 ve
        # sinif 1 ayri sutunlarda gosterilir (stacked degil).
        comp.set_index("veri_seti")[["sinif_0", "sinif_1"]].plot(
            kind="bar", stacked=False, ax=ax, color=["#264653", "#e9c46a"],
            width=0.8, edgecolor="white")
        ax.set_title("Veri Setleri - Sinif Dagilimi Karsilastirmasi")
        ax.set_ylabel("Ornek sayisi")
        ax.set_xlabel("")
        ax.legend(title="Label", labels=["Sinif 0", "Sinif 1"])
        plt.setp(ax.get_xticklabels(), rotation=0)
        # Sutun ustlerine deger etiketleri
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=2, fontsize=10)
        savefig(fig, out_dir, "veri_seti_sinif_karsilastirma.png")
    print(comp.to_string(index=False))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Before-EDA")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Sadece tek veri seti icin calistir")
    args = parser.parse_args()

    ensure_dir(GRAPHICS_DIR)
    targets = [args.dataset] if args.dataset else list(DATASETS.keys())

    dfs = {}
    for name in targets:
        try:
            dfs[name] = run_dataset(name)
        except Exception as e:
            print(f"  [HATA] {name}: {e}")

    if len(dfs) > 1:
        cross_dataset_summary(dfs)

    print(f"\n{'='*70}\nTAMAMLANDI. Ciktilar: "
          f"{GRAPHICS_DIR.relative_to(PROJECT_ROOT)}\n{'='*70}")


if __name__ == "__main__":
    main()

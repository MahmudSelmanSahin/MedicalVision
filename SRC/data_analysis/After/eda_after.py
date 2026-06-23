"""
============================================================================
KESIFCI VERI ANALIZI (EDA) - ON ISLEME SONRASI ("After")
============================================================================
SYZ2026 / MedicalVision - Missense Genetik Varyant Siniflandirma

Amac:
    DATA/Extraction-imputation icindeki ISLENMIS (imputation + ozellik
    cikarimi + tipatip ayni sutun temizligi yapilmis) modele hazir veri
    setlerini incelemek. "Before" EDA'nin aksine burada olcekleme sonrasi
    dagilimlar, kalan eksiklik, turetilen ozelliklerin hedefle iliskisi ve
    indirgenmis korelasyon yapisi gosterilir.

Analizler:
    1. Yapi           : shape, dtype, bellek, kalan ayni/sabit sutun
    2. Eksik deger    : isleme sonrasi (AL=0 dolu; EK senaryoya bagli)
    3. Sinif dagilimi : Label
    4. Dagilimlar     : EK (olceklenmis) + turetilen ozellikler (hist/countplot)
    5. Hedef iliskisi : tum ozellik-hedef korelasyonu + turetilen ozellik vurgusu
    6. Korelasyon     : EK + turetilen ozellikler isi haritasi + yuksek cift

Cikti: SRC/Result/Graphics/eda_after/<VERISETI>/<VARYANT>/
       (grafik PNG + tablo XLSX + 00_ozet.md)

Kullanim:
    python SRC/data_analysis/After/eda_after.py
    python SRC/data_analysis/After/eda_after.py --dataset MASTER --variant ek-robust_nan
    python SRC/data_analysis/After/eda_after.py --all-variants
============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 11

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EI_DIR = PROJECT_ROOT / "DATA" / "Extraction-imputation"
GRAPHICS_DIR = PROJECT_ROOT / "SRC" / "Result" / "Graphics" / "eda_after"

ID_COL = "Variant_ID"
TARGET = "Label"
DATASETS = ["MASTER", "CFTR", "KANSER", "PAH"]
VARIANTS = [f"ek-{s}_{m}" for s in ("robust", "standard") for m in ("median", "nan")]
DEFAULT_VARIANT = "ek-robust_nan"
TOP_N = 25

# feature_extraction'da uretilen turetilen ozellikler
ENGINEERED = [
    "is_exome", "CAT_1_te", "CAT_2_te", "is_segdup", "AL_185_present",
    "AA_1_te", "AA_2_te", "Delta_MW", "Delta_PI", "Delta_HYDRO", "Delta_Charge",
    "ek7_x_ek9", "ek2_ek3", "A", "T", "C", "G", "has_archaic_delta",
]


def savefig(fig, out_dir, name):
    fig.savefig(out_dir / name)
    plt.close(fig)
    print(f"    [grafik] {(out_dir / name).relative_to(PROJECT_ROOT)}")


def groups(df):
    ek = [c for c in df.columns if c.startswith("EK_")]
    eng = [c for c in ENGINEERED if c in df.columns]
    numeric = [c for c in df.columns
               if c not in (ID_COL, TARGET) and pd.api.types.is_numeric_dtype(df[c])]
    return ek, eng, numeric


# --------------------------------------------------------------------------
def analyze_structure(df, name, out_dir, report):
    print("  [1/6] Yapi...")
    n_rows, n_cols = df.shape
    mem = df.memory_usage(deep=True).sum() / 1024**2
    feats = [c for c in df.columns if c not in (ID_COL, TARGET)]
    dup = int(df[feats].T.duplicated().sum())
    const = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    report += [
        "## 1. Yapi\n",
        f"- **Satir x Kolon**: {n_rows} x {n_cols}",
        f"- **Bellek**: {mem:.2f} MB",
        f"- **Kalan tipatip ayni sutun**: {dup} (dedup sonrasi 0 beklenir)",
        f"- **Sabit (tek degerli) sutun**: {len(const)}",
        "",
    ]
    df.describe(include="all").transpose().to_excel(out_dir / "01_ozet_istatistik.xlsx")


def analyze_missing(df, name, out_dir, report):
    print("  [2/6] Eksik deger (isleme sonrasi)...")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    report += ["## 2. Eksik Deger (Isleme Sonrasi)\n",
               f"- **Eksik iceren kolon**: {len(miss)} / {df.shape[1]}",
               f"- **Toplam eksik hucre**: {int(df.isna().sum().sum())}", ""]
    if miss.empty:
        report.append("- Eksik deger yok (tum kolonlar dolu).\n")
        return
    miss_pct = (miss / len(df) * 100).round(2)
    pd.DataFrame({"eksik_sayisi": miss, "eksik_yuzdesi": miss_pct}).to_excel(
        out_dir / "02_eksik_degerler.xlsx")
    top = miss_pct.head(TOP_N)
    fig, ax = plt.subplots(figsize=(11, max(4, len(top) * 0.32)))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#d1495b")
    ax.set_title(f"{name} - Isleme Sonrasi Eksik Deger (en cok {len(top)})")
    ax.set_xlabel("Eksik Yuzdesi (%)")
    savefig(fig, out_dir, "02_eksik_deger_bar.png")


def analyze_class(df, name, out_dir, report):
    print("  [3/6] Sinif dagilimi...")
    if TARGET not in df.columns:
        return
    counts = df[TARGET].value_counts(dropna=False).sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    report.append("## 3. Sinif Dagilimi\n")
    for k, v in counts.items():
        report.append(f"- Sinif **{k}**: {v} (%{pct[k]})")
    report.append("")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="viridis")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title(f"{name} - Sinif Dagilimi")
    ax.set_xlabel("Label"); ax.set_ylabel("Ornek")
    savefig(fig, out_dir, "03_sinif_dagilimi.png")


def _grid(cols, df, title, out_dir, fname, kind):
    cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not cols:
        return
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(cols):
        if kind == "hist":
            sns.histplot(df[c].dropna(), kde=True, ax=axes[i], color="#30638e")
        else:
            vc = df[c].value_counts().sort_index()
            sns.barplot(x=vc.index.astype(str), y=vc.values, ax=axes[i],
                        palette="crest")
        axes[i].set_title(c); axes[i].set_xlabel("")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, y=1.01); fig.tight_layout()
    savefig(fig, out_dir, fname)


def analyze_distributions(df, name, ek, eng, out_dir, report):
    print("  [4/6] Dagilimlar...")
    report.append("## 4. Dagilimlar (Isleme Sonrasi)\n")
    # EK (olceklenmis) histogram
    _grid(ek, df, f"{name} - EK (olceklenmis) Dagilim", out_dir,
          "04_EK_dagilim.png", "hist")
    # Surekli turetilen ozellikler
    cont = [c for c in eng if df[c].nunique(dropna=True) > 2]
    _grid(cont, df, f"{name} - Surekli Turetilen Ozellikler", out_dir,
          "04_turetilen_surekli.png", "hist")
    # Ikili (binary) turetilen ozellikler
    binary = [c for c in eng if df[c].nunique(dropna=True) <= 2]
    _grid(binary, df, f"{name} - Ikili Turetilen Ozellikler", out_dir,
          "04_turetilen_ikili.png", "count")
    report.append(f"- Surekli turetilen: {len(cont)}, ikili turetilen: {len(binary)}\n")


def analyze_target_corr(df, name, eng, numeric, out_dir, report):
    print("  [5/6] Hedef iliskisi...")
    if TARGET not in df.columns:
        return
    report.append("## 5. Ozellik - Hedef Iliskisi\n")
    y = df[TARGET].astype(float)
    corrs = {}
    for c in numeric:
        if df[c].notna().sum() < 10 or df[c].nunique(dropna=True) <= 1:
            continue
        v = df[c].corr(y)
        if pd.notna(v):
            corrs[c] = v
    s = pd.Series(corrs).sort_values(key=np.abs, ascending=False)
    s.to_excel(out_dir / "05_hedef_korelasyon.xlsx", header=["korelasyon"])

    # Genel top-N
    top = s.head(TOP_N)
    fig, ax = plt.subplots(figsize=(11, max(4, len(top) * 0.32)))
    sns.barplot(x=top.values, y=top.index, ax=ax,
                palette=["#2a9d8f" if v > 0 else "#e76f51" for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"{name} - Hedefle En Iliskili {len(top)} Ozellik")
    ax.set_xlabel("Label ile korelasyon")
    savefig(fig, out_dir, "05_hedef_korelasyon_bar.png")

    # Sadece turetilen ozellikler
    eng_corr = s[[c for c in s.index if c in eng]].sort_values(key=np.abs,
                                                               ascending=False)
    if not eng_corr.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, len(eng_corr) * 0.34)))
        sns.barplot(x=eng_corr.values, y=eng_corr.index, ax=ax,
                    palette=["#2a9d8f" if v > 0 else "#e76f51"
                             for v in eng_corr.values])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{name} - Turetilen Ozelliklerin Hedefle Korelasyonu")
        ax.set_xlabel("Label ile korelasyon")
        savefig(fig, out_dir, "05_turetilen_hedef_korelasyon.png")
        report.append(f"- **Hedefle en iliskili turetilen ozellik**: "
                      f"{eng_corr.index[0]} (r={eng_corr.iloc[0]:.3f})")
    if not top.empty:
        report.append(f"- **Genel en iliskili ozellik**: {top.index[0]} "
                      f"(r={top.iloc[0]:.3f})")
    report.append("")


def analyze_correlation(df, name, ek, eng, out_dir, report):
    print("  [6/6] Korelasyon isi haritasi...")
    report.append("## 6. Korelasyon (EK + Turetilen)\n")
    cols = [c for c in (ek + eng) if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    if len(cols) < 2:
        report.append("- Yeterli sayisal kolon yok.\n")
        return
    corr = df[cols].corr()
    corr.to_excel(out_dir / "06_korelasyon_matrisi.xlsx")
    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.5),
                                    max(8, len(cols) * 0.45)))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=True,
                ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title(f"{name} - EK + Turetilen Ozellik Korelasyonu")
    savefig(fig, out_dir, "06_korelasyon_heatmap.png")

    # |r|>0.95 ciftler
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (upper.stack().reset_index()
             .rename(columns={"level_0": "ozellik_1", "level_1": "ozellik_2",
                              0: "korelasyon"}))
    pairs = pairs[pairs["korelasyon"].abs() > 0.95]
    pairs.to_excel(out_dir / "06_yuksek_korelasyon_ciftleri.xlsx", index=False)
    report.append(f"- **|r|>0.95 cift (EK+turetilen)**: {len(pairs)}\n")


def run(name, variant):
    path = EI_DIR / name / f"{name}_{variant}.xlsx"
    if not path.exists():
        print(f"  [atlandi] bulunamadi: {path.relative_to(PROJECT_ROOT)}")
        return
    print(f"\n{'='*70}\n{name} / {variant}\n{'='*70}")
    df = pd.read_excel(path)
    out_dir = GRAPHICS_DIR / name / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ek, eng, numeric = groups(df)
    report = [f"# {name} ({variant}) - EDA (After)\n",
              "_Otomatik uretildi: eda_after.py_\n"]
    analyze_structure(df, name, out_dir, report)
    analyze_missing(df, name, out_dir, report)
    analyze_class(df, name, out_dir, report)
    analyze_distributions(df, name, ek, eng, out_dir, report)
    analyze_target_corr(df, name, eng, numeric, out_dir, report)
    analyze_correlation(df, name, ek, eng, out_dir, report)
    (out_dir / "00_ozet.md").write_text("\n".join(report), encoding="utf-8")
    print(f"  [ozet]  {(out_dir / '00_ozet.md').relative_to(PROJECT_ROOT)}")


def main():
    p = argparse.ArgumentParser(description="After-EDA (islenmis veri)")
    p.add_argument("--dataset", choices=DATASETS, help="Tek veri seti")
    p.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT,
                   help=f"Imputation varyanti (vars: {DEFAULT_VARIANT})")
    p.add_argument("--all-variants", action="store_true",
                   help="Tum 4 varyanti analiz et")
    args = p.parse_args()

    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset else DATASETS
    variants = VARIANTS if args.all_variants else [args.variant]
    for name in datasets:
        for variant in variants:
            run(name, variant)
    print(f"\n{'='*70}\nTAMAMLANDI. Ciktilar: "
          f"{GRAPHICS_DIR.relative_to(PROJECT_ROOT)}\n{'='*70}")


if __name__ == "__main__":
    main()

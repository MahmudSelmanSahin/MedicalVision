"""
TEKNOFEST 2026 - Şartname Verisi Detaylı EDA
==============================================
Amaç: 4 panelin yapısını anlamak, eksik veri profilini çıkarmak,
      feature korelasyonlarını ve hedef ile ilişkiyi incelemek.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Stil ayarları
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ══════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════
files = {
    'MASTER': './YARISMA_TRAIN_MASTER.csv',
    'KANSER': './YARISMA_TRAIN_KANSER.csv',
    'PAH':    './YARISMA_TRAIN_PAH.csv',
    'CFTR':   './YARISMA_TRAIN_CFTR.csv',
}

data = {name: pd.read_csv(path, sep=None, engine='python')
        for name, path in files.items()}

print("=" * 80)
print("1. GENEL BAKIŞ - VERİ YAPISI")
print("=" * 80)

for name, df in data.items():
    print(f"\n{name} Panel:")
    print(f"  Boyut: {df.shape[0]} satır × {df.shape[1]} sütun")
    print(f"  Hedef değişken (Label): {df['Label'].value_counts().to_dict()}")
    
    p_count = df['Label'].sum()
    b_count = len(df) - p_count
    ratio = max(p_count, b_count) / min(p_count, b_count)
    print(f"  Dengesizlik: {ratio:.2f}:1 ({p_count}P / {b_count}B)")
    
    missing_total = df.isnull().sum().sum()
    missing_pct = 100 * missing_total / (df.shape[0] * df.shape[1])
    print(f"  Eksik veri: {missing_total} hücre ({missing_pct:.1f}%)")

# ══════════════════════════════════════════════════════════════
# 2. ÖZELLİK TİPLERİ ANALİZİ
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. ÖZELLİK TİPLERİ VE KATEGORİLER")
print("=" * 80)

df_master = data['MASTER']
columns = df_master.columns.tolist()
columns.remove('Variant_ID')
columns.remove('Label')

# Kategorize et
al_cols = [c for c in columns if c.startswith('AL_')]  # Allel frekansı
bk_cols = [c for c in columns if c.startswith('BK_')]  # Biyokimyasal
ek_cols = [c for c in columns if c.startswith('EK_')]  # Evrimsel korunmuşluk
ye_cols = [c for c in columns if c.startswith('YE_')]  # Yapısal etkiler
aa_cols = [c for c in columns if c.startswith('AA_')]  # Amino asit sekans
cat_cols = [c for c in columns if c.startswith('CAT_')] # Kategorik

print(f"\nToplam özellik: {len(columns)}")
print(f"  AL_ (Allel Frekansı):      {len(al_cols):3d} sütun")
print(f"  BK_ (Biyokimyasal):        {len(bk_cols):3d} sütun")
print(f"  EK_ (Evrimsel Korunmuşluk):{len(ek_cols):3d} sütun")
print(f"  YE_ (Yapısal Etkiler):     {len(ye_cols):3d} sütun")
print(f"  AA_ (Amino Asit Sekans):   {len(aa_cols):3d} sütun")
print(f"  CAT_ (Kategorik):          {len(cat_cols):3d} sütun")

# ══════════════════════════════════════════════════════════════
# 3. EKSİK VERİ ANALİZİ - DETAYLI
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. EKSİK VERİ PROFİLİ (MASTER Panel)")
print("=" * 80)

missing_stats = []
for col in columns:
    missing_count = df_master[col].isnull().sum()
    missing_pct = 100 * missing_count / len(df_master)
    if missing_count > 0:
        missing_stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'category': col.split('_')[0]
        })

missing_df = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)

print("\nEn çok eksik olan 40 sütun:")
print(missing_df.head(40).to_string(index=False))

print("\n\nKategori bazında eksik veri oranı:")
for cat in ['AL', 'BK', 'EK', 'YE', 'AA', 'CAT']:
    cat_missing = missing_df[missing_df['category'] == cat]['missing_pct']
    if len(cat_missing) > 0:
        print(f"  {cat}_: Ortalama {cat_missing.mean():.1f}% eksik "
              f"(min={cat_missing.min():.1f}%, max={cat_missing.max():.1f}%)")

# ══════════════════════════════════════════════════════════════
# 4. SAYISAL ÖZELLİK İSTATİSTİKLERİ
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. SAYISAL ÖZELLİK İSTATİSTİKLERİ (Label bazında)")
print("=" * 80)

numeric_cols = al_cols + bk_cols + ek_cols + ye_cols
numeric_data = df_master[numeric_cols + ['Label']].copy()

# Her sınıf için özet istatistikler
for label in [0, 1]:
    label_name = "Benign" if label == 0 else "Patojenik"
    subset = numeric_data[numeric_data['Label'] == label]
    
    print(f"\n{label_name} Sınıfı (n={len(subset)}):")
    
    # Kategori bazında ortalama değerler
    for cat, cols in [('AL (Allel Frekansı)', al_cols[:5]),  # ilk 5'i
                      ('BK (Biyokimyasal)', bk_cols[:3]),
                      ('EK (Evrimsel)', ek_cols[:3])]:
        if cols:
            vals = subset[cols].mean().mean()  # genel ortalama
            print(f"  {cat}: Ort. {vals:.6f}")

# ══════════════════════════════════════════════════════════════
# 5. KORELASYON ANALİZİ - KRİTİK
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. KORELASYON ANALİZİ")
print("=" * 80)

# Eksik veriyi medyanla doldur (sadece korelasyon hesabı için)
numeric_filled = numeric_data[numeric_cols].fillna(
    numeric_data[numeric_cols].median()
)

# Hedef ile korelasyon
target_corr = []
for col in numeric_cols:
    if col in numeric_filled.columns:
        corr, pval = spearmanr(numeric_filled[col], df_master['Label'], 
                                nan_policy='omit')
        if not np.isnan(corr):
            target_corr.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': pval
            })

target_corr_df = pd.DataFrame(target_corr).sort_values(
    'abs_correlation', ascending=False
)

print("\nHedef (Label) ile en yüksek korelasyonlu 20 özellik:")
print(target_corr_df.head(20)[['feature', 'correlation']].to_string(index=False))

# Özellikler arası yüksek korelasyon (multicollinearity)
print("\n\nÖzellikler arası yüksek korelasyon (>0.80):")
high_corr_pairs = []
for i, col1 in enumerate(numeric_cols[:100]):  # ilk 100 özellik
    for col2 in numeric_cols[i+1:100]:
        if col1 in numeric_filled.columns and col2 in numeric_filled.columns:
            corr, _ = spearmanr(numeric_filled[col1], numeric_filled[col2],
                                 nan_policy='omit')
            if abs(corr) > 0.80 and not np.isnan(corr):
                high_corr_pairs.append({
                    'feature1': col1,
                    'feature2': col2,
                    'correlation': corr
                })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
        'correlation', ascending=False, key=abs
    )
    print(high_corr_df.head(15).to_string(index=False))
else:
    print("  (İlk 100 özellikte >0.80 korelasyon bulunamadı)")

# ══════════════════════════════════════════════════════════════
# 6. KATEGORİK ÖZELLİKLER ANALİZİ
# ══════════════════════════════════════════════════════════════
if cat_cols:
    print("\n" + "=" * 80)
    print("6. KATEGORİK ÖZELLİKLER")
    print("=" * 80)
    
    for col in cat_cols[:5]:  # ilk 5 kategorik
        if col in df_master.columns:
            print(f"\n{col}:")
            print(f"  Benzersiz değer: {df_master[col].nunique()}")
            print(f"  En sık 5:")
            print(df_master[col].value_counts().head().to_string())

# ══════════════════════════════════════════════════════════════
# 7. AMINO ASİT SEKANS ANALİZİ
# ══════════════════════════════════════════════════════════════
if aa_cols:
    print("\n" + "=" * 80)
    print("7. AMİNO ASİT SEKANS ÖZELLİKLERİ")
    print("=" * 80)
    
    print(f"\nToplam {len(aa_cols)} sekans özelliği var.")
    print("İlk 5 örnek:")
    for col in aa_cols[:5]:
        if col in df_master.columns:
            sample_vals = df_master[col].dropna().head(3).tolist()
            print(f"  {col}: {sample_vals}")

# ══════════════════════════════════════════════════════════════
# 8. PANEL KARŞILAŞTIRMASI
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("8. PANEL KARŞILAŞTIRMASI - ÖZELLİK DAĞILIMLARI")
print("=" * 80)

# Her panelde aynı özelliklerin ortalamasını karşılaştır
sample_features = target_corr_df.head(5)['feature'].tolist()

comparison_data = []
for feat in sample_features:
    for panel_name, panel_df in data.items():
        if feat in panel_df.columns:
            mean_val = panel_df[feat].mean()
            comparison_data.append({
                'panel': panel_name,
                'feature': feat,
                'mean': mean_val
            })

comparison_df = pd.DataFrame(comparison_data)
print("\nEn önemli 5 özelliğin panel bazında ortalamaları:")
for feat in sample_features:
    print(f"\n{feat}:")
    subset = comparison_df[comparison_df['feature'] == feat]
    for _, row in subset.iterrows():
        print(f"  {row['panel']:<10}: {row['mean']:.6f}")

# ══════════════════════════════════════════════════════════════
# 9. ÖNERİLER VE SONUÇ
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("9. ÖNERİLER - PIPELINE STRATEJİSİ")
print("=" * 80)

print("""
KRİTİK BULGULAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DENGESİZLİK:
   • MASTER: 2.75:1 (orta)
   • PAH:    5.00:1 (ağır) ⚠️
   • CFTR:   4.29:1 (ağır) ⚠️
   ➜ Strateji: class_weight='balanced', SMOTE (dikkatli), MCC metrik

2. EKSİK VERİ:
   • Genel: %30-57 arası
   • AL_ sütunları en yoğun eksik
   ➜ Strateji: Ablasyon testi gerekli (sil/ham/doldur karşılaştırması)

3. ÖZELLİK SAYISI:
   • 353 sütun (ön çalışmadan 2.5x fazla)
   ➜ Strateji: Feature selection, PCA, veya AutoEncoder

4. KORELASYON:
   • Bazı özellikler >0.80 korelasyonlu (multicollinearity)
   ➜ Strateji: VIF analizi, yüksek korelasyonlu çiftlerden birini çıkar

5. KÜÇÜK PANELLER:
   • CFTR: sadece 111 satır (test 30+30=60 olacak)
   ➜ Strateji: Transfer Learning (MASTER → CFTR fine-tune)

ÖNERİLEN ADIMLAR (SIRA İLE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] Eksik Veri Ablasyon Testi (PDR için kritik)
    → 1A: Eksik satırları sil
    → 1B: Ham bırak (GBDT NaN tolere eder)
    → 1C: Medyan/0 doldur + Missing Indicator bayrak
    
[2] Normalizasyon Testi
    → AL_ sütunları: log1p + MinMax
    → BK_, EK_ sütunları: StandardScaler
    → Karşılaştır: Ham vs Normalize

[3] Özellik Seçimi
    → Feature Importance (CatBoost/XGBoost)
    → Yüksek korelasyonlu çiftlerden birini çıkar
    → Sıfır varyans kontrolü

[4] Model Seçimi
    → GBDT (XGBoost, LightGBM, CatBoost) - tabular veriye optimal
    → Ensemble (5 model + Meta-learner)
    → Transfer Learning (MASTER → küçük paneller)

[5] Dengesizlik Yönetimi
    → class_weight='balanced'
    → Eşik optimizasyonu (MCC ve F1 birlikte)
    → SMOTE (dikkatli, evrimsel bilgiyi bozabilir)

SONRAKİ ADIM: Ablasyon testine başla (Eksik veri stratejileri)
""")

print("\n" + "=" * 80)
print("EDA TAMAMLANDI")
print("=" * 80)
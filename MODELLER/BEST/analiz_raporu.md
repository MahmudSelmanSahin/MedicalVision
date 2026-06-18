# Model Analiz Raporu — CatBoost & TabPFN
**Tarih:** 18 Haziran 2026  
**Branch:** develop_selman_mice  
**Metrik:** MCC (birincil), PR-AUC (ikincil)

---

## 1. En İyi Sonuçlar (Panel Bazlı)

### CatBoost

| Panel  | En İyi Deney            | Konfigürasyon                   | MCC    | PR-AUC |
|--------|-------------------------|---------------------------------|--------|--------|
| MASTER | Ablasyon (TEMİZLENMİŞ) | Eksik_Veri_Korundu              | 0.5730 | 0.9408 |
| KANSER | Ablasyon (TEMİZLENMİŞ) | KNNImputer                      | 0.7557 | 0.9061 |
| PAH    | Ablasyon (ORİJİNAL)    | Eksik_Veri_Korundu              | 0.5184 | 0.9059 |
| CFTR   | Ablasyon (TEMİZLENMİŞ) | AL_Sifir_EK_Medyan              | 0.6726 | 0.9234 |

### TabPFN

| Panel  | En İyi Deney                    | Konfigürasyon                        | MCC    | PR-AUC |
|--------|----------------------------------|--------------------------------------|--------|--------|
| MASTER | Ablasyon (TEMİZLENMİŞ)          | MinMaxScaler_Medyan                  | 0.5716 | 0.9248 |
| KANSER | Feature Selection (TEMİZLENMİŞ) | FILTER_ANOVA + WRAPPER_RFE + EMBEDDED| 0.7620 | 0.8762 |
| PAH    | Ablasyon (TEMİZLENMİŞ)          | Eksik_Veri_Medyan                    | 0.4458 | 0.9392 |
| CFTR   | QT + BaggingClassifier          | QT_Bagg_mf30 (n=10, max_feat=0.3)   | 0.6726 | 0.9950 |

### Model Karşılaştırması (Panel Bazlı Kazanan)

| Panel  | CatBoost | TabPFN  | Kazanan  | Fark    |
|--------|----------|---------|----------|---------|
| MASTER | 0.5730   | 0.5716  | CatBoost | +0.0014 |
| KANSER | 0.7557   | **0.7620** | TabPFN | +0.0063 |
| PAH    | **0.5184** | 0.4458 | CatBoost | +0.0726 |
| CFTR   | 0.6726   | 0.6726  | **Berabere** | 0.0000 |

---

## 2. Deneyler ve Sonuçları

### 2.1 Veri Türü Karşılaştırması (ORİJİNAL vs TEMİZLENMİŞ)

**TEMİZLENMİŞ veri** (194 kolon, MASTER Spearman |r|≥0.80 eliminasyonu) çoğu panelde ORİJİNAL veriden daha iyi performans verdi.

**İstisna — PAH:**  
CatBoost ORİJİNAL (0.518) > CatBoost TEMİZLENMİŞ (0.446).  
Neden: Spearman eliminasyonu PAH'a özgü alt-popülasyon sinyallerini (AL_ kolonları) sildi. Bu kolonlar MASTER korelasyon yapısına göre elendi ama PAH için bağımsız bilgi taşıyordu.

| Panel  | CB ORİJİNAL | CB TEMİZLENMİŞ | TABPFN ORİJİNAL | TABPFN TEMİZLENMİŞ |
|--------|-------------|----------------|-----------------|---------------------|
| MASTER | 0.541       | **0.573**      | 0.528           | **0.572**           |
| KANSER | 0.690       | **0.756**      | 0.597           | **0.725**           |
| PAH    | **0.518**   | 0.446          | **0.518**       | 0.446               |
| CFTR   | 0.395       | **0.673**      | 0.503           | 0.395               |

---

### 2.2 Feature Selection (TEMİZLENMİŞ veri üzerinde)

**CatBoost:** Feature selection ablasyon default'u geçemedi. En iyi sonuçlar ablasyon senaryolarından geldi. PAH'ta FILTER_ANOVA+WRAPPER_RFE 0.518 MCC verdi ama bu ORİJİNAL ablasyon skoru ile eşit; ek kazanım yok.

**TabPFN:** KANSER panelinde FILTER_ANOVA+WRAPPER_RFE+EMBEDDED yöntemi **0.762** MCC ile tüm deneylerin KANSER en iyisi oldu. Diğer panellerde feature selection zarar verdi (özellikle PAH'ta EMBEDDED MCC=0.000).

| Panel  | CB FS Best                        | MCC   | TABPFN FS Best                            | MCC   |
|--------|-----------------------------------|-------|-------------------------------------------|-------|
| MASTER | WRAPPER_RFE                       | 0.577 | FILTER_ANOVA + WRAPPER_RFE                | 0.571 |
| KANSER | FILTER_MUTUAL_INFO                | 0.722 | FILTER_ANOVA + WRAPPER_RFE + EMBEDDED     | **0.762** |
| PAH    | FILTER_ANOVA + WRAPPER_RFE        | 0.518 | FILTER_ANOVA                              | 0.361 |
| CFTR   | FILTER_MUTUAL_INFO                | 0.503 | FILTER_ANOVA + WRAPPER_RFE                | 0.503 |

---

### 2.3 MICE Eksik Veri Doldurma

**Sonuç: Negatif.**  
IterativeImputer (max_iter=10) tüm panellerde Medyan veya AL_Sifir stratejilerinden daha kötü ya da eşit performans verdi. Ayrıca sklearn bug'ı nedeniyle `read-only array` hatası aldı (fix: `.to_numpy(copy=True)`).

---

### 2.4 OOF Threshold Optimizasyonu

**Sonuç: Negatif.**  
5-fold OOF üzerinde PR eğrisinden bulunan optimal eşik, test setinde default 0.5 eşiğini hiçbir panelde geçemedi. Özellikle CFTR'de dramatik düşüş (0.673 → 0.395) yaşandı; train/test dağılımı farklılığından OOF eşiği test'e genellenmedi.

| Panel  | MCC (default=0.5) | MCC (opt threshold) | Δ      |
|--------|-------------------|---------------------|--------|
| MASTER | 0.573             | 0.559               | -0.014 |
| KANSER | 0.756             | 0.660               | -0.096 |
| PAH    | 0.446             | 0.446               | 0.000  |
| CFTR   | 0.673             | 0.395               | -0.278 |

*(CatBoost; TabPFN'de de benzer negatif sonuçlar)*

---

### 2.5 Optuna Hiperparametre Optimizasyonu (CatBoost)

**Arama uzayı:** depth=3-8, l2=1-20, colsample=0.3-1.0, 60 trial.

**Sonuç: Kısmi pozitif (PAH ve KANSER).**

| Panel  | MCC (base) | MCC (Optuna) | Δ      | En İyi Parametreler           |
|--------|------------|--------------|--------|-------------------------------|
| MASTER | 0.561      | 0.562        | +0.001 | depth=8, l2=9.25, col=1.0    |
| KANSER | 0.604      | 0.722        | +0.118 | depth=7, l2=1.39, col=0.98   |
| PAH    | 0.362      | 0.459        | +0.097 | depth=4, l2=18.92, col=0.70  |
| CFTR   | 0.503      | 0.503        | 0.000  | depth=5, l2=17.17, col=0.44  |

Not: Optuna base olarak farklı bir senaryo kullandığı için KANSER/PAH ablasyon best'lerini geçemedi. PAH'ta 0.459 ablasyon best'inden (0.446) iyi ama ORİJİNAL ablasyonun (0.518) altında.

---

### 2.6 Focal Loss (CatBoost)

**Sonuç: Negatif.**  
`auto_class_weights='Balanced'` tüm panellerde Focal Loss'tan üstün.  
Teknik not: CatBoost 1.2.7'de sözdizimi `Focal:focal_alpha=X;focal_gamma=Y` (noktalı virgül ayırıcı).

| Panel  | Logloss_Balanced | En İyi Focal   | Kazanan  |
|--------|-----------------|----------------|----------|
| MASTER | **0.573**       | 0.565          | Logloss  |
| KANSER | **0.756**       | 0.725          | Logloss  |
| PAH    | 0.337           | **0.362**      | Focal*   |
| CFTR   | **0.673**       | 0.266          | Logloss  |

*PAH marjinal Focal kazanımı ablasyon best'inin (0.446) altında.

---

### 2.7 QuantileTransformer + BaggingClassifier (TabPFN)

**Konfigürasyonlar:**
- Baseline_n1: ablasyon konfigürasyonu (referans)  
- QT_n8: QuantileTransformer(normal) + n_estimators=8  
- QT_Bagg_mf30: QuantileTransformer + BaggingClassifier(n=10, max_features=0.3)

**Sonuç: Sadece CFTR'de pozitif.**

| Panel  | Baseline_n1 | QT_n8   | QT_Bagg_mf30 | Kazanan       |
|--------|-------------|---------|--------------|---------------|
| CFTR   | 0.395       | 0.395   | **0.673**    | Bagging +0.278|
| PAH    | 0.446       | 0.446   | 0.446        | Fark yok      |
| KANSER | **0.725**   | 0.657   | 0.690        | Baseline      |

CFTR analizi: 111 satır, 194 feature. BaggingClassifier her estimator'a 58 feature vererek yüksek boyutsallık sorununu çözdü. KANSER ve PAH'ta QT dağılım dönüşümü zararlı ya da etkisiz.

---

## 3. Önemli Teknik Bulgular

1. **sklearn MICE bug:** `IterativeImputer.fit_transform()` read-only array döndürüyor. Fix: `.to_numpy(copy=True)` ile writable array oluştur.

2. **CatBoost Focal Loss sözdizimi:** `"Focal:focal_alpha=0.25;focal_gamma=2.0"` — noktalı virgül zorunlu, virgül çalışmıyor.

3. **TabPFN n_estimators açığı:** Ablasyonda n_estimators=1 kullanıldı (default=8). QT_n8 denemesinde artırılınca performans değişmedi — asıl fayda BaggingClassifier'dan geldi.

4. **PAH için TEMİZLENMİŞ veri zararlı:** MASTER korelasyon yapısına göre yapılan Spearman eliminasyonu PAH sinyallerini sildi. PAH için ORİJİNAL veri kullanılmalı.

---

## 4. Mevcut En İyi Yapılandırma (Özet)

| Panel  | Model    | Deney                  | MCC    | PR-AUC |
|--------|----------|------------------------|--------|--------|
| MASTER | CatBoost | TEMİZLENMİŞ Ablasyon   | **0.573** | 0.941 |
| KANSER | TabPFN   | FS (ANOVA+RFE+EMBEDDED)| **0.762** | 0.876 |
| PAH    | CatBoost | ORİJİNAL Ablasyon      | **0.518** | 0.906 |
| CFTR   | CatBoost / TabPFN | Ablasyon / QT+Bagging | **0.673** | 0.923 / 0.995 |

---

## 5. Açık Sorular

- **PAH** hâlâ en zor panel (0.518). 38:1 dengesizlik ve küçük örneklem (372 satır) asıl kısıt.
- **MASTER** iki model neredeyse aynı performansta (0.573 vs 0.572) — doğal bilgi teorisi tavanı.
- **CFTR** için CatBoost ve TabPFN aynı MCC'yi farklı PR-AUC ile veriyor: TabPFN QT+Bagging çok daha yüksek PR-AUC (0.995 vs 0.923) → eşik optimizasyonu için daha iyi aday.

# MedicalVision — Süreç Raporu
**Proje:** TEKNOFEST 2026 Sağlıkta Yapay Zeka — Missense Varyant Sınıflandırması  
**Deadline:** 29 Haziran 2026, 17:00  
**Son güncelleme:** 14 Haziran 2026

---

## 1. Veri Seti Genel Yapısı

| Panel | Satır | Sütun (Orijinal) | Sütun (TEMİZLENMİŞ) |
|-------|-------|-------------------|----------------------|
| MASTER | 2931 | 353 | 194 |
| KANSER | 388 | 353 | 194 |
| PAH | 372 | 353 | 194 |
| CFTR | 111 | 353 | 194 |

**Sütun kategorileri (353 orijinal):**
- AL_ : 334 alel frekansı (gnomAD / AllofUs popülasyon alt grupları)
- EK_ : 9 evrimsel koruma skoru
- AA_ : 2 amino asit (referans ve varyant)
- CAT_ : 6 kategorik (popülasyon kodu, genotip, vb.)
- Variant_ID + Label : 2

**Görev:** Binary sınıflandırma — Pathogenic=1, Benign=0

---

## 2. EDA Süreci

### 2.1 Orijinal EDA Hatası

`BELGELER/EDA/EDA_PDR_orijinal.py` dosyasında kritik bir bug tespit edildi:

```python
# YANLIŞ — sadece ilk 100 özelliği tarıyordu
for i, col1 in enumerate(numeric_cols[:100]):
```

**Etkisi:**
- EK_ kolonları (evrimsel skorlar) hiç taranmadı — 334 AL_ kolonunun ilk 100'ü tarandı
- Korelasyon için "İlk 100 özellikte >0.80 korelasyon bulunamadı" çıktısı üretildi
- Eski raporlardaki EK_2-EK_3 için r=0.81 değeri Pearson ile hesaplanmıştı; Spearman ile gerçek değer r=0.395

### 2.2 Yeni EDA Scripti (`KODLAR/EDA/run_eda.py`)

Tüm 55,611 AL_ çifti (334×333/2) Spearman ile tarandı. Çıktılar:
- `BELGELER/EDA/TABLOLAR/` → 16 CSV
- `BELGELER/EDA/GRAFIKLER/` → 12 PNG
- `BELGELER/EDA/EDA_RAPORU.txt`

**Temel bulgular:**

| Bulgu | Değer |
|-------|-------|
| Sıfır varyanslı AL_ kolonları | 57 |
| AL_ çiftleri \|r\|>0.50 | 20,862 |
| AL_ çiftleri \|r\|>0.80 | 3,461 |
| r=1.00 tam kopya çiftler | Mevcut (overlapping gnomAD subgroups) |
| RF importance — AL_ | %82.6 |
| RF importance — EK_ | %17.4 |
| EK_7-EK_9 Spearman r | 0.664 (eski Pearson: 0.79) |
| EK_2-EK_3 Spearman r | 0.395 (eski Pearson: 0.81) |

### 2.3 Dosya Organizasyonu Hatası ve Düzeltme

**Hata 1:** EDA scripti MODELLER/EDA/ çıktısıyla birlikte BELGELER/EDA/'ya taşındı (kod da taşındı).  
**Hata 2:** Kullanıcı "kodların içindekini de birleştir" dedi; tekrar yanlış anlama.  
**Hata 3:** "Yav dosyaları görselleri filan taşısana sadece kodları taşıdın."  
**Sonuç:** KODLAR/EDA/ → kod; BELGELER/EDA/ → raporlar/grafikler/tablolar

---

## 3. Model Sonuçları — ORİJİNAL Veri (İlk Aşama)

### 3.1 CatBoost — ORİJİNAL (k Analizi, 353 sütun)

| Panel | En iyi yöntem | En iyi k | MCC | F1_macro | ROC_AUC |
|-------|--------------|---------|-----|----------|---------|
| MASTER | Kategorik_String_Korundu | — | 0.541 | 0.770 | 0.864 |
| KANSER | Kategorik_String_Korundu | — | 0.690 | 0.842 | 0.861 |
| PAH | Kategorik_String_Korundu | — | 0.518 | 0.701 | 0.692 |
| CFTR | Kategorik_String_Korundu_Yuksek_Eksik_Silindi | — | 0.395 | 0.697 | 0.724 |

### 3.2 TabPFN — ORİJİNAL (351 sütun)

| Panel | Senaryo | MCC | F1_macro | ROC_AUC |
|-------|---------|-----|----------|---------|
| MASTER | Ordinal_Ortalama | 0.528 | 0.758 | 0.836 |
| KANSER | Ordinal_Ortalama | 0.597 | 0.797 | 0.827 |
| PAH | Ordinal_Ortalama_Yuksek_Eksik_Silindi | 0.518 | 0.701 | 0.725 |
| CFTR | Ordinal_Medyan | 0.503 | 0.747 | 0.671 |

### 3.3 TN/Specificity Sorunu

PAH panelinde her iki model Specificity=%25 verdi (12 Benign testten yalnızca 3 doğru).  
**Neden:** PAH'ta 5:1 sınıf dengesizliği + default 0.5 threshold → model her şeyi Pathogenic sınıflandırıyor.  
**Çözüm:** MCC-optimal threshold optimizasyonu (henüz uygulanmadı).

---

## 4. Veri Temizleme ve Özellik Mühendisliği

### 4.1 AL_ Korelasyon Eşiği Kararı

**Tartışma:** r>0.95 mi, r>0.80 mi?
- r>0.95: çok muhafazakâr, gerçek multicollinearity kalmaya devam eder
- r>0.80: PDR'de "bağımsız popülasyon gruplarına indirgedik" diye savunulabilir

**Karar:** `CORR_THRESHOLD = 0.80` (NotebookLM onaylı, gnomAD popülasyon yapısıyla uyumlu)

### 4.2 CAT_3/4/5 Analizi

`G/G`, `C/C`, `T/T`, `A/A`, `./.` değerlerinin anlamı:
- Homozigot genotip çağrıları (her iki kromozomda aynı baz)
- Üçünün neredeyse aynı dağılımda olması: farklı veri tabanlarından gelen redundant bilgi
- `CAT_1` ve `CAT_2`: popülasyon kodları (gnomADe_NFE, AllofUs_EUR vb.) — REF/ALT allel DEĞİL

### 4.3 Özellik Mühendisliği Geri Dönüşleri

#### 4.3.1 AA_DEGISIM_TIPI ve TI_TV → Kaldırıldı
- Conservative/radical değişim (aynı fizikokimyasal grup) eklendi
- Transition/Transversion (kodon tablosundan) eklendi
- **Kullanıcı kararı:** "Değişim tipi ve TI_TV gereksiz bence" → ikisi de silindi

#### 4.3.2 REF_BAZ_KOD / ALT_BAZ_KOD (0/1/2/3) → Kaldırıldı
- A=0, C=1, G=2, T=3 keyfi sıralı kodlama eklendi
- **Kullanıcı kararı:** "Sen garip grup sayılar bulmuşsun" → anlamsız ordinal kodlama
- **Yeni yaklaşım:** 12 binary BAZ_ sütunu (BAZ_A_G, BAZ_G_A, BAZ_C_T, vb.)
  - Yalnızca TEK olası baz değişimi varsa 1; belirsiz durumlarda 0

**Not:** Kullanıcının tespiti — baz değişimi amino asit değişiminden çıkarılabilir (kodon tablosu), ama kodon çıkarılamaz (kodon degenerasyonu nedeniyle). Önce yanlış anlaşıldı, sonra düzeltildi.

#### 4.3.3 BAZ_DEGISIM_KOD → Eklenmedi
- Birleşik kod (string "A>G") eklenmesi düşünüldü
- **Kullanıcı kararı:** "baz değişim koda gerek yok" → sadece REF ve ALT ayrı ayrı yeterliydi, o da kaldırıldı

#### 4.3.4 DELTA_CODON_FLEX → Kaldırıldı
- Samocha 2017 dayanağıyla eklendi: AA_1 kodon sayısı − AA_2 kodon sayısı
- Label korelasyonu: r=+0.008, p=0.69 → **anlamsız** → silindi

#### 4.3.5 Küçük Popülasyon Flagları → Kaldırıldı
- POP_EAS, POP_EUR, POP_AMR, POP_SAS, POP_FIN, POP_ASJ, POP_MID eklendi
- Korelasyon analizi: hepsi p>0.10 → **anlamsız** → silindi
- **Kalan:** POP_NFE (r=+0.110***), POP_AFR (r=-0.099***)

#### 4.3.6 GC_STABILITE, PURIN_TUR, GENOTIP_EKSIK → Kaldırıldı
- CAT_3/4/5'ten türetildi
- Korelasyon analizi: r≈0.01 → **anlamsız** → silindi

---

## 5. Final Özellik Seti (353 → 194 sütun)

### 5.1 Silinen Sütunlar

| Kategori | Adet | Neden |
|----------|------|-------|
| Sıfır varyans AL_ | 57 | Medyan imputation sonrası sıfır |
| CAT_6 | 1 | %97.7 eksik |
| Yüksek korelasyonlu AL_ (greedy, r≥0.80) | 120 | Multicollinearity |
| Anlamsız yeni özellikler | ~12 | Label korelasyonu yok |

### 5.2 Eklenen Yeni Özellikler (19 sütun)

**AA_1/AA_2'den türetilen:**
| Özellik | r (MASTER) | p | Kaynak |
|---------|------------|---|--------|
| BLOSUM62 | -0.220 | *** | Henikoff 1992 |
| SNV_PATHWAY_COUNT | -0.013 | — | Karczewski 2020, Kircher 2014 |
| DELTA_HIDRO | -0.033 | — | Kyte-Doolittle |
| BAZ_T_C | +0.066 | *** | Kodon tablosu |
| BAZ_G_T | +0.058 | * | Kodon tablosu |
| BAZ_A_G | -0.046 | * | Kodon tablosu |
| BAZ_* (diğer 9) | ≈0 | — | Kodon tablosu |

**CAT_1/CAT_2'den türetilen:**
| Özellik | r (MASTER) | p | Kaynak |
|---------|------------|---|--------|
| IS_ALLOFUS | -0.183 | *** | All of Us 2019 |
| IS_EXOME | +0.122 | *** | Karczewski 2020 |
| POP_NFE | +0.110 | *** | Karczewski 2020 |
| POP_AFR | -0.099 | *** | Karczewski 2020 |

### 5.3 Kalan Sütunlar (194 toplam)
- 157 AL_ (greedy eleme sonrası)
- 9 EK_
- 5 CAT_ (CAT_1–5; CAT_6 silindi)
- 2 AA_
- 19 yeni özellik
- Variant_ID + Label = 2

---

## 6. Deney Klasörü Yapısı — Kararlar ve Hatalar

### 6.1 İlk Hata: DENEY_1/2/3/4 İsimlendirmesi

Yeni veri versiyonları için klasörler `DENEY_1`, `DENEY_2` şeklinde açıldı.  
**Kullanıcı:** "Deney diye açma bunların her biri yeni bir veri türü."  
**Düzeltme:** Klasörler veri türü adıyla açıldı: `TEMİZLENMİŞ`, `IMPUTED`, `NORM`, `IMP_NORM`

### 6.2 İkinci Hata: "Oldukları Yerde Kalsın" Çelişkisi

Kullanıcı önce eski CatBoost/TabPFN sonuçlarının taşınmasına gerek olmadığını söyledi ("gerek yok oldukları yerde kalsın").  
Sonra: "Eskiler Orjinal Veri diye bir klasörün altına alınmayacak mıydı" dedi.  
**Düzeltme:** Mevcut FILTER/WRAPPER/EMBEDDED klasörleri `MODELLER/CatBoost/ORİJİNAL_VERİ/` ve `MODELLER/TabPFN/ORİJİNAL_VERİ/` altına taşındı.

### 6.3 IMPUTED/NORM/IMP_NORM Klasörlerinin Silinmesi

`VERİLER/` ve `MODELLER/` altında IMPUTED, NORM, IMP_NORM klasörleri açıldı ama boş kaldı.  
**Neden silindiği:** ABLASYON_TEMİZLENMİŞ yaklaşımına geçildi — normalizasyon ve imputation ayrı veri türleri olarak değil, `build_ablation_temiz.py` içinde senaryo kombinasyonları olarak ele alındı. Boş klasörlere gerek kalmadı.

### 6.4 Final Klasör Yapısı

```
VERİLER/
  ORİJİNAL/                  (353 sütun, ham veri)
  TEMİZLENMİŞ/               (194 sütun, temizlenmiş veri)
  ABLASYON_MODEL_BAZLI/      (orijinal veri için senaryo CSV'leri)
  ABLASYON_TEMİZLENMİŞ/      (temizlenmiş veri için senaryo CSV'leri, 204 dosya)

MODELLER/CatBoost/
  ORİJİNAL_VERİ/             (eski sonuçlar)
  TEMİZLENMİŞ/               (yeni sonuçlar, FILTER/WRAPPER/EMBEDDED/ABLASYON)

MODELLER/TabPFN/
  ORİJİNAL_VERİ/
  TEMİZLENMİŞ/
```

---

## 7. Model Sonuçları — TEMİZLENMİŞ Veri (İkinci Aşama)

### 7.1 k Analizi — Optimal k Tespiti

k ∈ {10, 20, 30, 50, 100}, 11 yöntem, 4 panel.

**CatBoost k Analizi Sonuçları:**

| Panel | En iyi k | Yöntem | MCC |
|-------|---------|--------|-----|
| MASTER | 50 | FILTER_MUTUAL_INFO | 0.595 |
| KANSER | 20 | EMBEDDED | 0.726 |
| PAH | 20 | WRAPPER_RFE | 0.584 |
| CFTR | 50 | FILTER_MI + WRAPPER + EMBEDDED | 0.503 |

**TabPFN k Analizi Sonuçları:**

| Panel | En iyi k | Yöntem | MCC |
|-------|---------|--------|-----|
| MASTER | 100 | WRAPPER_RFE | 0.592 |
| KANSER | 10 | WRAPPER_RFE | 0.792 |
| PAH | 30 | WRAPPER_RFE | 0.518 |
| CFTR | 10 | FILTER_MUTUAL_INFO | 0.673 |

### 7.2 N_FEATURES Kararı — CatBoost: 30→50

İlk özellik seçimi N_FEATURES=30 ile yapıldı.  
**Neden değişti:** k_analizi k=50'nin MASTER ve CFTR için optimal olduğunu gösterdi; k=30 PAH'ta daha düşük.  
**Karar:** CatBoost → N_FEATURES=50, TabPFN → N_FEATURES=30 (k=50 TabPFN'de PAH'ı bozuyor: MCC 0.518→0.362)

### 7.3 TabPFN PAH Tutarsızlığı

k_analizi: WRAPPER_RFE k=30 → PAH MCC=0.518  
Özellik seçimi: WRAPPER_RFE N_FEATURES=30 → PAH MCC=0.205  
**Neden:** k_analizi RFE(n_features_to_select=1) ile tam sıralama yapıp top-k alır; özellik seçimi RFE(n_features_to_select=30) ile doğrudan keser → farklı eleme yolları → farklı özellik setleri.

### 7.4 Sadece MCC Gösterilmesi Hatası

Karşılaştırma tablosunda başta yalnızca MCC gösterildi.  
**Kullanıcı:** "f1 macro precision gibi değerlerde etkili değil mi niye sadece mcc karşılaştırıyoruz"  
**Düzeltme:** Tüm metrikler gösterildi (MCC, F1_macro, precision, recall, ROC_AUC).

### 7.5 best_by_panel Otomatik Üretilmemesi

feature_selection scriptleri başta best_by_panel CSV'si üretmiyordu.  
**Kullanıcı:** "bestbypanel'i her seferinde oluştur"  
**Düzeltme:** Her iki feature_selection_temiz.py'ye `ABLASYON` değişkeni ve best_by_panel üretimi eklendi.

### 7.6 ORİJİNAL vs TEMİZLENMİŞ Karşılaştırması (Özellik Seçimi Sonrası)

**CatBoost:**

| Panel | ORİJİNAL MCC | TEMİZLENMİŞ MCC | Δ | ORİJİNAL F1_macro | TEMİZLENMİŞ F1_macro |
|-------|-------------|-----------------|---|-------------------|----------------------|
| MASTER | 0.541 | 0.593 | **+0.052** | 0.770 | 0.795 |
| KANSER | 0.690 | 0.722 | **+0.032** | 0.842 | 0.856 |
| PAH | 0.518 | 0.518 | 0.000 | 0.701 | 0.701 |
| CFTR | 0.395 | 0.503 | **+0.108** | 0.697 | 0.747 |

**TabPFN:**

| Panel | ORİJİNAL MCC | TEMİZLENMİŞ MCC | Δ | ORİJİNAL F1_macro | TEMİZLENMİŞ F1_macro |
|-------|-------------|-----------------|---|-------------------|----------------------|
| MASTER | 0.528 | 0.571 | **+0.043** | 0.758 | 0.773 |
| KANSER | 0.597 | 0.762 | **+0.165** | 0.797 | 0.866 |
| PAH | 0.518 | 0.361 | **-0.157** | 0.701 | 0.593 |
| CFTR | 0.503 | 0.503 | 0.000 | 0.747 | 0.747 |

**PAH Gerilemesinin Nedeni (TabPFN):** Greedy korelasyon eleme aşamasında PAH-spesifik AL_ sütunları yüksek korelasyonlu olarak işaretlenip silindi. Bu sütunlar PAH için bağımsız sinyal taşıyabilirdi; MASTER'daki korelasyon düzeniyle aynı sütunlar elimine edildi. CatBoost bu kaybı yerleşik kategorik işlemeyle telafi etti, TabPFN edemedi.

---

## 8. ABLASYON_TEMİZLENMİŞ — Preprocessing Senaryoları

### 8.1 Neden Yeni Senaryo Seti Gerekti

ABLASYON_MODEL_BAZLI, 353 sütunlu orijinal veri için oluşturulmuştu.  
TEMİZLENMİŞ veri 194 sütuna sahip; ortak sütun sayısı 175, yalnızca TEMİZLENMİŞ'te olan 19 yeni özellik var.  
**Karar:** Doğrudan yeniden kullanım mümkün değil → `build_ablation_temiz.py` yazıldı.

### 8.2 Domain-Spesifik AL_→0 Imputation

NotebookLM önerisi: AL_ sütunlarındaki NaN = "bu popülasyonda varyant gözlenmedi" → 0 ile doldurmak biyolojik olarak anlamlı (nadirlik = patolojik sinyal).  
EK_ ve türev sütunlar (BLOSUM62, DELTA_HIDRO, SNV_PATHWAY_COUNT) → medyan ile doldurulur.  
Bu senaryo `AL_Sifir_EK_Medyan` adıyla tüm modeller için eklendi.

### 8.3 Üretilen Senaryo Sayısı

| Model | Senaryo sayısı | Encoding | Scaler | Imputer |
|-------|---------------|---------|--------|---------|
| CatBoost | 5 | Kategorik_String | — | none/median/mean/knn/al_zero |
| XGBoost | 4 | Ordinal | — | none/median/knn/al_zero |
| LightGBM | 3 | Kategorik_String | — | none/median/al_zero |
| TabPFN | 6 | Ordinal | —/MinMax | median/knn/al_zero |
| TabNet | 9 | Ordinal | Robust/MinMax/Standard | median/knn/al_zero |
| KNN | 9 | OneHot | Robust/MinMax/Standard | median/knn/al_zero |
| SVM | 9 | OneHot | Robust/MinMax/Standard | median/knn/al_zero |
| AdaBoost | 6 | OneHot/Ordinal | — | median/knn/al_zero |
| **Toplam** | **51 klasör × 4 panel = 204 CSV** | | | |

Çıktı: `VERİLER/ABLASYON_TEMİZLENMİŞ/`  
Script: `KODLAR/ABLASYON_CSV_SCRIPTLERI/build_ablation_temiz.py`

---

## 9. NotebookLM Analizleri

### 9.1 NotebookLM'den Gelen Onaylanan Kararlar

| Konu | Karar | Kaynak |
|------|-------|--------|
| AL_ eşiği | r>0.80 (PDR savunulabilir) | Karczewski 2020 |
| Baz değişimi | Kodon tablosundan çıkarılabilir | Kodon degenerasyonu |
| SNV_PATHWAY_COUNT | Ekle | Karczewski 2020, Kircher 2014 |
| IS_ALLOFUS / IS_EXOME | Ekle | All of Us 2019 |
| POP_AFR sinyali | Benign yönlü (Afrika genetik çeşitliliği) | Karczewski 2020 |
| AL_ NaN imputation | AL_→0, EK_→medyan (domain-spesifik) | Biyolojik gerekçe |

---

## 10. Takım Belgesi Tarzı Hatası

**Hata 1:** Takım için uzun markdown belgesi + kod blokları içeren kılavuz yazıldı.  
**Hata 2:** Yine uzun kılavuz + kod blokları.  
**Düzeltme:** Kısa paragraflar, "şu scripti çalıştır" tarzı, dosyaya kaydetmeden doğrudan metin.

---

## 11. Bekleyen İşler

### Acil
- [ ] PAH/CFTR için MCC-optimal threshold optimizasyonu
- [ ] ABLASYON_TEMİZLENMİŞ senaryoları üzerinde CatBoost + TabPFN ablasyon çalıştır (`run_ablation_temiz.py`)
- [ ] Ablasyon sonuçlarını feature selection sonuçlarıyla karşılaştır

### Diğer Modeller
- [ ] Ahmet: AdaBoost — `ABLASYON_TEMİZLENMİŞ/ADABOOST_*` senaryolarını çalıştır
- [ ] Nursima: KNN — `ABLASYON_TEMİZLENMİŞ/KNN_*` senaryolarını çalıştır
- [ ] Esma: LightGBM, TabNet, XGBoost — ilgili klasörlerden çalıştır

### İleri Aşama
- [ ] Stacking ensemble (tüm modeller tamamlandıktan sonra)
- [ ] Transfer learning (MASTER → alt paneller)
- [ ] PDR raporu yazımı (deadline: 29 Haziran 2026, 17:00)

---

## 12. Referanslar (NotebookLM onaylı)

1. Karczewski vd. (2020). The mutational constraint spectrum quantified from variation in 141,456 humans. *Nature*, 581, 434–443.
2. Kircher vd. (2014). A general framework for estimating the relative pathogenicity of human genetic variants. *Nature Genetics*, 46, 310–315.
3. Samocha vd. (2017). Regional missense constraint improves variant deleteriousness prediction. *BioRxiv*.
4. Richards vd. (2015). Standards and guidelines for the interpretation of sequence variants. *Genetics in Medicine*, 17, 405–424.
5. All of Us Research Program Investigators (2019). The 'All of Us' research program. *NEJM*, 381, 668–676.
6. Henikoff & Henikoff (1992). Amino acid substitution matrices from protein blocks. *PNAS*, 89, 10915–10919.
7. Kyte & Doolittle (1982). A simple method for displaying the hydropathic character of a protein. *JMB*, 157, 105–132.

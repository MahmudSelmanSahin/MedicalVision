# Metrik Tanımları

Bu belge, projede kullanılan tüm değerlendirme ölçülerinin tanımını, formülünü,
ne işe yaradığını ve dengesiz veride (≈%73–83 patojenik) nasıl davrandığını
özetler. İki grup var: **(A) Model performans metrikleri**, **(B) Özellik–label
ilişki ölçüleri**.

---

## 0. Temel — Konfüzyon Matrisi

İkili sınıflandırmada (pozitif = patojenik = 1, negatif = benign = 0):

|  | Tahmin: Patojenik | Tahmin: Benign |
|--|--|--|
| **Gerçek: Patojenik** | TP (doğru pozitif) | FN (yanlış negatif) |
| **Gerçek: Benign** | FP (yanlış pozitif) | TN (doğru negatif) |

Aşağıdaki çoğu metrik bu dört sayıdan türer.

---

## A. Model Performans Metrikleri

### Accuracy (Doğruluk)
```
 Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Ne ölçer:** Doğru tahmin oranı.
- **Aralık:** [0, 1].
- **Dengesizlikte:** ⚠️ **Yanıltıcı.** %80 patojenik veride "hep patojenik" diyen
  model %80 accuracy alır ama işe yaramaz. **Tek başına kullanılmaz.**

### Precision (Kesinlik)
```
 Precision = TP / (TP + FP)
```
- **Ne ölçer:** "Patojenik" dediklerimin ne kadarı gerçekten patojenik.
- Yüksek precision = az yanlış alarm.

### Recall / Sensitivity / Duyarlılık (TPR)
```
 Recall = TP / (TP + FN)
```
- **Ne ölçer:** Gerçek patojeniklerin ne kadarını yakaladım.
- Klinikte kritik: patojenik kaçırmamak (yüksek recall) genelde önemli.

### Specificity / Özgüllük (TNR)
```
 Specificity = TN / (TN + FP)
```
- **Ne ölçer:** Gerçek benignlerin ne kadarını doğru benign dedim.

### F1-score
```
 F1 = 2 · (Precision · Recall) / (Precision + Recall) = 2TP / (2TP + FP + FN)
```
- **Ne ölçer:** Precision ve Recall'un harmonik ortalaması (denge).
- **Not:** Standart F1 pozitif sınıfa odaklıdır → dengesizlikte çoğunluk lehine
  şişebilir.

### F1-macro
```
 F1_macro = ( F1(benign) + F1(patojenik) ) / 2
```
- **Ne ölçer:** Her sınıfın F1'i ayrı hesaplanıp ortalanır.
- **Dengesizlikte:** ✅ Daha adil — azınlık sınıfı (benign) eşit ağırlık alır.
  Projede F1'den daha bilgilendirici.

### MCC (Matthews Korelasyon Katsayısı)
```
              TP·TN − FP·FN
 MCC = ───────────────────────────────────────
        √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
- **Ne ölçer:** Tahmin ile gerçeğin korelasyonu; dört hücreyi birden kullanır.
- **Aralık:** [−1, +1] (+1 kusursuz, 0 rastgele, −1 tam ters).
- **Dengesizlikte:** ✅✅ **En güvenilir.** Yalnız iki sınıfta da iyiyse yükselir;
  azınlığı ihmal eden model 0'a yakın MCC alır. **Projenin birincil metriği.**

### ROC-AUC
```
 ROC-AUC = TPR–FPR eğrisinin altındaki alan
         = P(skor_pozitif > skor_negatif)   (sıralama olasılığı)
```
- **Ne ölçer:** Modelin patojeniği benign üstüne **sıralama** yeteneği. Eşikten
  bağımsız, **rank-based**.
- **Aralık:** [0, 1]; 0.5 = rastgele.
- **Dengesizlikte:** İyimser olabilir (precision'ı yansıtmaz). MCC ile birlikte
  okunmalı: ROC = sıralama, MCC = karar kalitesi.

### PR-AUC (Average Precision)
```
 PR-AUC = Precision–Recall eğrisinin altındaki alan
```
- **Ne ölçer:** Precision ve Recall dengesini eşik boyunca özetler.
- **Dengesizlikte:** ✅ Azınlık/pozitif sınıf önemliyse ROC-AUC'den daha
  bilgilendirici (baz çizgisi = pozitif oranı).

---

## B. Özellik–Label İlişki Ölçüleri

Bir özelliğin label ile ilişkisini ölçmek için (model değil, tek sütun).

### Rank-based (sıra-tabanlı) — temel fikir
Değer yerine **sıra numarası** kullanılır → uç değer büyüklüğüyle sürüklemez,
**outlier'a dayanıklı**. Spearman ve AUC bunun üstüne kuruludur.

### Pearson r (parametrik)
```
 r = cov(x, y) / (σ_x · σ_y)
```
- Doğrusal ilişki; **outlier ve çarpık dağılıma duyarlı.** Bu genomik veride
  (0/1 ağırlıklı) yanıltabilir → karar için kullanılmaz, referans.

### Spearman ρ (rank-based)
```
 ρ = Pearson( rank(x), rank(y) )

 (bağ yoksa)   ρ = 1 − 6·Σdᵢ² / (n(n²−1))     dᵢ = sıra farkı
```
- Pearson'ın sıralar üzerindeki hali; outlier'a dayanıklı. **Adil önem
  tablosunda "anlamlılık kapısı" için kullanılır.**

### Cohen's d (parametrik etki büyüklüğü)
```
 d = (M₁ − M₀) / s_pooled        (M₁,M₀ = sınıf ortalamaları)
```
- Sınıf oranından bağımsız ama **outlier/çarpıklığa duyarlı** (varyans şişerse
  yanıltır). Karar için kullanılmaz, referans.

### AUC (tek özellik) / Cliff's delta (rank-based etki büyüklüğü)
```
       Σ(pozitif sıraları) − n₊(n₊+1)/2
 AUC = ───────────────────────────────       Cliff's δ = 2·AUC − 1
                 n₊ · n₋
```
- Sıra-tabanlı ayrım gücü; oran ve outlier'dan etkilenmez. **Adil önem
  tablosunda "güç sıralaması" için kullanılır.**

---

## Özet — Hangi metrik ne zaman

| Metrik | Tür | Rank-based | Dengesizlikte | Projede rol |
|--------|-----|:---:|:---:|---|
| Accuracy | model | ❌ | ⚠️ yanıltıcı | kullanma (tek başına) |
| Precision / Recall | model | ❌ | bilgilendirici (çift) | yan metrik |
| F1 | model | ❌ | çoğunluk lehine | yan metrik |
| F1-macro | model | ❌ | ✅ adil | raporlanan |
| **MCC** | model | ❌ | ✅✅ en iyi | **birincil** |
| ROC-AUC | model | ✅ | iyimser olabilir | sıralama göstergesi |
| PR-AUC | model | kısmen | ✅ | yan metrik |
| Pearson r | özellik | ❌ | — | referans |
| Spearman ρ | özellik | ✅ | — | önem **kapısı** |
| Cohen's d | özellik | ❌ | — | referans |
| AUC / Cliff's δ | özellik | ✅ | — | önem **sıralaması** |

## Projede neden MCC birincil?

Veri ≈%73–83 patojenik (çok dengesiz). Accuracy bu durumda yanıltır ("hep
patojenik" → yüksek accuracy, MCC=0). MCC dört konfüzyon hücresini birden
kullandığı, dengesizlikten etkilenmediği ve azınlığı (benign) ihmal eden modeli
cezalandırdığı için **birincil metrik** seçildi; yanında F1-macro (adil ortalama)
ve ROC-AUC (sıralama) raporlanır.

**Örnek (100 örnek: 80 patojenik / 20 benign):**
```
 Model A "hep patojenik":  Accuracy=0.80 ama MCC=0   (çöp model yakalanır)
 Model B gerçek ayrım:     Accuracy=0.85,    MCC=0.58 (gerçek başarı görünür)
```
Accuracy A ile B'yi ayırt edemez; MCC net ayırır.

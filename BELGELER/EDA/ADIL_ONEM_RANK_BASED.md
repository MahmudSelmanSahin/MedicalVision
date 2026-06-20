# Adil Özellik Önemi — Rank-Based (Sıra-Tabanlı) Yöntem

Bu belge, özellik–label ilişkisini **sınıf dengesizliğinden ve aykırı
değerlerden etkilenmeden** ölçen "adil önem" tablosunun mantığını, hesaplanışını
ve kullanımını açıklar.

Çıktı tabloları: `BELGELER/EDA/TABLOLAR/adil_onem_{MASTER,KANSER,PAH,CFTR}.csv`

---

## 1. Problem — neden ham korelasyon / Cohen's d yetmez

İki ayrı bozucu etki var:

1. **Sınıf dengesizliği:** Ham Pearson korelasyonu `√(p·q)` çarpanıyla ölçeklenir;
   dengesizlikte (örn. %80/%20) büyüklüğü **küçültür** → özelliği olduğundan zayıf
   gösterir.
2. **Aykırı değer / çarpık dağılım:** Bu genomik özelliklerin çoğu 0/1 ağırlıklı,
   medyanla doldurulmuş, küçük örneklemli. Pearson r ve Cohen's d **parametrik**
   olduğu için birkaç uç değer istatistiği şişirir/çarpıtır.

**Örnek (CFTR, AL_27):**
```
 benign(21):  0.0 ×15, 1.0 ×6      patojenik(90): ~hep 0.0, max 0.40
 Cohen's d = -1.39  →  "ÇOK güçlü" (YANLIŞ — küçük varyans d'yi şişirmiş)
 AUC       =  0.36  →  orta ayrım  (DOĞRU)
```
Cohen's d ile sıralarsan AL_27'yi en üste koyarsın (hatalı). Bu yüzden karar
mekanizmasını **sıra-tabanlı** ölçülere taşıyoruz.

## 2. "Rank-based" (sıra-tabanlı) nedir

Gerçek değer yerine **sıra numarası** kullanılır; aykırı değer sadece "en yüksek
sıra" olur, büyüklüğüyle istatistiği sürüklemez.
```
 Değerler:  [1, 2, 3, 1000]   →   Sıralar: [1, 2, 3, 4]
```
- **Spearman r** = Pearson'ın sıralar üzerinde hesaplanmış hali (outlier-dayanıklı).
- **AUC** = doğası gereği rank-based: "rastgele patojenik, rastgele benign'den
  yüksek sırada olma olasılığı". **Cliff's delta = 2·AUC − 1**.

## 3. İki ayrı iş — kapı + sıralama (karıştırma yok)

```
 SORU 1: "Bu ilişki GERÇEK mi, şans mı?"   → KAPI (anlamlılık)
 SORU 2: "Gerçekse NE KADAR güçlü?"         → SIRALAMA (etki büyüklüğü)
```

| Araç | Soru | Rol |
|------|------|-----|
| **Spearman şans tabanı** (permütasyon) | Gerçek mi? | KAPI (geç/kal) |
| **AUC / Cliff's delta** | Ne kadar güçlü? | SIRALAMA |

İkisi de gereklidir: taban tek başına gücü, AUC tek başına gerçekliği söylemez.

## 4. İşleyiş şeması

```
 HER ÖZELLİK
    │ Spearman r, AUC, Cliff's delta hesapla
    ▼
 [KAPI]  |Spearman r| > panel şans tabanı ?
    ├── HAYIR → GÜRÜLTÜ (ele, sıralama yapma)
    └── EVET  → gerçek
                 │
                 ▼
 [SIRALA]  ayrım_gücü = |AUC − 0.5| (büyükten küçüğe)
                 │
                 ▼
          Nihai: anlamlı + yüksek ayrım = güvenilir önemli özellik
```

## 5. Şans tabanı nasıl hesaplanır (Spearman permütasyonu)

```
 500'e yakın tekrar:
   1) Label'i KARIŞTIR (gerçek ilişki yok olur, n ve oran sabit)
   2) Her özellik için |Spearman(rank(x), y_karışık)|
   3) En yükseğini al  → null_max[i]   (çoklu-test düzeltmesi)
 ŞANS TABANI = null_max dağılımının %95'lik dilimi
```
Az örneklem → geniş şans dağılımı → yüksek taban. (CFTR n=111 → 0.35; MASTER
n=2931 → 0.07.) Dengesizliği eşitlemek tabanı düşürmez; downsample n'i küçültüp
yükseltir, upsample kopyalarla sahte düşürür — bu yüzden **orijinal veride
permütasyon** doğru yöntemdir.

## 6. Sonuçlar (panel başına)

| Panel | n | Spearman şans tabanı | Anlamlı özellik | En güçlü özellik (AUC) |
|-------|:--:|:--:|:--:|:--|
| MASTER | 2931 | 0.070 | 156 | EK_7 (0.71) |
| KANSER | 388 | 0.190 | 95 | EK_9 (0.78) |
| PAH | 372 | 0.192 | 2 | EK_7 (0.70) |
| CFTR | 111 | 0.348 | 4 | EK_9 (0.80) |

Her panelde **EK_7 / EK_9 / EK_2** (patojenite skorları) tepede ve hem anlamlı hem
yüksek ayrım gücüne sahip → gerçek, güvenilir sinyal. PAH/CFTR'de anlamlı özellik
çok az (2–4): küçük benign örneklemi nedeniyle gerçek sinyal kıt.

## 7. 2×2 karar matrisi

```
                │  GÜÇLÜ (yüksek AUC)        │  ZAYIF (düşük AUC)
 ───────────────┼───────────────────────────┼─────────────────────
  ANLAMLI ✅     │  ✅✅ EN DEĞERLİ           │  ✅ gerçek ama zayıf
                │  CFTR EK_9 (AUC 0.80)      │  (düşük öncelik)
 ───────────────┼───────────────────────────┼─────────────────────
  ANLAMSIZ ❌    │  ⚠️ GÜRÜLTÜ TUZAĞI         │  ❌ yok say
                │  CFTR AL_7 (AUC 0.275 ama  │
                │  Spearman 0.31 < taban     │
                │  0.348 → güvenme)          │
```

## 8. Tablo sütunları

| Sütun | Anlamı | Rol |
|-------|--------|-----|
| `spearman_r` | Sıra-tabanlı korelasyon (işaret = yön) | KAPI |
| `auc` | Sıra-tabanlı ayrım (0.5 = yok) | SIRALAMA |
| `cliff_delta` | 2·AUC − 1 | SIRALAMA (eş) |
| `ayrim_gucu` | \|AUC − 0.5\| | sıralama anahtarı |
| `yon` | patojenik / benign | yorum |
| `anlamli` | \|spearman_r\| > şans tabanı | KAPI sonucu |

> **Not:** Cohen's d ve Pearson r bilinçli olarak **dışarıda bırakıldı** —
> parametrik oldukları ve bu çarpık/aykırı-değerli genomik veride yanılttıkları
> için (bkz. AL_27). Tüm karar zinciri rank-based ölçülerden geçer.

## 9. Nasıl kullanılır

1. **FILTER özellik seçimi:** `anlamli = EVET` olanları al, `ayrim_gucu`'ne göre
   sırala → her panel için adil, gürültüden arınmış özellik seti.
2. **Rapor/klinik yorum:** Bir özelliği "patojeniteyle ilişkili" diye sunmadan
   önce `anlamli = EVET` mi diye bak (sahte bulgu kalkanı).
3. **Küçük panel stratejisi:** PAH/CFTR'de anlamlı özellik çok az → özellik
   kararlarını MASTER üzerinden ver, panele uygula (hibrit yaklaşım).

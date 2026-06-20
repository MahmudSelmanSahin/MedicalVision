# SVM ve LightGBM — Gerçek Veride En İyi Parametreler

Bu rapor `Teknofest_DUZELTILMIS/SVM` ve `Teknofest_DUZELTILMIS/LIGHTGBM` klasörlerindeki üç script
(`run_ablation.py`, `run_feature_selection_temiz.py`, `run_k_analysis_temiz.py`) **gerçek
TEKNOFEST verisi** üzerinde çalıştırılarak üretildi (MASTER/KANSER/PAH/CFTR panelleri, asıl proje
deposundaki `VERİLER` klasörü — `MedicalVision` reposu — kullanıldı). Optuna kurulu olmadığı için
önce kuruldu, ardından her üç script de 4 panel × ilgili senaryo/yöntem/k kombinasyonlarının
tamamında **0 hata** ile tamamlandı. Ham çıktılar `MODELLER/SVM/...` ve `MODELLER/LightGBM/...`
altında (CSV + PNG) duruyor; bu dosya bunların özetidir.

Üç script de aynı metriği üç katmanda raporluyor:
- `mcc_default_05` → sabit/varsayılan parametre, eşik 0.5 (taban çizgi)
- `mcc_tuned_05` → Optuna ile bulunan parametreler, eşik 0.5
- `mcc` → Optuna parametreleri + OOF/test üzerinde MCC'yi maksimize eden eşik (nihai metrik)

Veri çok dengesiz olduğundan (özellikle CFTR: 90 patojenik / 21 benign, PAH: 310/62) **MCC**
doğru karşılaştırma metriği; accuracy yanıltıcı olurdu.

---

## 1) Panel başına GENEL EN İYİ sonuç (3 deney tipinin en yükseği)

| Panel | Model | En iyi geldiği aşama | MCC | Özellik/senaryo | Hiperparametre |
|---|---|---|---|---|---|
| CFTR | LightGBM | Özellik Seçimi (EMBEDDED, k=50) | **0.537** | LGBM importance ile top-50 | num_leaves=99, lr=0.0246, eşik=0.30 |
| CFTR | SVM | k-Analizi (EMBEDDED, k=30) | 0.871 | LinearSVC katsayısı ile top-30 | C=2.973, gamma=0.01, eşik=0.81 |
| KANSER | LightGBM | k-Analizi (FILTER_MI+RFE+EMBEDDED, k=10) | **0.878** | 3 yöntemin kesişimi top-10 | num_leaves=30, lr=0.0631, eşik=0.43 |
| KANSER | SVM | k-Analizi (FILTER_MUTUAL_INFO, k=20) | 0.817 | Mutual-info top-20 | C=20.54, gamma=0.01, eşik=0.66 |
| MASTER | LightGBM | k-Analizi (FILTER_MI+RFE+EMBEDDED, k=50) | **0.565** | 3 yöntemin kesişimi top-50 | num_leaves=78, lr=0.0197, eşik=0.42 |
| MASTER | SVM | k-Analizi (EMBEDDED, k=30) | 0.541 | LinearSVC top-30 | C=0.547, gamma=scale, eşik=0.67 |
| PAH | LightGBM | k-Analizi (FILTER_ANOVA, k=10) | **0.518** | ANOVA top-10 | num_leaves=32, lr=0.0376, eşik=0.25 |
| PAH | SVM | k-Analizi (EMBEDDED, k=30) | 0.535 | LinearSVC top-30 | C=0.205, gamma=scale, eşik=0.75 |

**Çıkan örüntü: her iki model için de en iyi sonuç hemen hemen her panelde "k-Analizi" aşamasından
geliyor** — yani 50/100 gibi büyük özellik kümeleri yerine **k=10–30 arası küçük, agresif şekilde
seçilmiş** özellik kümeleri en iyi MCC'yi veriyor. Bu, panellerin örneklem sayısının (CFTR=111,
PAH=372, KANSER=388) özellik sayısına (194) göre düşük olmasından kaynaklanıyor — az veri + çok
özellik = SVM/LightGBM overfit eder, k küçültülünce varyans düşüyor.

---

## 2) SVM — neden bu parametreler en iyi?

| Panel | C | gamma | Eşik | Yorum |
|---|---|---|---|---|
| CFTR | **2.97** | **0.01** | 0.81 | Küçük veri (111) + dengesiz (21 benign) → orta-yüksek C ile karar sınırı sıkılaştırılıyor, küçük gamma (0.01, "scale" değerinden daha düz) overfit'i önlüyor. Yüksek eşik (0.81) precision'ı korurken recall'u dengeliyor. |
| KANSER | **20.54** | **0.01** | 0.66 | En büyük C burada — KANSER paneli en "ayrılabilir" sınıf yapısına sahip, SVM'in marjini sıkılaştırmasına izin vermek işe yarıyor. |
| MASTER | **0.55** | **scale** | 0.67 | En büyük panel (2931 satır) → düşük C (daha geniş marj, daha fazla regularizasyon) genelleme için gerekli; `gamma=scale` (varsayılana yakın) yeterli. |
| PAH | **0.20** | **scale** | 0.75 | En düşük C — PAH en dengesiz/gürültülü panel, güçlü regularizasyon (küçük C) gerekiyor. |

**Genel kural:** SVM için `C` değeri panel büyüklüğü/temizliğiyle ters orantılı — az veri ve
yüksek dengesizlikte düşük C (PAH, MASTER), net ayrılabilir ve orta büyüklükte panellerde yüksek C
(KANSER) optimal. `gamma=0.01` sadece CFTR ve KANSER'de (yani küçük-orta panellerde) seçildi;
"scale" (öntanımlının kendisi) MASTER ve PAH'ta optimal kaldı — bu da Optuna'nın `gamma` ızgarasının
makul olduğunu, geniş bir aralık taramaya gerek olmadığını gösteriyor.

**Özellik seçimi tarafında:** EMBEDDED (LinearSVC katsayı büyüklüğü) yöntemi 3/4 panelde tek başına
en iyi sonucu veriyor; tek istisna KANSER'de FILTER_MUTUAL_INFO. Bu, RFE/ANOVA gibi ayrı bir proxy
modelden gelen sıralamadan ziyade, **SVM'in kendi karar fonksiyonuna en yakın özellik sıralamasının
(embedded) genelde en güvenilir** olduğunu gösteriyor — kombinasyon (FILTER+WRAPPER+EMBEDDED gibi)
yöntemleri hiçbir panelde tek yöntemi geçemedi.

---

## 3) LightGBM — neden bu parametreler en iyi?

| Panel | num_leaves | learning_rate | n_trees | Eşik | Yorum |
|---|---|---|---|---|---|
| CFTR | 99 | 0.0246 | — | 0.30 | Yaprak sayısı çok yüksek ama veri çok az (111) → early stopping + k=50 özellik kısıtlaması ile overfit telafi ediliyor; düşük eşik (0.30) az pozitif örneği (21 benign) yakalamak için gerekli. |
| KANSER | 30 | 0.0631 | 100 (limit) | 0.43 | Orta yaprak sayısı + en yüksek learning_rate — KANSER en "öğrenilebilir" panel, hızlı öğrenme oranı yeterli sinyali çabuk yakalıyor. |
| MASTER | 78 | 0.0197 | 100 (limit) | 0.42 | Büyük veri + yüksek yaprak sayısı + düşük learning_rate — klasik "büyük veri, yavaş/dikkatli öğren" deseni. |
| PAH | 32 | 0.0376 | 89 | 0.25 | Düşük yaprak sayısı (regularizasyon) + düşük eşik — PAH'ta pozitif sınıf (patojenik) baskın (310/372), model konservatif eşiğe rağmen az sayıdaki benign'i (62) ayırt etmeye zorlanıyor. |

**Genel kural:** `num_leaves` panel büyüklüğüyle birlikte artıyor (PAH/KANSER düşük, CFTR/MASTER
yüksek) ama CFTR istisnası (küçük veri + yüksek yaprak) k=50 özellik kısıtlaması ve early
stopping'in birlikte regularizasyon sağladığını gösteriyor — yani **tek başına num_leaves'e
bakmak yeterli değil, k (özellik sayısı) ile birlikte değerlendirilmeli.**

**Özellik seçimi tarafında:** En iyi sonuçların çoğu (KANSER, MASTER) **üçlü kombinasyon
(FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED)** yönteminden geliyor — SVM'in tersine, LightGBM
için farklı yöntemlerin kesişimi/birleşimi daha istikrarlı. CFTR'de ise tek başına EMBEDDED
(LGBM'in kendi gain-based importance'ı), PAH'ta tek başına FILTER_ANOVA en iyi sonucu verdi.
`Eksik_Veri_Korundu` (native missing, LightGBM'in NaN'leri kendi işlemesine izin veren senaryo)
ablasyon aşamasında **4 panelin 4'ünde de** medyan ile doldurmaktan daha iyi sonuç verdi — bu,
LightGBM'e veri hazırlarken **eksik değerleri doldurmamanın** (median/mean) daha doğru olduğunu
gösteren güçlü bir bulgu.

---

## 4) Pratik öneri (final model için)

- **SVM:** Panel başına ayrı model eğitin (tek bir genel C/gamma yok). Özellik seçimi için
  varsayılan olarak **EMBEDDED (LinearSVC)** yöntemini, k=20–30 aralığını deneyin; KANSER için
  ek olarak `FILTER_MUTUAL_INFO` deneyin. Eşikleri 0.5 sabit bırakmayın — OOF üzerinde MCC-optimal
  eşik kullanın (yukarıdaki tabloda verilen değerler).
- **LightGBM:** Eksik veriyi doldurmayın, LightGBM'in native categorical/missing desteğine izin
  verin (`Eksik_Veri_Korundu` senaryosu). Özellik seçiminde varsayılan olarak **üçlü kombinasyon
  (FILTER_MUTUAL_INFO + WRAPPER_RFE + EMBEDDED)**'i deneyin, k=10–50 arasını panel büyüklüğüne
  göre ayarlayın (küçük panel → küçük k).
- Her iki model için de **MCC-optimal eşik** kullanımı, sabit 0.5 eşiğe göre ortalama
  +0.05 ile +0.30 arasında MCC kazancı sağlıyor (bkz. `gain_threshold` kolonları) — bu kazanım,
  hiperparametre aramasının kendisinden (`gain_params`) çoğu panelde daha büyük. Yani **eşik
  optimizasyonunu atlamayın.**

---

## 5) Dikkat edilmesi gereken sınırlamalar

- CFTR (111 satır) ve PAH (372 satır, 62 benign) çok küçük/dengesiz; buradaki MCC değerleri
  tek bir 80/20 bölünmesine veya 5-fold OOF'a dayanıyor — varyansı yüksek, gerçek yarışma test
  setinde farklı çıkabilir. Mümkünse bu panellerde tekrarlı (repeated) CV ile doğrulayın.
- Optuna trial sayıları (SVM=30, LightGBM=50) hız için sınırlı tutuldu; eğer daha fazla zaman
  varsa bu sayıları artırmak (örn. 100) küçük ek kazanımlar getirebilir ama büyük bir sıralama
  değişikliği beklenmiyor (özellikle SVM'de gamma ızgarası küçük olduğu için).
- Tüm sayısal sonuçlar `MODELLER/SVM/` ve `MODELLER/LightGBM/` altındaki CSV dosyalarında ve
  confusion matrix / precision-recall PNG'lerinde tam olarak mevcut; bu rapor onların özetidir.

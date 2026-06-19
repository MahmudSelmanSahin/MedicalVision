# Özellik–Label Korelasyonunda Şans Tabanı (Panel-Bazlı Güven Eşiği)

Bu belge, özellik–label korelasyonlarının **hangi |r| değerinden sonra güvenilir**
olduğunu panel-bazlı belirler. Motivasyon: benign sınıfı azınlıkta ve panellerde
örneklem küçük olduğunda, korelasyonlar **şans eseri** yüksek çıkabilir.

## 1. Problem

İkili label ile korelasyon tek bir sayıdır; "benign ile korelasyon" ile "patojenik
ile korelasyon" aynı **|r|**'dir (yalnız işaret değişir). Azınlık olmak korelasyonu
bir sınıfa **kaydırmaz** (point-biserial `r = (M₁−M₀)/sₓ · √(p·q)`, `√(p·q)` simetrik).
**Ama** azınlık sınıfı küçükse korelasyon tahminleri **yüksek varyanslı / şişmiş**
olur — az sayıda benign örnek, gerçek bir ilişki olmasa bile yüksek |r| üretebilir.

## 2. Yöntem — Permütasyon Şans Tabanı

Her panel için, **gerçek örneklem boyutu ve sınıf dengesi** korunarak label 500 kez
karıştırılır (gerçek ilişki yok edilir) ve şu eşikler ölçülür:

- **Tek-özellik %95:** Tek bir özelliğin şans eseri aşabileceği |r| (gevşek eşik).
- **Çoklu-test %95 (güvenilir eşik):** ~280 özellik arasından **en yüksek** |r|'nin
  şans tabanı. Çoklu-test (multiple-testing) düzeltmeli — asıl güvenmen gereken eşik.
- **Anlamlı #:** Gerçek |r| > çoklu-test %95 olan özellik sayısı.

Kod: tek seferlik permütasyon analizi (orijinal veri, sayısal değişken özellikler).

## 3. Sonuç — Panel-Bazlı Şans Tabanı

| Panel | n | Benign | Gerçek max\|r\| | Tek-özellik %95 | **Çoklu-test %95** | Çoklu-test %99 | Anlamlı # |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MASTER | 2931 | 782 | 0.331 | 0.036 | **0.069** | 0.076 | **152** |
| KANSER | 388 | 120 | 0.488 | 0.095 | **0.180** | 0.203 | **98** |
| PAH | 372 | 62 | 0.267 | 0.103 | **0.204** | 0.228 | **2** |
| CFTR | 111 | 21 | 0.482 | 0.184 | **0.379** | 0.412 | **5** |

> **"Çoklu-test %95" = pratik güven eşiği.** Bir özelliğin |r|'si bu sayının
> üstündeyse gerçek; altındaysa şans eseri çıkmış olabilir.

## 4. Yorum

Benign sayısı azaldıkça şans tabanı **fırlıyor**:

- **MASTER (782 benign):** eşik **0.069** → 0.07 üstü güvenilir, 152 anlamlı özellik.
- **KANSER (120 benign):** eşik **0.180** → 98 anlamlı özellik.
- **PAH (62 benign):** eşik **0.204**, gerçek max yalnız 0.267 → **sadece 2 özellik**
  şansı geçiyor; korelasyonların neredeyse tamamı gürültü. (PAH'ın neden zor
  modellendiğini de açıklıyor.)
- **CFTR (21 benign):** eşik **0.379** → güvenmek için |r| > 0.38 gerekiyor; sadece
  5 özellik geçiyor. (Örn. AL_27=−0.48, AL_33=−0.45 eşiğin üstünde → gerçek.)

## 5. Pratik Kurallar

1. **Korelasyonu yalnız kendi panelinin eşiğiyle kıyasla:**
   MASTER > 0.07 · KANSER > 0.18 · PAH > 0.20 · CFTR > 0.38.
2. **CFTR/PAH'ta tekil korelasyona dayanıp rapor yazma** — örneklem küçük, çoğu
   gürültü. Bu panellerdeki özellik kararlarını **MASTER üzerinden** doğrula
   (hibrit korelasyon yaklaşımı tam bunu yapıyor → `MODELLER/_KORELASYON_HIBRIT/`).
3. **|r| = 0.2** MASTER'da güçlü kanıt, CFTR'de saf gürültü — aynı sayı panele göre
   tamamen farklı anlam taşır.
4. Sınıf dengesizliğini **modelleme** tarafında `scale_pos_weight`/`class_weight`
   ile yönet; bu korelasyon güvenilirliğinden ayrı bir konudur (örneklem boyutu).

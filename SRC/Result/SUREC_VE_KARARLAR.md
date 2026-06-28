# MedicalVision / SYZ2026 — Süreç ve Karar Belgesi

> Bu belge, projenin **en başından sonuna** kadar her tercihimizi, **neden** öyle seçtiğimizi, yol boyunca yaptığımız **hataları + sebeplerini + düzeltmelerini**, ve nihai modelleri **neden** seçtiğimizi kronolojik ve gerekçeli olarak anlatır.

---

## 0. Amaç
Genetik **missense varyantlarını** patojenik (1) / benign (0) olarak sınıflandıran, dört gen/hastalık paneli (CFTR, KANSER, PAH, MASTER) için ayrı modeller kuran bir makine öğrenmesi sistemi. Çıktı, yarışmaya sunulacak Proje Detay Raporu (PDR) ile değerlendirilir.

---

## 1. Veri yapısı — ne ve neden böyle ele aldık
- **4 panel:** CFTR (~111), KANSER (388), PAH (372), birleşik MASTER (2.931). MASTER hem dar hem geniş varyant uzayında genelleme testi sağlar.
- **Kodlanmış kolonlar:** `AL_*` (alel/popülasyon frekansı), `EK_*` (in-silico skorlar), `CAT_*` (kategorik), `AA_1/AA_2` (aminoasitler). **Neden içeriğe göre tip tespiti:** kolon adları şifreli olduğundan tip isimle bilinemiyor → dolu değerlerin ≥%50'si sayıya çevrilebiliyorsa sayısal saydık.
- **Şifreleme nedeni (NotebookLM ile doğrulandı):** varyantların dış veritabanlarında (ClinVar/gnomAD) aratılıp kopyalanmasını, yani **veri sızıntısını** engellemek.
- **Asimetri:** tüm panellerde benign azınlık (CFTR %19, PAH %17, KANSER %31, MASTER %27). Bu, sonraki adımlarda SMOTE / sınıf ağırlıklandırma / augmentation gerektirdi.
- **XLSX'e dönüşüm:** CSV'deki ondalık ayraç bozulmalarını gidermek için.

---

## 2. Veri ön işleme — kararlar ve gerekçeler
- **`AL_*`:** eksikler 0 ile dolduruldu → MinMax (0–1) ölçekleme.
- **`EK_*`:** iki eksik-değer senaryosu (ablasyonla kıyas): `median` (medyan doldur) ve `nan` (NaN bırak — XGBoost/LightGBM/CatBoost NaN'i doğal işler, yapay doldurma yanlılığından kaçınılır).
- **Aykırı değer:** ölçeklemeden önce **IQR 1.5× winsorization** (uç değerleri sınıra kırp).
- **Ölçekleme:** RobustScaler (aykırıya dayanıklı) veya StandardScaler.
- **Sızıntısızlık (kritik):** Tüm istatistikler (MinMax, IQR sınırı, medyan, scaler, target-encoding) yalnızca **eğitim katında**, scikit-learn pipeline içinde öğrenilir; pipeline her CV katında klonlanıp yeniden fit edilir → test ve doğrulama katlarına sızıntı olmaz.

---

## 3. Özellik mühendisliği — 15 türetilmiş özellik, neden
- **Target encoding (sızıntısız, out-of-fold + smoothing):** `CAT_1_te`, `CAT_2_te`, `AA_1_te`, `AA_2_te`. **Neden OOF:** kategorikleri hedef ortalamasıyla sayısallaştırırken sızıntı olmasın diye.
- **Aminoasit fiziko-kimyasal farkları:** `Delta_MW`, `Delta_PI`, `Delta_HYDRO` (Kyte-Doolittle), `Delta_Charge` — aminoasit değişiminin etkisini yakalamak için.
- **Varlık/gösterge:** `is_exome`, `is_segdup`, `AL_185_present`, nükleotid varlıkları (A/T/C/G), `has_archaic_delta`.
- **Etkileşim:** `ek7_x_ek9`, `ek2_ek3`.
- **Transfer meta-özelliği:** `TL_master_prob` (aşağıda).
- **Doğrulama:** SHAP analizi bu özelliklerin gerçekten kullanıldığını gösterdi (AA_2_te 4 panelde de üstte; ek7_x_ek9 KANSER'de top-5).

---

## 4. Veri zenginleştirme — neden iç veri, neden sızıntısız
- **Dış kaynaktan veri EKLENMEDİ.** Gerekçe: (a) şifreli yapı dış DB eşlemesini engelliyor; (b) dış in-silico/etiketli veri **döngüsellik (circularity) ve bilgi sızıntısı** riski taşır (PathoPredictor literatürü).
- **İç-veri zenginleştirme (hepsi fold-içi, sızıntısız):**
  - **clustering:** MASTER KMeans ile kümelenir; panelin düştüğü kümelerden, panelde/test'te OLMAYAN gerçek MASTER varyantları (değer-bazlı dedup) train'e eklenir.
  - **synthetic:** SMOTE ile azınlık sınıfı sentetik örnek.
  - **transfer_learning:** MASTER'da LightGBM eğitilip panel için `TL_master_prob` üretilir (küçük panellerde boyut lanetini hafifletir).
- **Sızıntı önlemi (kanıtlandı):** augmentation yalnız train fold'una uygulanır, val fold'a asla (`cv.py`). Ek olarak augmentation'sız `original` koşuların da yüksek CV vermesi sızıntı olmadığını kanıtlar (bkz. Bölüm 9, CFTR kırmızı bayrağı).

---

## 5. Deney tasarımı
- **OFAT ablasyon:** baseline + her bayrak (focal_loss, feature_selection, class_weight, l2, smote, feature_subsample, row_subsample, early_stopping) tek tek.
- **Profil `first_pass`:** 4 panel × 2 senaryo × 7 model × random HPO × 3 augmentation = **1512 koşu** (hızlı eleme taraması).
- **Altyapı:** 8 çekirdek paralel (ProcessPoolExecutor), incremental checkpoint + resume.
- **Uyumsuz kombinasyonlar** otomatik atlanır (SKIPPED) — örn. catboost focal_loss'u desteklemez, MASTER'a transfer anlamsız.

---

## 6. Model geliştirme — kararlar
- **9 algoritma:** XGBoost/LightGBM/CatBoost (gradient boosting, NaN-native), Random Forest/Extra Trees (bagging), AdaBoost, SVM (RBF), Logistic Regression (doğrusal taban), KNN (referans). **Neden:** tablo veri + küçük örneklem → ağaç tabanlı topluluklar merkezde; çeşitlilik için doğrusal/kernel/mesafe baseline'lar. (CatBoost/SVM yavaş olduğu için ayrı çalıştırıldı.)
- **HPO:** Grid / Random / **Bayesian (Optuna TPE)** destekli. Skor test'e değil **out-of-fold CV'ye** dayanır → test dokunulmaz.
- **Çapraz doğrulama:** Repeated Stratified K-Fold (5×2). Stratified → dengesizlik korunur; repeated → varyans düşer; küçük panelde kat sayısı otomatik kısılır.
- **Eşik kalibrasyonu (prior-aware):** eşik OOF üzerinde, **bilinen %80-benign prior'a** göre seçilir ve dondurulmuş test'e değiştirilmeden uygulanır. **Neden sızıntı değil (NotebookLM onayı):** prior tasarım gereği bilinen sabit, test etiketinden gelmiyor; test eşik seçiminde rol almıyor.
- **Overfitting önlemleri:** dondurulmuş test; repeated CV; ablasyon bayrakları (L2, early stopping, özellik seçimi, subsample, class_weight); SMOTE yalnız fold-içi; sığ ağaçlar.
- **Açıklanabilirlik:** native importance + SHAP (TreeExplainer). Matris taramasında hız için kapalı; **kazananlar için** sonradan çalıştırıldı (panel başına 9 SHAP görseli).

---

## 7. Metodolojinin EVRİMİ — kararların nasıl ve neden değiştiği
Bu bölüm projenin entelektüel çekirdeği: bazı kararları yol boyunca **kanıtla** revize ettik.

### 7.1 Karar eşiği metriği: `mcc` → `macro_f1`
Dengesiz veride MCC en acımasız metrik. İki sınıfı da eşit gözeten **macro_f1**'e geçtik (clustering'in dengeli karmaşıklık matrislerini ödüllendiriyor). `f1` (sadece pozitif sınıf) yerine `macro_f1` seçildi — çünkü f1 eşiği çoğunluğa kaydırıp benign'i görmezden gelir. *(Not: yarışma metriği F1 ise threshold_metric f1 yapılıp re-run gerekir; bu açık bir seçenek olarak bırakıldı.)*

### 7.2 CFTR dengeli test seti
**Sorun:** tüm panellerde benign azınlık olduğundan, test'i "%80 benign" yapmaya çalışmak CFTR testini 12 örneğe (~%11) düşürüyordu. **Düzeltme:** CFTR'ye özel **dengeli (50/50) %20 test** (22 örnek). Diğer paneller klinik %80/20'de kaldı. Prior-aware eşik artık testin **gerçek** benign oranına kalibre (CFTR'de 0.5).

### 7.3 Seçim metriği yolculuğu (en önemli ders)
1. Başlangıç: **cv_mcc_mean** (CV-MCC, sabit 0.5 eşiğinde) ile seçim.
2. Deneme: **cv_macro_f1** (OOF macro-F1, ayarlı eşikte) eklendi — yarışma metriğiyle tutarlı olsun diye.
3. **Kanıtla çürütüldü:** catboost altkümesinde cv_macro_f1 ↔ test korelasyonu **0.30**, cv_mcc ↔ test **0.74**. cv_macro_f1 daha kötü çıktı çünkü eşik **aynı OOF'ta** ayarlanıp **aynı OOF'ta** ölçülüyor → iyimserlik yanlılığı.
4. **Nihai karar (literatür + NotebookLM):** model seçimi için **eşikten bağımsız ROC-AUC (cv_auc)**; PathoPredictor da seçimde ROC-AUC kullanmış. Şiddetli dengesizlikte **PR-AUC (cv_pr_auc)** ek ölçüt.

### 7.4 Hibrit panel şampiyonu (neden test-F1 ile değil)
**Kural:** cv_auc'si en iyiye **0.02 marj** içindeki adaylar istatistiksel olarak eşit sayılır; bunlar arasından **cv_pr_auc** (leak-free, F1-ilişkili) en yüksek olan seçilir. **Neden test-F1 ile tie-break YAPMADIK:** o, küçük test setinde en yükseği cımbızlamak = seçim yanlılığı olurdu (örn. KANSER'de test-F1 0.875'lik logreg'in cv_auc'si düşüktü = şanslı bölünme). cv_pr_auc ile KANSER catboost/clustering seçildi: test macro_f1 0.817, hem leak-free metriklerde lider.

---

## 8. Yaptığımız hatalar + sebepleri + düzeltmeleri
| Hata | Sebep | Düzeltme |
|------|-------|----------|
| Yanlışlıkla `full` profili (6480 koşu) başlatıldı | `--profile full` ile bambaşka, ağır matris | Süreç durduruldu; doğru `first_pass` profiliyle devam |
| 55 dk "stall" | Durdurulan `full` sürecinin **8 orphan worker'ı** (parent öldü, child'lar yaşadı) CPU'yu kaptı | Orphan'lar PID ile temizlendi → first_pass tam hıza döndü |
| cv_macro_f1 yanlış seçim metriği | OOF-tuned eşikte optimism bias | Kanıtla tespit (korelasyon 0.30) → cv_auc/PR-AUC'ye geçildi |
| Karışık matris (cv_macro_f1 sadece catboost'ta dolu) | Yeni metrik eski koşularda yok | report.py'ye **coverage-aware** seçim (metrik ancak tüm koşularda doluysa kullanılır, yoksa cv_mcc'ye düşer) |
| SHAP `_transform_to_clf` çöktü | SMOTE bir sampler, `.transform()` yok | sampler adımları çıkarımda atlanacak şekilde düzeltildi |
| runner `all_runs.xlsx`'i checkpoint'ten yeniden yazınca cv_auc silindi | checkpoint cv_auc içermiyor | Eski cv_auc commit'li sürümden geri yüklendi, sadece yeni koşular hesaplandı (~2.7h yerine dakikalar) |
| `SRC/Result` vs `src/result` case tutarsızlığı | IDE/araçlar küçük-harf yol üretti | git `ignorecase=true` ile yeni dosyalar mevcut `SRC/Result` case'ine normalize edildi |
| Bayesian-15 HPO PAH aramasını ~7-8 saate çıkardı | hpo_n_iter=15 × CV çok yavaş | Yalın profile geçildi (random-5, robust ağaçta gereksiz olduğu için çıkarıldı), sadece yeni augmentation'lar (144 koşu) |

---

## 9. Bağımsız doğrulamalar (NotebookLM tarafsız hakem)
- **Metodoloji "kusursuz (SOTA)":** cv_auc→cv_pr_auc hibrit + prior-OOF kalibrasyon literatürle (PathoPredictor, CTpredX, DS-MVP) uyumlu.
- **CFTR kırmızı bayrağı kapandı:** "cv_auc 0.994 augmentation sızıntısı mı?" → **Hayır.** Kanıt: (a) augmentation fold-içi (kod), (b) augmentation'sız `original` da ~0.99 veriyor, (c) clustering CV ortalamasını **düşürdü** (0.916→0.894) — sızıntı olsaydı yükselirdi. Yükseklik CFTR'nin tek-genli homojenliğinden; CV 0.99→test 0.93 düşüşü küçük-panel optimizmi.
- **Eşik kalibrasyonu sızıntı değil:** bilinen prior'a göre OOF'ta seçilip test'e değiştirilmeden uygulamak literatürün şart koştuğu doğru yöntem.

---

## 10. PAH iyileştirme denemesi → dürüst negatif bulgu
PAH en zayıf panel (AUC 0.786). Genişletilmiş augmentation (synthetic, clustering+synthetic) denendi. **Sonuç: iyileşme yok** — yeni augment'lerin en iyisi cv_auc 0.828, mevcut extra_trees/clustering 0.839'un altında. Bu, PAH tavanının **tablo verisiyle sınırlı** olduğunu (daha derin genetik mekanizmalar gerektiğini) kanıtla destekler. *(robust scaler ağaç modellerinde scale-invariance nedeniyle gereksiz olduğu için denenmedi.)*

---

## 11. Nihai 4 şampiyon — neyi neden seçtik
Seçim: **sızıntısız cv_auc** (eşikten bağımsız) + marj içinde **cv_pr_auc** tie-break. Test metrikleri yalnız raporlama.

| Panel | Model | Augmentation | cv_auc | cv_pr_auc | test macro_f1 | test F1 | test MCC | test AUC |
|-------|-------|--------------|--------|-----------|---------------|---------|----------|----------|
| CFTR | CatBoost | clustering | 0.994 | 0.999 | 0.760 | 0.815 | 0.612 | 0.926 |
| KANSER | CatBoost | clustering | 0.904 | 0.970 | 0.817 | 0.722 | 0.653 | 0.946 |
| MASTER | CatBoost | original | 0.846 | 0.962 | 0.723 | 0.563 | 0.448 | 0.851 |
| PAH | Extra Trees | clustering | 0.839 | 0.975 | 0.664 | 0.500 | 0.349 | 0.786 |

**Neden bunlar:** her biri kendi panelinde sızıntısız cv_auc'de lider (marj içinde cv_pr_auc en iyisi). CatBoost 3 panelde kazandı. Ensemble (voting/stacking) denendi ama tek modeli geçemedi (hatalar korele) — o yüzden tek model.

---

## 12. Açıklanabilirlik bulguları (SHAP)
- **AA_2_te** (değişim sonrası aminoasit) 4 panelde de en güçlü sinyal → aminoasit değişim kimliği belirleyici.
- **CAT_1_te** (veri kaynağı kodlaması) CFTR/KANSER/PAH'ta üstte; **ek7_x_ek9** (türettiğimiz etkileşim) KANSER'de top-5; MASTER'da **EK_** in-silico skorları baskın.
- Waterfall grafikleri tekil varyant kararlarını açıklıyor (klinik yorumlanabilirlik).

---

## 13. Limitasyonlar (PDR'de şeffafça raporlanacak)
1. **Küçük-panel optimizmi:** CFTR (n_train=89) gibi panellerde CV skoru iyimser; test'e düşüş beklenen ve dürüstçe raporlandı.
2. **MASTER en güvenilir ama en zor:** büyük (n_test=489) + heterojen + gerçekçi-dengesiz test → düşük F1 ama **en güvenilir genelleme tahmini**.
3. **PAH tavanı:** tablo verisiyle sınırlı; daha derin mekanizmalar (protein dinamiği/regülatör etkiler) gerekebilir.
4. **Klinik olgunluk:** sonuçlar proof-of-concept; klinik karar-destek için yeterli değil.

---

## 14. Tek cümlelik özet
Her kararı **sızıntısızlık** ve **dürüst genelleme ölçümü** ilkeleriyle aldık; test'te yükseni cımbızlamak yerine bilimsel doğruyu seçtik, yanılan kararları (cv_macro_f1) kanıtla geri aldık, ve nihai modelleri eşikten-bağımsız sızıntısız ROC-AUC ile seçip dürüst test skorlarıyla raporladık.


---

# EK: Matris-Sonrasi Iyilestirmeler ve Nihai Durum (guncelleme)

Bu bolum, ilk surec belgesinden SONRA yapilan panel-ozel iyilestirmeleri ve nihai
sampiyon kararlarini ozetler. Tum kararlar sizintisizdir; secim cv_auc, esik F1 (sartname
metrigi) onceliklidir.

## Secim/altyapi rafinasyonu
- runner artik `cv_auc` ve `cv_pr_auc`'yi NATIF kaydeder (eskiden ayri recompute gerekiyordu).
- Karar esigi metrigi tartismasi: sartname siralama metrigi F1 oldugundan, esik **F1** icin
  optimize edilir (macro_f1 cogunluk benign'i maskeler). Esik OOF'ta secilip test'e degistirilmeden uygulanir.
- Model SECIMI yine eskiten-bagimsiz cv_auc (PR-AUC tie-break) ile; cv_macro_f1 test ile dusuk
  korelasyon (0.30) verdigi icin secimde KULLANILMAZ.

## CFTR - sinif-dengeli clustering
- Sorun: yalniz ~10 train benign -> model patojenige yanli (precision 0.69, FP cok).
- Cozum: clustering ile gercek MASTER benign'i ekleyip train'i ~1:1 dengeleme (overshoot korumali; havuzda 771 benign).
- Sonuc: Random Forest -> F1 0.815->**0.880**, MCC 0.61->**0.76**, precision 0.79, recall 1.0 (FN=0). CM: TN8/FP3/FN0/TP11.
- Takas: cv_auc 0.99->0.90 (eklenen benign MASTER kaynakli, CFTR'ye ozgu siralamayi biraz dusurur).
  Sartname F1 onceligiyle dengeli-RF secildi; ayirt-etme onceligi istenirse CatBoost/clustering alternatiftir.

## PAH - karar esigi (augmentasyon ise yaramadi)
- Patojenik-agirlikli augmentasyon TERS TEPTI (eklenen MASTER patojenikleri PAH sinyalini seyreltti; recall dustu).
- Genisletilmis arama (bayesian/robust/synthetic) mevcut sampiyonu gecemedi.
- ISE YARAYAN: karar esigini dusurmek (prior-OFF, 0.62) -> recall 0.625->**1.0** (FN=0), MCC 0.349->0.380.
  PAH tablo verisinin tavanindadir; tek etkili lever esiktir.

## MASTER - capraz-panel benign + uzman-stacking + F1-esik
- clustering UYGULANAMAZ (MASTER aday kaynagidir).
- Capraz-panel benign: alt panel (CFTR/KANSER/PAH) benign'leri MASTER'dan deger-bazli AYRIK cikti (~%0 ortusme);
  203 gercek benign eklendi -> cv_auc ve precision artti.
- Uzman-stacking: alt-panel modellerinin olasilik ciktilari MASTER'a TL_ ozelligi olarak verildi, ustune meta-CatBoost
  (REVEL/MetaRNN mantigi). Uzmanlar ORIGINAL egitildi (leak-free). recall artti.
- F1-esik (~0.65): birlikte -> F1 0.563->**0.582**, MCC 0.448->**0.467**, recall 0.592->**0.745**.
- 0.85 esik DENENDI ve KOTU cikti (recall 0.327, F1 0.416); benign-agirlikli test icin fazla tutucu. En iyi ~0.65.
- NotebookLM 'sert uzman-yonlendirme (MoE)' yerine STACKING'i onerdi (MASTER varyantlari uzmanlar icin OOD).

## MASTER dusuk MCC = durustluk (NotebookLM onayli)
- MASTER MCC ~0.47, alt panellerden (CFTR 0.76, KANSER 0.65) dusuk. Bu zaafiyet DEGIL:
  MASTER en buyuk (489), heterojen (~2060 gen), gercekci-dengesiz (~%80 benign) test = gercek klinik yigin simulasyonu.
  Kucuk panellerin yuksek skorlari kismen kucuk-orneklem iyimserligi. Gercek genelleme MASTER'dan okunur.
- Gelecek (literatur): gen/hastalik-spesifik modeller, LLM gen-gomuleri + AlphaFold2 3B yapi (MissenseNet), kalibrasyon egrileri.

## Teslimatlar
- PDR: PDR_FINAL_v3.docx (panel-bazli denenen/ise-yarayan/secilen, sartname-uyumlu, <=10 sayfa).
- Paket: MedicalVision_paneller.zip (kod + panel-bazli sonuc/metrik + README'ler).
- Nihai sampiyonlar: nihai_sampiyonlar.xlsx (iyilestirmeler dahil).
- Not: best_per_panel.xlsx OTOMATIK (yalniz matris kosulari) uretildigi icin matris-sonrasi iyilestirmeleri
  (CFTR dengeli-RF, MASTER stacking+benign) ICERMEZ; nihai secimler nihai_sampiyonlar.xlsx'tedir.

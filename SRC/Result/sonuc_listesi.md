# MedicalVision / SYZ2026 — Model Sonuç Listesi (GÜNCEL)

> Bu belge güncellendi. Eski `cv_mcc_mean` sıralaması terk edildi; nihai seçim **sızıntısız cv_auc**
> (eşikten bağımsız) + **cv_pr_auc** tie-break ile, karar eşiği **F1-önceliği** ile yapılır.
> Tam tablo: `nihai_sampiyonlar.xlsx` · cv_auc referansı: `best_per_panel.xlsx` · gerekçeler: `SUREC_VE_KARARLAR.md`.

## Metodoloji (güncel)
- **Model seçimi:** sızıntısız ROC-AUC (cv_auc, Repeated Stratified 5×2 CV) — eşikten bağımsız; yakın adaylar cv_pr_auc ile ayrılır. Dondurulmuş TEST yalnız raporlama içindir.
- **Karar eşiği:** şartname metriği F1 olduğundan F1-farkında; OOF'ta seçilip test'e değiştirilmeden uygulanır (panel-bazlı operasyon noktası).
- **Test setleri:** CFTR=22 (dengeli 11/11), KANSER=75, PAH=39, MASTER=489 (~%80 benign).
- **Uyarı:** yüksek TEST + düşük cv_auc = küçük sette şanslı bölünme (selection-bias); seçimde kullanılmaz.

## NİHAİ ŞAMPİYONLAR (cv_auc + benimsenen iyileştirmeler, F1-önceliği)

| Panel | Model / Müdahale | Ablasyon | Eşik | F1 | MCC | recall | cv_auc |
|-------|------------------|----------|------|-----|-----|--------|--------|
| **CFTR** | CatBoost / clustering | feature_subsample | 0.90 | 0.815 | 0.612 | 1.00 | 0.994 |
| **KANSER** | CatBoost / clustering | smote | 0.80 | 0.722 | 0.653 | 0.867 | 0.904 |
| **PAH** | Extra Trees / clustering (düşük eşik) | class_weight | 0.62 | 0.485 | 0.380 | 1.00 | 0.839 |
| **MASTER** | CatBoost / +çapraz-benign +uzman-stacking +F1-eşik | class_weight | 0.65 | 0.582 | 0.467 | 0.745 | 0.854 |

## Saf cv_auc liderleri (REFERANS — best_per_panel.xlsx)
- CFTR: CatBoost/clustering (cv_auc 0.994, F1 0.815) = nihai ile aynı (AUC-tutarlı seçim; RF/original F1 0.880 daha yüksekti ama cv_auc 0.916 düşük → küçük-test F1 iyimserliği riski, seçilmedi).
- KANSER: CatBoost/clustering/smote (cv_auc 0.904) = nihai ile aynı.
- MASTER: CatBoost/original (cv_auc 0.846) — nihai stacking+benign ile geçildi.
- PAH: Extra Trees/clustering (cv_auc 0.839) = nihai ile aynı (eşik düşürüldü).

## Ensemble durumu
- Panel-içi top-3 voting/stacking DENENDİ ama tek-en-iyi modeli **geçemedi** (hatalar korele) → tek model kullanıldı.
- **MASTER'da uzman-stacking** (alt-panel olasılıkları → meta-özellik, REVEL tarzı, leak-free) **benimsendi** (F1 0.563→0.582).

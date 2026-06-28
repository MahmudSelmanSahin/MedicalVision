# CFTR Paneli

## Uygulananlar
4 senaryo (std/robust x median/nan) x 9 model x (original/clustering/transfer/synthetic) x OFAT ablasyon; DENGELI %20 test (benign kit oldugu icin); ek olarak sinif-dengeli clustering (gercek MASTER benign'i ~1:1).

## Ise yarayan / yaramayan
RF/original (F1 0.880) ve sinif-dengeli clustering DENENDI; F1 acisindan cazip olsalar da cv_auc'leri dusuktu (RF 0.916).
Kucuk-test (22 ornek) F1 iyimserligine guvenmemek icin (cv_macro_f1<->test korelasyonu 0.30) bunlar SECILMEDI.

## Secilen model (sizintisiz cv_auc lideri — AUC-tutarli)
CatBoost / clustering / feature_subsample. NEDEN: dort panelle tutarli sekilde **cv_auc lideri (0.994)** ile secildi (istisna yok).
- En iyi ablasyon: **feature_subsample**
- Test metrikleri: F1=0.815  MCC=0.612  macro-F1=0.760  recall=1.00 (FN=0)  precision=0.688  ROC-AUC=0.926  | cv_auc=0.994
- Karar evrimi: ara asamada RF/original (F1 0.880, cv_auc 0.916) F1-onceligiyle dusunuldu; cv_auc dusuk oldugundan AUC-tutarliligi icin CatBoost'a donuldu.

## Yeniden calistirma
```
python SRC/modeling/runner.py --profile cftr_balanced --results-dir SRC/Result/runs_cftr_bal
```
Model secimi sizintisiz cv_auc ile yapilir; test metrikleri yalniz raporlama icindir.
Bu klasor: `CFTR_tum_kosular.xlsx` (tum konfigler) + `explain/` (SHAP/onem/karmasiklik matrisi).

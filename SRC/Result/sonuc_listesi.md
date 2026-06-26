# MedicalVision / SYZ2026 — Model Sonuc Listesi (panel basina secim icin)

## Metodoloji (onemli)
- Model SECIMI sizintisiz capraz-dogrulama (cv_mcc_mean = Repeated Stratified 5-fold CV uzerinde MCC) ile yapilir; dondurulmus TEST seti yalnizca nihai raporlama icindir.
- cv_mcc_mean ~ test macro_f1 korelasyonu 0.74 (catboost altkumesinde); yani CV guvenilir bir gosterge.
- Test setleri: CFTR=22 (DENGELI 11/11), KANSER=75 (~%80 benign), PAH=39 (~%80 benign), MASTER=489 (~%80 benign).
- UYARI: yuksek TEST skoru + dusuk CV skoru = kucuk test setinde sansli bolunme (lucky split), klinik olarak guvenilmez.

## CFTR paneli — en iyi 6 aday (cv_mcc_mean sirali)
model | data_aug | senaryo | cv_mcc_mean(LEAK-FREE) | test_macro_f1 | test_mcc | test_auc
---|---|---|---|---|---|---
extra_trees | transfer_learning | ek-standard_nan | 0.776 | 0.812 | 0.683 | 0.905
extra_trees | transfer_learning | ek-standard_median | 0.776 | 0.812 | 0.683 | 0.905
xgboost | transfer_learning | ek-standard_median | 0.770 | 0.646 | 0.471 | 0.777
xgboost | transfer_learning | ek-standard_nan | 0.770 | 0.646 | 0.471 | 0.777
xgboost | transfer_learning | ek-standard_nan | 0.742 | 0.760 | 0.612 | 0.806
xgboost | transfer_learning | ek-standard_median | 0.742 | 0.760 | 0.612 | 0.806

## KANSER paneli — en iyi 6 aday (cv_mcc_mean sirali)
model | data_aug | senaryo | cv_mcc_mean(LEAK-FREE) | test_macro_f1 | test_mcc | test_auc
---|---|---|---|---|---|---
xgboost | transfer_learning | ek-standard_median | 0.793 | 0.755 | 0.579 | 0.913
xgboost | transfer_learning | ek-standard_nan | 0.792 | 0.694 | 0.497 | 0.918
adaboost | transfer_learning | ek-standard_nan | 0.786 | 0.724 | 0.564 | 0.896
adaboost | transfer_learning | ek-standard_nan | 0.786 | 0.724 | 0.564 | 0.896
adaboost | transfer_learning | ek-standard_median | 0.786 | 0.724 | 0.564 | 0.896
adaboost | transfer_learning | ek-standard_median | 0.786 | 0.724 | 0.564 | 0.896

## PAH paneli — en iyi 6 aday (cv_mcc_mean sirali)
model | data_aug | senaryo | cv_mcc_mean(LEAK-FREE) | test_macro_f1 | test_mcc | test_auc
---|---|---|---|---|---|---
catboost | clustering | ek-standard_nan | 0.461 | 0.567 | 0.319 | 0.746
catboost | clustering | ek-standard_nan | 0.448 | 0.632 | 0.394 | 0.754
catboost | clustering | ek-standard_median | 0.439 | 0.567 | 0.319 | 0.766
catboost | clustering | ek-standard_median | 0.439 | 0.567 | 0.319 | 0.766
catboost | clustering | ek-standard_median | 0.439 | 0.567 | 0.319 | 0.766
catboost | transfer_learning | ek-standard_median | 0.427 | 0.545 | 0.295 | 0.677

## MASTER paneli — en iyi 6 aday (cv_mcc_mean sirali)
model | data_aug | senaryo | cv_mcc_mean(LEAK-FREE) | test_macro_f1 | test_mcc | test_auc
---|---|---|---|---|---|---
catboost | clustering | ek-standard_median | 0.485 | 0.732 | 0.471 | 0.851
catboost | original | ek-standard_median | 0.485 | 0.732 | 0.471 | 0.851
catboost | original | ek-standard_nan | 0.477 | 0.709 | 0.418 | 0.851
catboost | clustering | ek-standard_nan | 0.477 | 0.709 | 0.418 | 0.851
catboost | original | ek-standard_median | 0.475 | 0.698 | 0.395 | 0.832
catboost | clustering | ek-standard_median | 0.475 | 0.698 | 0.395 | 0.832

## Ensemble (top-3 soft voting + stacking) vs tek-en-iyi model
panel | birlestirilen | voting_mcc | stack_mcc | tek_en_iyi_mcc | voting_macro_f1 | tek_en_iyi_macro_f1
---|---|---|---|---|---|---
CFTR | extra_trees;xgboost;logreg | 0.542 | 0.542 | 0.683 | 0.705 | 0.812
KANSER | xgboost;adaboost;lightgbm | 0.512 | 0.497 | 0.579 | 0.706 | 0.755
MASTER | catboost;random_forest;lightgbm | 0.446 | 0.454 | 0.471 | 0.721 | 0.732
PAH | catboost;extra_trees;xgboost | 0.368 | 0.343 | 0.319 | 0.610 | 0.567
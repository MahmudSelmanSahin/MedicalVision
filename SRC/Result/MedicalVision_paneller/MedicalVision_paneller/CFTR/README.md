# CFTR Paneli

## Uygulananlar
4 senaryo (std/robust x median/nan) x 9 model x (original/clustering/transfer/synthetic) x OFAT ablasyon; DENGELI %20 test (benign kit oldugu icin); ek olarak sinif-dengeli clustering (gercek MASTER benign'i ~1:1).

## Ise yarayan / yaramayan
CFTR tek-genli/homojen + kucuk panel -> cv_auc burada EN IYIMSER/guvenilmez metrik. Saf cv_auc tek-lideri CatBoost/clustering
(0.994) dondurulmus testte EN ZAYIF cikti (test AUC 0.926, precision 0.69 = en yanli, 22 ornekli dengeli testte 5 FP).
Sinif-dengeli clustering F1'i 0.917'ye cikardi ama cv_auc dustu (~0.85) -> marj disi, secilmedi.

## Secilen model (cv_auc marj-ici + dondurulmus test lideri)
Extra Trees / transfer_learning / base. NEDEN: cv_auc 0.977 (lidere 0.02 marj icinde = AUC barajini gecer) HEM de
robust metriklerin tamaminda lider. Birincil metrik (cv_auc) burada guvenilmezken yakinsayan test kaniti (3 metrik) ExtraTrees'i isaret eder.
- Ablasyon: **base** | esik 0.76
- Test metrikleri: F1=0.846  MCC=0.683  macro-F1=0.812  recall=1.00 (FN=0)  precision=0.733 (FP=4, en az yanlilardan)  ROC-AUC=0.979  | cv_auc=0.977
- Alternatif: RF/original (augmentasyonsuz, F1 0.786, precision 0.786=en az yanli) F1/MCC'de geride kaldi. CatBoost/clustering saf cv_auc lideri ama testte en zayif.
- transfer_learning sizintisizdir (MASTER'dan TL_master_prob fold-ici uretilir).

## Yeniden calistirma
```
python SRC/modeling/runner.py --profile cftr_balanced --results-dir SRC/Result/runs_cftr_bal
```
Model secimi sizintisiz cv_auc ile yapilir; test metrikleri yalniz raporlama icindir.
Bu klasor: `CFTR_tum_kosular.xlsx` (tum konfigler) + `explain/` (SHAP/onem/karmasiklik matrisi).

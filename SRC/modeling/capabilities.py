"""
===========================================================================
MODEL YETENEK MATRISI ve UYGULANABILIRLIK KONTROLU
===========================================================================
SYZ2026 / MedicalVision

Her modelin hangi teknikleri DESTEKLEDIGINI tanimlar. Runner, bir kosu
(model + ablasyon kombinasyonu) gecersizse (ör. KNN + focal_loss) bu tabloya
gore otomatik ATLAR ve raporda "uygulanamaz" olarak isaretler.

Bayraklar:
  nan_native        : NaN'i dogal isler (imputer gerekmez)
  needs_scaling     : mesafe/lineer tabanli -> tum-ozellik StandardScaler sart
  focal_loss        : focal loss objective destekler
  class_weight      : sinif agirliklandirma destekler
  l2                : L2 regularization parametresi var
  early_stopping    : eval-set ile erken durdurma destekler
  feature_subsample : kolon alt-orneklemesi (yerlesik)
  row_subsample     : satir alt-orneklemesi (yerlesik)
  native_importance : yerlesik feature_importances_/coef_ var
  shap_explainer    : tercih edilen SHAP explainer turu
===========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Caps:
    nan_native: bool = False
    needs_scaling: bool = False
    focal_loss: bool = False
    class_weight: bool = True
    l2: bool = False
    early_stopping: bool = False
    feature_subsample: bool = False
    row_subsample: bool = False
    native_importance: bool = True
    shap_explainer: str = "kernel"     # tree | linear | kernel | deep
    family: str = "other"              # gbdt | bagging | linear | svm | knn | dnn | prior


CAPABILITIES: dict[str, Caps] = {
    "xgboost": Caps(
        nan_native=True, focal_loss=True, class_weight=True, l2=True,
        early_stopping=True, feature_subsample=True, row_subsample=True,
        shap_explainer="tree", family="gbdt"),
    "lightgbm": Caps(
        nan_native=True, focal_loss=True, class_weight=True, l2=True,
        early_stopping=True, feature_subsample=True, row_subsample=True,
        shap_explainer="tree", family="gbdt"),
    "catboost": Caps(
        nan_native=True, focal_loss=False, class_weight=True, l2=True,
        early_stopping=True, feature_subsample=True, row_subsample=True,
        shap_explainer="tree", family="gbdt"),
    "random_forest": Caps(
        nan_native=False, class_weight=True, l2=False, early_stopping=False,
        feature_subsample=True, row_subsample=True,
        shap_explainer="tree", family="bagging"),
    "extra_trees": Caps(
        nan_native=False, class_weight=True, l2=False, early_stopping=False,
        feature_subsample=True, row_subsample=True,
        shap_explainer="tree", family="bagging"),
    "adaboost": Caps(
        nan_native=False, class_weight=False, l2=False, early_stopping=False,
        feature_subsample=False, row_subsample=False,
        shap_explainer="tree", family="bagging"),
    "svm": Caps(
        nan_native=False, needs_scaling=True, class_weight=True, l2=True,
        early_stopping=False, feature_subsample=False, row_subsample=False,
        native_importance=False, shap_explainer="kernel", family="svm"),
    "knn": Caps(
        nan_native=False, needs_scaling=True, class_weight=False, l2=False,
        early_stopping=False, feature_subsample=False, row_subsample=False,
        native_importance=False, shap_explainer="kernel", family="knn"),
    "logreg": Caps(
        nan_native=False, needs_scaling=True, class_weight=True, l2=True,
        early_stopping=False, feature_subsample=False, row_subsample=False,
        shap_explainer="linear", family="linear"),
    "tabnet": Caps(
        nan_native=False, needs_scaling=True, focal_loss=True, class_weight=True,
        l2=True, early_stopping=True, feature_subsample=False, row_subsample=False,
        shap_explainer="deep", family="dnn"),
    "tabpfn": Caps(
        nan_native=False, needs_scaling=True, focal_loss=False, class_weight=False,
        l2=False, early_stopping=False, feature_subsample=False, row_subsample=False,
        native_importance=False, shap_explainer="kernel", family="prior"),
}

# Ablasyon bayragi -> gerekli yetenek alani (None ise her modelde gecerli)
ABLATION_REQUIRES: dict[str, str | None] = {
    "focal_loss": "focal_loss",
    "class_weight": "class_weight",
    "l2": "l2",
    "early_stopping": "early_stopping",
    "feature_subsample": "feature_subsample",
    "row_subsample": "row_subsample",
    "feature_selection": None,   # her modelde uygulanabilir (pipeline disinda)
    "smote": None,               # her modelde uygulanabilir
}

# TabPFN limitleri (v1)
TABPFN_MAX_SAMPLES = 1000
TABPFN_MAX_FEATURES = 100


def applicable(model: str, active_ablations: dict[str, str]) -> tuple[bool, str]:
    """active_ablations: {bayrak: 'on'|'off'}.
    'on' olan her bayrak icin modelin yetenegini kontrol eder.
    Donus: (uygulanabilir_mi, gerekce). Uygulanamaz toggle 'off' yapilarak
    kosu yine de calistirilabilir; runner buna karar verir."""
    caps = CAPABILITIES[model]
    for flag, state in active_ablations.items():
        if state != "on":
            continue
        req = ABLATION_REQUIRES.get(flag)
        if req is None:
            continue
        if not getattr(caps, req):
            return False, f"{model} '{flag}' desteklemiyor"
    return True, ""

"""
===========================================================================
SIZINTISIZ OZELLIK HATTI (Leakage-safe Feature Pipeline)
===========================================================================
SYZ2026 / MedicalVision

Mevcut imputation.py + feature_extraction.py mantigini TEK bir sklearn
transformer'ina tasir. Tum istatistikler (MinMax, clip sinirlari, medyan,
scaler, target-encoding) YALNIZCA fit() icinde (train'de) ogrenilir; test'e
sadece transform uygulanir. Boylece:

  * Klinik hold-out test setine TE/scaler sizintisi OLMAZ.
  * Transformer sklearn Pipeline icine konup CV'de her fold'da yeniden
    fit edildiginde fold-ici sizinti da onlenir.

Senaryo, dosya degil PARAMETREDIR:
  ek_scaler  : 'robust' | 'standard'
  ek_missing : 'median' | 'nan'
  -> 'ek-standard_nan' = FeaturePipeline(ek_scaler='standard', ek_missing='nan')

Girdi : ham DataFrame (Variant_ID, AL_*, EK_*, CAT_*, AA_1, AA_2, ...)
Cikti : DataFrame (AL_* olcekli, EK_* olcekli/nan, turetilen ozellikler)
        Variant_ID ve ham CAT_/AA_ kolonlari DUSURULUR.
===========================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Aminoasit fiziko-kimyasal tablolari (feature_extraction.py ile ayni)
AA_MW = {"G": 75.07, "A": 89.09, "S": 105.09, "P": 115.13, "V": 117.15,
         "T": 119.12, "C": 121.16, "L": 131.17, "I": 131.17, "N": 132.12,
         "D": 133.10, "Q": 146.15, "K": 146.19, "E": 147.13, "M": 149.21,
         "H": 155.16, "F": 165.19, "R": 174.20, "Y": 181.19, "W": 204.23}
AA_PI = {"A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07, "E": 3.22,
         "Q": 5.65, "G": 5.97, "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74,
         "M": 5.74, "F": 5.48, "P": 6.30, "S": 5.68, "T": 5.60, "W": 5.89,
         "Y": 5.66, "V": 5.96}
AA_HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
            "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
            "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
            "Y": -1.3, "V": 4.2}
AA_CHARGE = {"A": 0, "R": 1, "N": 0, "D": -1, "C": 0, "Q": 0, "E": -1, "G": 0,
             "H": 0.1, "I": 0, "L": 0, "K": 1, "M": 0, "F": 0, "P": 0, "S": 0,
             "T": 0, "W": 0, "Y": 0, "V": 0}

ID_COL = "Variant_ID"
TARGET = "Label"


def _clip_bounds(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return None
    return q1 - k * iqr, q3 + k * iqr


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Leakage-safe imputation + feature extraction.

    Parametreler
    ------------
    ek_scaler  : {'robust','standard'}
    ek_missing : {'median','nan'}
    te_smoothing : target-encoding smoothing katsayisi
    """

    def __init__(self, ek_scaler: str = "standard", ek_missing: str = "nan",
                 te_smoothing: float = 10.0):
        self.ek_scaler = ek_scaler
        self.ek_missing = ek_missing
        self.te_smoothing = te_smoothing

    # -- yardimcilar --------------------------------------------------------
    @staticmethod
    def _aa(df, col):
        s = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)
        return s.astype("string").str.strip().str.upper()

    @staticmethod
    def _num(df, col):
        s = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        return pd.to_numeric(s, errors="coerce")

    def _fit_te(self, cat: pd.Series, y: pd.Series):
        """Smoothed target-encoding haritasini TRAIN'de ogrenir."""
        cat = cat.astype("string").fillna("__MISSING__")
        y = y.astype(float)
        gmean = y.mean()
        stats = y.groupby(cat).agg(["mean", "count"])
        enc = (stats["count"] * stats["mean"] + self.te_smoothing * gmean) \
            / (stats["count"] + self.te_smoothing)
        return enc.to_dict(), gmean

    def _apply_te(self, cat: pd.Series, enc: dict, gmean: float):
        cat = cat.astype("string").fillna("__MISSING__")
        return cat.map(enc).astype(float).fillna(gmean)

    # -- fit ----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        df = X
        y = pd.Series(np.asarray(y), index=df.index).astype(float)

        self.al_cols_ = [c for c in df.columns if c.startswith("AL_")]
        self.ek_cols_ = [c for c in df.columns if c.startswith("EK_")]

        # AL: eksikler 0 (biyolojik nadirlik sinyali) -> MinMax
        self.al_scaler_ = None
        if self.al_cols_:
            al = df[self.al_cols_].apply(pd.to_numeric, errors="coerce").fillna(0)
            self.al_scaler_ = MinMaxScaler().fit(al)

        # EK: clip sinirlari -> (medyan) -> scaler  (hepsi train'de)
        self.ek_bounds_, self.ek_median_, self.ek_scaler_obj_ = {}, {}, None
        if self.ek_cols_:
            ek = df[self.ek_cols_].apply(pd.to_numeric, errors="coerce")
            for c in self.ek_cols_:
                b = _clip_bounds(ek[c])
                self.ek_bounds_[c] = b
                if b is not None:
                    ek[c] = ek[c].clip(*b)
            if self.ek_missing == "median":
                self.ek_median_ = ek.median(numeric_only=True).to_dict()
                ek = ek.fillna(pd.Series(self.ek_median_))
            scaler = RobustScaler() if self.ek_scaler == "robust" else StandardScaler()
            self.ek_scaler_obj_ = scaler.fit(ek)

        # Target encoding haritalari (train'de)
        self.te_ = {}
        for col in ("CAT_1", "CAT_2"):
            src = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)
            self.te_[col] = self._fit_te(src, y)
        self.te_["AA_1"] = self._fit_te(self._aa(df, "AA_1"), y)
        self.te_["AA_2"] = self._fit_te(self._aa(df, "AA_2"), y)

        # Cikti kolon sirasi (transform'da sabitlenir)
        self.feature_names_ = None
        self.feature_names_ = list(self.transform(df).columns)
        return self

    # -- transform ----------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X
        cols: dict[str, np.ndarray] = {}   # parça parça topla, sonda tek concat

        # AL olcekli
        if self.al_cols_ and self.al_scaler_ is not None:
            al = df.reindex(columns=self.al_cols_).apply(pd.to_numeric, errors="coerce").fillna(0)
            arr = self.al_scaler_.transform(al)
            for i, c in enumerate(self.al_cols_):
                cols[c] = arr[:, i]

        # EK clip -> (medyan) -> scale
        if self.ek_cols_ and self.ek_scaler_obj_ is not None:
            ek = df.reindex(columns=self.ek_cols_).apply(pd.to_numeric, errors="coerce")
            for c in self.ek_cols_:
                b = self.ek_bounds_.get(c)
                if b is not None:
                    ek[c] = ek[c].clip(*b)
            if self.ek_missing == "median" and self.ek_median_:
                ek = ek.fillna(pd.Series(self.ek_median_))
            arr = self.ek_scaler_obj_.transform(ek)   # 'nan' modunda NaN korunur
            for i, c in enumerate(self.ek_cols_):
                cols[c] = arr[:, i]

        # --- Turetilen ozellikler ---
        c1 = (df["CAT_1"] if "CAT_1" in df else pd.Series(pd.NA, index=df.index)).astype("string")
        is_exome = pd.Series(np.nan, index=df.index)
        is_exome[c1.str.contains("gnomADe", case=False, na=False)] = 1
        is_exome[c1.str.contains("gnomADg", case=False, na=False)] = 0
        cols["is_exome"] = is_exome.to_numpy()

        cols["CAT_1_te"] = self._apply_te(
            df["CAT_1"] if "CAT_1" in df else pd.Series(pd.NA, index=df.index), *self.te_["CAT_1"]).to_numpy()
        cols["CAT_2_te"] = self._apply_te(
            df["CAT_2"] if "CAT_2" in df else pd.Series(pd.NA, index=df.index), *self.te_["CAT_2"]).to_numpy()

        c6 = (df["CAT_6"] if "CAT_6" in df else pd.Series(pd.NA, index=df.index)).astype("string")
        cols["is_segdup"] = c6.str.contains("segdup", case=False, na=False).astype(int).to_numpy()
        cols["AL_185_present"] = (df["AL_185"].notna().astype(int).to_numpy()
                                  if "AL_185" in df else np.zeros(len(df), dtype=int))

        aa1, aa2 = self._aa(df, "AA_1"), self._aa(df, "AA_2")
        cols["AA_1_te"] = self._apply_te(aa1, *self.te_["AA_1"]).to_numpy()
        cols["AA_2_te"] = self._apply_te(aa2, *self.te_["AA_2"]).to_numpy()
        for name, tbl in (("Delta_MW", AA_MW), ("Delta_PI", AA_PI),
                          ("Delta_HYDRO", AA_HYDRO), ("Delta_Charge", AA_CHARGE)):
            cols[name] = (aa2.map(tbl).astype(float) - aa1.map(tbl).astype(float)).to_numpy()

        cols["ek7_x_ek9"] = (self._num(df, "EK_7") * self._num(df, "EK_9")).to_numpy()
        cols["ek2_ek3"] = (self._num(df, "EK_2") - self._num(df, "EK_3")).to_numpy()

        cat345 = pd.concat(
            [(df[c] if c in df else pd.Series(pd.NA, index=df.index)).astype("string").fillna("")
             for c in ("CAT_3", "CAT_4", "CAT_5")], axis=1)
        joined = cat345.agg("".join, axis=1).str.upper()
        for nuc in ("A", "T", "C", "G"):
            cols[nuc] = joined.str.contains(nuc, regex=False).astype(int).to_numpy()
        cat345_raw = pd.concat(
            [(df[c] if c in df else pd.Series(pd.NA, index=df.index))
             for c in ("CAT_3", "CAT_4", "CAT_5")], axis=1)
        cols["has_archaic_delta"] = cat345_raw.notna().any(axis=1).astype(int).to_numpy()

        out = pd.DataFrame(cols, index=df.index)
        # inf -> NaN (ör. ek7_x_ek9 tasmasi); downstream imputer temizler
        out = out.replace([np.inf, -np.inf], np.nan)

        if self.feature_names_ is not None:
            out = out.reindex(columns=self.feature_names_)
        return out

"""
src/preprocessing/cleaner.py
=============================
UNSW-NB15 data cleaner and feature encoder.

Responsibilities
----------------
- Drop irrelevant columns (IPs, timestamps)
- Coerce mixed-type columns to numeric
- Impute missing values
- Clip outliers via IQR-based bounds (fit on train only)
- Label-encode categorical features (proto, state, service)
- Z-score scale numeric features (fit on train only)

Design decisions
----------------
Categorical encoding: LabelEncoding for proto, state, service. Cardinality
is small (~5–50) and tree-based models handle integer codes natively without
the dimensionality cost of one-hot encoding.

IP addresses (srcip, dstip): dropped. IPs encode specific hosts rather than
traffic behaviour patterns, causing models to memorise rather than generalise.

Timestamps (Stime, Ltime): dropped. They encode session order, creating
temporal leakage under a random train/test split.

Outlier clipping: IQR × 3 (conservative). Removal would discard the extreme
values most likely to represent anomalous traffic. Clipping preserves the row
while bounding feature influence.

StandardScaler: fit on training data only, then applied at inference. Fitting
on the full dataset would constitute data leakage.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

DROP_COLS = [
    "srcip", "dstip",           # IP addresses — not generalisable
    "Stime", "Ltime",           # timestamps — temporal leakage risk
    "_source_partition",        # loader metadata
    "id",                       # Kaggle row ID
]

# Categorical columns encoded with LabelEncoder.
# sport and dsport are numeric port numbers handled by the loader.
CAT_FEATURES = ["proto", "state", "service"]

LABEL_COL      = "label"
ATTACK_CAT_COL = "attack_cat"

# Variant spellings present in the raw CSV partitions.
# Applied after str.strip().str.title() so only canonical-form mismatches
# need to be listed.
ATTACK_CAT_ALIASES: dict[str, str] = {
    "Backdoor":        "Backdoors",    # singular → plural (partition 3/4)
    "Dos":             "DoS",          # str.title() downcases the acronym
    "Reconnaissance ": "Reconnaissance",
    "Shellcode ":      "Shellcode",
    "Fuzzers ":        "Fuzzers",
    "Analysis ":       "Analysis",
    "Generic ":        "Generic",
    "Exploits ":       "Exploits",
    "Worms ":          "Worms",
    "Normal ":         "Normal",
    "Nan":             "Normal",       # stringified NaN → Normal
    "":                "Normal",
}

# Columns permitted to remain as object dtype after all transformations.
_ALLOWED_OBJECT_COLS = {ATTACK_CAT_COL}


class DataCleaner:
    """
    Stateful feature cleaner: fit_transform() on training data, transform()
    on test / inference data. Fitted state (encoders, scaler, clip bounds)
    is persisted to disk for use in the inference pipeline.
    """

    def __init__(self, clip_iqr_multiplier: float = 3.0):
        self.clip_iqr_multiplier = clip_iqr_multiplier
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.attack_cat_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self._fitted = False

    def _drop_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in DROP_COLS if c in df.columns]
        if to_drop:
            logger.info(f"Dropping columns: {to_drop}")
        return df.drop(columns=to_drop, errors="ignore")

    @staticmethod
    def _normalize_attack_cat(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise attack_cat to the 10 canonical category names.

        Pipeline: cast to str → strip whitespace → title-case →
        apply alias map for known variant spellings.
        """
        if ATTACK_CAT_COL not in df.columns:
            return df

        df = df.copy()
        series = df[ATTACK_CAT_COL].astype(str).str.strip().str.title()
        series = series.replace(ATTACK_CAT_ALIASES)

        from .loader import ATTACK_CATEGORIES
        unknown_cats = set(series.unique()) - set(ATTACK_CATEGORIES)
        if unknown_cats:
            logger.warning(
                f"attack_cat: {len(unknown_cats)} unrecognised categories after "
                f"normalisation: {unknown_cats}. "
                "Add to ATTACK_CAT_ALIASES if needed."
            )

        df[ATTACK_CAT_COL] = series
        return df

    @staticmethod
    def _coerce_remaining_numerics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce any non-categorical object-dtype column to float64.

        Catches columns that survived drop_cols still typed as object
        (e.g., sport/dsport with stray string values, or corrupted rows).
        Non-convertible cells become NaN and are filled in the next step.
        """
        known_cat = set(CAT_FEATURES) | _ALLOWED_OBJECT_COLS
        df = df.copy()

        for col in df.select_dtypes(include=["object", "str"]).columns:
            if col in known_cat:
                continue
            original_dtype = df[col].dtype
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_coerced_to_nan = coerced.isna().sum() - df[col].isna().sum()
            if n_coerced_to_nan > 0:
                sample_bad = df.loc[coerced.isna() & df[col].notna(), col].unique()[:5]
                logger.info(
                    f"Column '{col}' (dtype={original_dtype}): "
                    f"coerced {n_coerced_to_nan:,} non-numeric values to NaN "
                    f"(sample: {sample_bad})"
                )
            df[col] = coerced
        return df

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values.

        Numeric columns → fill with 0 (network counters default to 0 when
        absent, e.g., ct_ftp_cmd is 0 for non-FTP flows).
        Categorical columns → fill with 'unknown'.
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        null_total = df.isnull().sum().sum()
        if null_total > 0:
            logger.info(f"Imputing {null_total} missing values")

        df[num_cols] = df[num_cols].fillna(0)
        df[cat_cols] = df[cat_cols].fillna("unknown")
        return df

    def _clip_outliers(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        num_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        IQR-based outlier clipping.

        When fit=True, compute bounds from df and store them.
        When fit=False, apply stored bounds (test/inference path).
        Binary and target columns are excluded from clipping.
        """
        df = df.copy()
        if num_cols is None:
            num_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in [LABEL_COL, "attack_cat_encoded", "is_sm_ips_ports",
                             "is_ftp_login", "_source_partition"]
            ]

        if fit:
            self._clip_bounds = {}
            for col in num_cols:
                if col not in df.columns:
                    continue
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                self._clip_bounds[col] = (
                    q1 - self.clip_iqr_multiplier * iqr,
                    q3 + self.clip_iqr_multiplier * iqr,
                )

        for col, (lo, hi) in getattr(self, "_clip_bounds", {}).items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """
        Label-encode proto, state, service and optionally attack_cat.

        Unseen values at inference time are mapped to 'unknown', which was
        included in the LabelEncoder's training vocabulary.
        """
        df = df.copy()

        for col in CAT_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                all_values = list(df[col].astype(str).unique()) + ["unknown"]
                le.fit(all_values)
                self.label_encoders[col] = le

            le = self.label_encoders.get(col)
            if le:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else "unknown"
                )
                df[col] = le.transform(df[col])

        if ATTACK_CAT_COL in df.columns:
            if fit:
                all_cats = list(df[ATTACK_CAT_COL].unique()) + ["unknown"]
                self.attack_cat_encoder.fit(all_cats)

            known = set(self.attack_cat_encoder.classes_)
            df[ATTACK_CAT_COL] = df[ATTACK_CAT_COL].apply(
                lambda x: x if x in known else "unknown"
            )
            df["attack_cat_encoded"] = self.attack_cat_encoder.transform(
                df[ATTACK_CAT_COL]
            )

        return df

    def _scale_features(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """
        StandardScaler (zero mean, unit variance).

        Scaler is fitted on training data only and reused at inference.
        Target columns (label, attack_cat, attack_cat_encoded) are excluded.
        """
        df = df.copy()
        exclude = {
            LABEL_COL, ATTACK_CAT_COL, "attack_cat_encoded", "_source_partition"
        }
        feat_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        if fit:
            self.feature_cols = feat_cols
            self.scaler.fit(df[feat_cols])

        df[self.feature_cols] = self.scaler.transform(
            df[[c for c in self.feature_cols if c in df.columns]]
        )
        return df

    @staticmethod
    def _audit_object_columns(df: pd.DataFrame, stage: str = "") -> None:
        """
        Warn about any object-dtype columns outside the allowed set.

        Called before Parquet serialisation so dtype regressions surface as
        clear log messages rather than cryptic ArrowTypeError exceptions.
        """
        unexpected = [
            col for col in df.select_dtypes(include=["object", "str"]).columns
            if col not in _ALLOWED_OBJECT_COLS
        ]
        tag = f"[{stage}]" if stage else ""
        if unexpected:
            for col in unexpected:
                sample = df[col].dropna().unique()[:5]
                logger.warning(
                    f"Pre-save audit {tag}: column '{col}' is still dtype=object "
                    f"— this will break Parquet export. Sample values: {sample}"
                )
        else:
            logger.info(f"Pre-save audit {tag}: no unexpected object columns")

    # ── Public API ──────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return cleaned, encoded DataFrame."""
        df = self._drop_cols(df)
        df = self._normalize_attack_cat(df)
        df = self._coerce_remaining_numerics(df)
        df = self._handle_missing(df)
        df = self._clip_outliers(df, fit=True)
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_features(df, fit=True)
        self._audit_object_columns(df, stage="train")
        self._fitted = True
        logger.info(f"fit_transform complete. Shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted pipeline to test or inference data."""
        if not self._fitted:
            raise RuntimeError(
                "DataCleaner must be fitted before transform. Call fit_transform() first."
            )
        df = self._drop_cols(df)
        df = self._normalize_attack_cat(df)
        df = self._coerce_remaining_numerics(df)
        df = self._handle_missing(df)
        df = self._clip_outliers(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_features(df, fit=False)
        self._audit_object_columns(df, stage="test/infer")
        return df

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist fitted state (encoders, scaler, clip bounds) to disk."""
        path = path or (PROCESSED_DIR / "cleaner.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"DataCleaner saved to {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DataCleaner":
        """Load a previously saved DataCleaner from disk."""
        path = path or (PROCESSED_DIR / "cleaner.joblib")
        obj = joblib.load(path)
        logger.info(f"DataCleaner loaded from {path}")
        return obj

"""
UNSW-NB15 Data Cleaner & Feature Encoder
==========================================
Responsibilities:
  - Drop or impute missing values
  - Remove/flag constant and near-constant columns
  - Encode categorical features (proto, state, service, attack_cat)
  - Cap extreme outliers via IQR-based clipping
  - Normalize/standardize numeric features (fit on train only → transform test)

Key design decisions (defend these in your viva):
  1. Categorical encoding: Label Encoding for ordinal-like fields (proto, state);
     the cardinality is small and tree-based models (RF, XGBoost) handle integer
     codes natively without inflating dimensionality the way one-hot encoding does.
     For the LSTM-Autoencoder, we use the same encoding — the embedding is learned
     inside the model architecture anyway.

  2. IP address columns (srcip, dstip): We DROP these. IPs are not a generalizable
     feature — they'd cause the model to memorize specific hosts rather than
     learning traffic behavior patterns. This is standard in intrusion detection
     literature (e.g., Moustafa & Slay 2015, Kitsune 2018).

  3. Timestamps (Stime, Ltime): Also dropped. They encode session order, not
     traffic behavior; including them leaks temporal ordering into features,
     which would be a data leakage issue in our random train/test split.

  4. Outlier clipping: We use IQR × 3 (a conservative multiplier) rather than
     removal. Removing rows with extreme values in network traffic data is risky
     because extreme values are often exactly the anomalous events we want to detect.
     Clipping preserves the row while bounding the feature's influence.

  5. StandardScaler is fit ONLY on the training set and then applied to test/
     inference data. This is critical — fitting on the full dataset would be
     data leakage.

Bug-fixes & additions in this version:
  A. _coerce_remaining_numerics() (new, Step 2b):
       After dropping columns but before missing-value imputation, any column
       that *should* be numeric but landed as dtype=object (due to a stray
       string in one row) is coerced with pd.to_numeric(..., errors='coerce').
       Non-convertible cells become NaN and are then filled to 0 in Step 3.
       This is the defensive partner to the port-column fix in loader.py —
       together they guarantee no object-dtype column reaches Parquet.

  B. _normalize_attack_cat() (extended):
       Added an explicit alias map so that known variant spellings in the raw
       dataset are collapsed to the canonical ATTACK_CATEGORIES names:
         'Backdoor'          → 'Backdoors'
         'Dos'               → 'DoS'
         'Shellcode '        → 'Shellcode'  (trailing space, handled by strip)
         'Fuzzers '          → 'Fuzzers'    (ditto)
         etc.
       The alias map is applied AFTER str.strip().str.title() so we only need
       to enumerate canonical-form mismatches, not every whitespace variant.

  C. _audit_object_columns() (new, called from fit_transform / transform):
       Prints a warning listing any object-dtype columns that remain just before
       Parquet serialisation would happen.  This makes future dtype regressions
       immediately visible in the logs rather than as a cryptic ArrowTypeError.
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

# ─── Columns to drop before feature engineering ────────────────────────────────
# Rationale documented in module docstring
DROP_COLS = [
    "srcip", "dstip",           # IP addresses — not generalizable
    "Stime", "Ltime",           # timestamps — temporal leakage
    "_source_partition",        # loader metadata column
    "id",                       # Kaggle version row ID
]

# ─── Categorical columns to encode ─────────────────────────────────────────────
# These 3 are the only non-numeric network-attribute columns we keep.
# sport and dsport are NOT listed here — they are numeric port numbers and are
# handled by the port coercion in loader.py + _coerce_remaining_numerics() below.
CAT_FEATURES = ["proto", "state", "service"]

# ─── Target columns ────────────────────────────────────────────────────────────
LABEL_COL = "label"            # binary: 0=Normal, 1=Attack
ATTACK_CAT_COL = "attack_cat"  # multiclass attack category

# ─── Attack category alias map ──────────────────────────────────────────────────
# The raw UNSW-NB15 CSVs contain known variant spellings.
# This map is applied AFTER str.strip().str.title() normalisation,
# so we only need to handle mismatches in canonical title-cased form.
#
# Why 'Backdoor' → 'Backdoors'?
#   The dataset paper uses 'Backdoors' (plural). Some partitions use 'Backdoor'
#   (singular). Both refer to the same attack category. Keeping both would create
#   a phantom 11th class and break the MITRE ATT&CK mapping table.
#
# Why 'Dos' → 'DoS'?
#   str.title() lowercases the 'o' and 'S', producing 'Dos'. The canonical name
#   is 'DoS' (Denial of Service acronym), so we re-map after title-casing.
ATTACK_CAT_ALIASES: dict[str, str] = {
    "Backdoor":        "Backdoors",    # singular → plural (partition 3/4)
    "Dos":             "DoS",          # title-case artefact of the acronym
    "Reconnaissance ": "Reconnaissance",  # extra trailing space (belt + braces)
    "Shellcode ":      "Shellcode",
    "Fuzzers ":        "Fuzzers",
    "Analysis ":       "Analysis",
    "Generic ":        "Generic",
    "Exploits ":       "Exploits",
    "Worms ":          "Worms",
    "Normal ":         "Normal",
    "Nan":             "Normal",       # NaN stringified → treat as Normal
    "":                "Normal",       # empty string → Normal
}

# Columns that legitimately stay as object dtype after all transformations
# (they hold human-readable strings preserved for the dashboard/reports).
_ALLOWED_OBJECT_COLS = {ATTACK_CAT_COL}


class DataCleaner:
    """
    Stateful cleaner: fit_transform on train set, transform on test/inference.
    State (encoders, scaler) is saved to disk for use in the inference pipeline.
    """

    def __init__(self, clip_iqr_multiplier: float = 3.0):
        self.clip_iqr_multiplier = clip_iqr_multiplier
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.attack_cat_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []   # numeric feature columns after cleaning
        self._fitted = False

    # ── Step 1: Drop irrelevant columns ────────────────────────────────────────
    def _drop_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in DROP_COLS if c in df.columns]
        if to_drop:
            logger.info(f"Dropping columns: {to_drop}")
        return df.drop(columns=to_drop, errors="ignore")

    # ── Step 2a: Normalize attack_cat string values ─────────────────────────────
    @staticmethod
    def _normalize_attack_cat(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the attack_cat column to the 10 canonical category names.

        Pipeline:
          1. Cast to str (handles NaN → 'nan')
          2. Strip leading/trailing whitespace
          3. Apply str.title() for consistent casing
          4. Apply ATTACK_CAT_ALIASES to fix known variant spellings
             (e.g. 'Backdoor' → 'Backdoors', 'Dos' → 'DoS')

        Why str.title() first, then alias map?
          str.title() turns ' fuzzers ' → 'Fuzzers', so the alias map only
          needs to handle post-title-case mismatches, not the full combinatorial
          space of whitespace × casing variants.
        """
        if ATTACK_CAT_COL not in df.columns:
            return df

        df = df.copy()
        series = df[ATTACK_CAT_COL].astype(str).str.strip().str.title()
        series = series.replace(ATTACK_CAT_ALIASES)

        # Log any categories that are STILL not in the canonical set (sanity check)
        from .loader import ATTACK_CATEGORIES
        canonical_set = set(ATTACK_CATEGORIES)
        unknown_cats = set(series.unique()) - canonical_set
        if unknown_cats:
            logger.warning(
                f"attack_cat: {len(unknown_cats)} unrecognised categories found "
                f"after normalisation: {unknown_cats}. "
                "They will be encoded as-is; add them to ATTACK_CAT_ALIASES if needed."
            )

        df[ATTACK_CAT_COL] = series
        return df

    # ── Step 2b: Coerce any remaining object columns to numeric ─────────────────
    @staticmethod
    def _coerce_remaining_numerics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Safety net: any column that is NOT in the known-categorical list and NOT
        in the allowed-object set, but still has dtype=object, is coerced to
        float64 via pd.to_numeric(..., errors='coerce').

        Why here (after drop_cols, before handle_missing)?
          - drop_cols has already removed IPs and timestamps.
          - handle_missing (Step 3) uses select_dtypes to split numeric/categorical;
            if a column is still object here, it would be incorrectly imputed with
            the string 'unknown' rather than 0.
          - Coercing before imputation means the 0-fill logic applies correctly.

        Columns this catches in practice:
          - sport / dsport: belt-and-braces in case loader.py's coercion was
            bypassed (e.g. user constructs a DataFrame manually in a test).
          - Any other column where a stray header row, comment, or corrupted
            value was parsed as a string by pandas.
        """
        known_cat = set(CAT_FEATURES) | _ALLOWED_OBJECT_COLS
        df = df.copy()

        for col in df.select_dtypes(include=["object", "str"]).columns:
            if col in known_cat:
                continue  # leave legitimate string columns alone
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

    # ── Step 3: Handle missing values ─────────────────────────────────────────
    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy: fill numeric NaN with 0 (network features default to 0 when
        absent, e.g., 'ct_ftp_cmd' is 0 for non-FTP flows).
        Fill categorical NaN with the string 'unknown'.
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.info(f"Imputing {null_counts.sum()} missing values")

        df[num_cols] = df[num_cols].fillna(0)
        df[cat_cols] = df[cat_cols].fillna("unknown")
        return df

    # ── Step 4: Clip outliers ─────────────────────────────────────────────────
    def _clip_outliers(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        num_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        IQR-based clipping. When fit=True, compute bounds from df and store them.
        When fit=False, use stored bounds (for test/inference).
        """
        df = df.copy()
        if num_cols is None:
            # Exclude binary and target columns from clipping
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

    # ── Step 5: Encode categoricals ────────────────────────────────────────────
    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        df = df.copy()

        for col in CAT_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                # Add 'unknown' to handle unseen values at inference time
                all_values = list(df[col].astype(str).unique()) + ["unknown"]
                le.fit(all_values)
                self.label_encoders[col] = le

            le = self.label_encoders.get(col)
            if le:
                # Map unseen values to 'unknown' (safe inference)
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else "unknown"
                )
                df[col] = le.transform(df[col])

        # Encode attack_cat → integer (for supervised models)
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

    # ── Step 6: Scale numeric features ────────────────────────────────────────
    def _scale_features(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """
        StandardScaler (zero mean, unit variance). Applied after encoding.
        Scaler is fit on train only and reused at inference.
        """
        df = df.copy()
        # Feature columns = everything except targets and metadata
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

    # ── Pre-save audit ─────────────────────────────────────────────────────────
    @staticmethod
    def _audit_object_columns(df: pd.DataFrame, stage: str = "") -> None:
        """
        Log a warning for any object-dtype column outside the allowed set.
        Call this just before saving to Parquet so problems surface as clear
        log messages rather than as a cryptic ArrowTypeError deep in PyArrow.

        Expected output if everything is correct:
            INFO  Pre-save audit [train]: no unexpected object columns ✓
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
                    f"— this will break Parquet export! Sample values: {sample}"
                )
        else:
            logger.info(f"Pre-save audit {tag}: no unexpected object columns ✓")

    # ── Public API ─────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: fit on train data, return cleaned + encoded DataFrame."""
        df = self._drop_cols(df)
        df = self._normalize_attack_cat(df)
        df = self._coerce_remaining_numerics(df)   # ← NEW: safety net step
        df = self._handle_missing(df)
        df = self._clip_outliers(df, fit=True)
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_features(df, fit=True)
        self._audit_object_columns(df, stage="train")  # ← NEW: pre-save check
        self._fitted = True
        logger.info(f"fit_transform complete. Shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted pipeline to new data (test / inference)."""
        if not self._fitted:
            raise RuntimeError(
                "DataCleaner must be fit before transform. Call fit_transform() first."
            )
        df = self._drop_cols(df)
        df = self._normalize_attack_cat(df)
        df = self._coerce_remaining_numerics(df)   # ← NEW: safety net step
        df = self._handle_missing(df)
        df = self._clip_outliers(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_features(df, fit=False)
        self._audit_object_columns(df, stage="test/infer")  # ← NEW: pre-save check
        return df

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist encoder + scaler state for inference pipeline."""
        path = path or (PROCESSED_DIR / "cleaner.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"DataCleaner saved to {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DataCleaner":
        path = path or (PROCESSED_DIR / "cleaner.joblib")
        obj = joblib.load(path)
        logger.info(f"DataCleaner loaded from {path}")
        return obj

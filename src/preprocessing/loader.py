"""
UNSW-NB15 Data Loader
======================
Responsibilities:
  - Load one or more raw CSV partitions (or Kaggle pre-split train/test CSVs)
  - Assign canonical column names from the features CSV
  - Validate schema and basic data integrity
  - Expose a clean DataFrame for the preprocessing pipeline

Design note:
  We keep loading separate from cleaning so each step is independently testable
  and re-runnable (important for reproducibility in academic projects).

Bug-fix note (sport / dsport mixed types):
  The raw UNSW-NB15 CSVs contain mixed values in sport and dsport:
    - Decimal integers (e.g. 80)
    - Hex strings  (e.g. '0x0050')
    - Service names (e.g. 'http', 'ftp-data')
    - Empty / NaN cells
  Because of this mix, pandas reads both columns as dtype=object, which
  PyArrow refuses to serialise to Parquet.

  Fix applied here (in _coerce_port_columns):
    1. Hex strings → int via base-16 parsing
    2. Everything else → pd.to_numeric(..., errors='coerce') → NaN for
       non-convertible strings
    3. NaN → 0  (port 0 is the conventional "unknown" sentinel in UNSW-NB15)
    4. Cast to int64 so the column has a definite numeric dtype

  Doing this in the loader (rather than the cleaner) keeps it close to the
  source of the problem and means validate_schema() can report the corrected
  dtypes immediately.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─── Column definitions ─────────────────────────────────────────────────────────
# UNSW-NB15 has 49 features. The raw CSVs have NO header row in some versions.
# This is the canonical ordered feature list from the dataset paper:
#   Moustafa & Slay (2015), "UNSW-NB15: a comprehensive data set for
#   network intrusion detection systems"

COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto",           # 1–5
    "state", "dur", "sbytes", "dbytes", "sttl",              # 6–10
    "dttl", "sloss", "dloss", "service", "Sload",            # 11–15
    "Dload", "Spkts", "Dpkts", "swin", "dwin",               # 16–20
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",   # 21–25
    "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime",         # 26–30
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",      # 31–35
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",   # 36–38
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",              # 39–41
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",                # 42–44
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", # 45–47
    "attack_cat",                                              # 48  ← label (category)
    "label",                                                   # 49  ← binary (0=normal, 1=attack)
]

# Attack categories present in the dataset (canonical, normalised forms)
ATTACK_CATEGORIES = [
    "Normal",
    "Fuzzers",
    "Analysis",
    "Backdoors",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

# Columns whose values may be hex strings or service names in the raw CSVs.
# These must be coerced to int before Parquet serialisation.
PORT_COLUMNS = ["sport", "dsport"]

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


# ─── Port coercion helper ────────────────────────────────────────────────────────

def _parse_port_value(val) -> int:
    """
    Parse a single port value that may be:
      - An int already          → return as-is
      - A hex string '0x0050'   → parse with base 16
      - A decimal string '80'   → parse normally
      - A service name 'http'   → return 0 (unmappable; treated as unknown)
      - NaN / None              → return 0

    Why return 0 for service names rather than a lookup table?
    The cleaner already drops these columns' raw string content in favour of
    the encoded 'service' feature column, so port numbers are used only as
    numeric signals (high port → ephemeral, low port → well-known service).
    Mapping 'http' → 80 would be redundant with the 'service' column and
    would add complexity without measurable benefit to the models.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "-", ""):
        return 0
    try:
        # Try hex first (e.g. '0x0050', '0X50')
        if s.lower().startswith("0x"):
            return int(s, 16)
        # Try plain integer / float-as-string (e.g. '80', '80.0')
        return int(float(s))
    except (ValueError, OverflowError):
        # Non-numeric service name (e.g. 'http', 'ftp-data') → 0
        return 0


def _coerce_port_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply _parse_port_value to sport and dsport and cast to int64.

    This is the root-cause fix for the Parquet serialisation failure.
    PyArrow requires columns to have a single unambiguous dtype; an object
    column containing both Python ints and strings raises ArrowTypeError.
    After this function both port columns are dtype=int64.
    """
    df = df.copy()
    for col in PORT_COLUMNS:
        if col not in df.columns:
            continue
        # Fast path: already numeric (happens for Kaggle pre-split CSVs)
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].fillna(0).astype(np.int64)
            continue
        # Slow path: object dtype – apply element-wise parser
        original_non_numeric = df[col].apply(
            lambda x: not isinstance(x, (int, float, np.integer, np.floating))
                      and str(x).strip().lower() not in ("nan", "none", "")
                      and not str(x).strip().startswith("0x")
        )
        n_non_numeric = original_non_numeric.sum()
        if n_non_numeric > 0:
            sample = df.loc[original_non_numeric, col].unique()[:8]
            logger.info(
                f"Column '{col}': coercing {n_non_numeric:,} non-numeric values "
                f"to 0 (sample: {sample})"
            )
        df[col] = df[col].apply(_parse_port_value).astype(np.int64)
    return df


# ─── Loaders ────────────────────────────────────────────────────────────────────

def load_raw_partitions(
    raw_dir: Optional[Path] = None,
    partitions: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Load and concatenate UNSW-NB15 raw CSV partitions (1–4).

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing UNSW-NB15_*.csv files. Defaults to data/raw/.
    partitions : list[int], optional
        Which partition numbers to load (1–4). Defaults to all four.

    Returns
    -------
    pd.DataFrame with canonical column names and coerced port columns,
    shape ~ (2.5M, 49+1).
    """
    raw_dir = raw_dir or RAW_DIR
    partitions = partitions or [1, 2, 3, 4]

    dfs = []
    for i in partitions:
        path = raw_dir / f"UNSW-NB15_{i}.csv"
        if not path.exists():
            logger.warning(f"Partition {i} not found at {path}. Skipping.")
            continue

        logger.info(f"Loading partition {i}: {path}")
        # The raw files have no header; we assign COLUMN_NAMES explicitly.
        # low_memory=False avoids spurious mixed-type warnings on object columns.
        df = pd.read_csv(
            path,
            header=None,
            names=COLUMN_NAMES,
            low_memory=False,
            on_bad_lines="warn",
        )
        df["_source_partition"] = i

        # ── Fix: coerce port columns immediately after loading ─────────────────
        df = _coerce_port_columns(df)

        dfs.append(df)
        logger.info(f"  Loaded {len(df):,} rows")

    if not dfs:
        raise FileNotFoundError(
            f"No UNSW-NB15 partition CSVs found in {raw_dir}.\n"
            "Run: python data/download_data.py"
        )

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset: {combined.shape[0]:,} rows × {combined.shape[1]} cols")
    return combined


def load_kaggle_splits(raw_dir: Optional[Path] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-split train/test CSVs from the Kaggle version of UNSW-NB15.
    The Kaggle version already has headers and uses slightly different column casing.

    Returns
    -------
    (train_df, test_df) — both with columns normalized to COLUMN_NAMES casing
    and port columns coerced to int64.
    """
    raw_dir = raw_dir or RAW_DIR
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    missing = [p for p in [train_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Kaggle split files not found: {missing}\n"
            "Download from: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
        )

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    # Normalize column names: lowercase + strip whitespace
    for df in [train_df, test_df]:
        df.columns = df.columns.str.strip().str.lower()

    # ── Fix: coerce port columns for Kaggle format too ─────────────────────────
    train_df = _coerce_port_columns(train_df)
    test_df = _coerce_port_columns(test_df)

    logger.info(f"Kaggle train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df


def auto_load(raw_dir: Optional[Path] = None) -> Union[pd.DataFrame, tuple]:
    """
    Convenience function: detects which file format is present and loads accordingly.
    Priority: Kaggle pre-split → raw partitions.

    Returns
    -------
    If Kaggle format detected: (train_df, test_df)
    If raw partitions detected: combined_df (you'll split later in preprocessing)
    """
    raw_dir = raw_dir or RAW_DIR

    kaggle_train = raw_dir / "UNSW_NB15_training-set.csv"
    if kaggle_train.exists():
        logger.info("Detected Kaggle pre-split format.")
        return load_kaggle_splits(raw_dir)

    partition_1 = raw_dir / "UNSW-NB15_1.csv"
    if partition_1.exists():
        logger.info("Detected raw partition format.")
        return load_raw_partitions(raw_dir)

    raise FileNotFoundError(
        "No UNSW-NB15 data found in data/raw/.\n"
        "Run: python data/download_data.py"
    )


# ─── Schema validation ───────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> dict:
    """
    Quick integrity check on a loaded DataFrame.
    Returns a dict of validation results for inspection.

    Now includes an 'object_columns' entry that lists any remaining object-dtype
    columns and their sample values — used to catch future mixed-type surprises
    before they reach Parquet serialisation.
    """
    # Find object columns that are NOT the known string targets
    allowed_object_cols = {"attack_cat", "proto", "state", "service"}
    object_cols = {
        col: df[col].dropna().unique()[:5].tolist()
        for col in df.select_dtypes(include="object").columns
        if col not in allowed_object_cols
    }

    results = {
        "shape": df.shape,
        "missing_columns": [c for c in COLUMN_NAMES if c not in df.columns],
        "null_counts": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "object_columns_outside_expected": object_cols,   # ← new: debug audit
        "attack_cat_distribution": (
            df["attack_cat"].value_counts().to_dict()
            if "attack_cat" in df.columns
            else "N/A"
        ),
        "label_distribution": (
            df["label"].value_counts().to_dict()
            if "label" in df.columns
            else "N/A"
        ),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("UNSW-NB15 Loader — Quick Validation")
    print("=" * 60)

    data = auto_load()

    if isinstance(data, tuple):
        train_df, test_df = data
        print("\n[Train set]")
        for k, v in validate_schema(train_df).items():
            print(f"  {k}: {v}")
        print("\n[Test set]")
        for k, v in validate_schema(test_df).items():
            print(f"  {k}: {v}")
    else:
        df = data
        print("\n[Combined dataset]")
        for k, v in validate_schema(df).items():
            print(f"  {k}: {v}")

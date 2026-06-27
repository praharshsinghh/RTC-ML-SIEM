"""
src/preprocessing/loader.py
============================
UNSW-NB15 data loader.

Responsibilities
----------------
- Load raw CSV partitions (1–4) or Kaggle pre-split train/test CSVs
- Assign canonical column names from the dataset paper
- Coerce mixed-type port columns (sport, dsport) to int64
- Validate schema and basic data integrity

The raw UNSW-NB15 CSVs contain mixed values in sport and dsport:
  - Decimal integers (e.g., 80)
  - Hex strings (e.g., '0x0050')
  - Service names (e.g., 'http', 'ftp-data')
  - Empty / NaN cells

pandas reads both columns as object dtype, which PyArrow rejects during
Parquet serialisation. _coerce_port_columns() normalises them to int64
immediately after loading, before any other processing.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Column definitions ──────────────────────────────────────────────────────────
# Canonical ordered feature list from Moustafa & Slay (2015).
# The raw CSV files have no header row in some versions.
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
    "attack_cat",                                              # 48
    "label",                                                   # 49
]

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

# Columns with mixed types in the raw CSVs that must be coerced to int64.
PORT_COLUMNS = ["sport", "dsport"]

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR  = DATA_DIR / "raw"


# ── Port coercion ───────────────────────────────────────────────────────────────

def _parse_port_value(val) -> int:
    """
    Parse a single port value that may be an integer, hex string, decimal
    string, service name, or NaN.

    Service names (e.g., 'http') map to 0 rather than a port-number lookup
    table, because port numbers are redundant with the encoded 'service'
    feature column.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "-", ""):
        return 0
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(float(s))
    except (ValueError, OverflowError):
        return 0


def _coerce_port_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise sport and dsport to int64.

    Fast path: numeric dtype → fill NaN with 0, cast.
    Slow path: object dtype → apply _parse_port_value element-wise.
    """
    df = df.copy()
    for col in PORT_COLUMNS:
        if col not in df.columns:
            continue
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].fillna(0).astype(np.int64)
            continue
        # Object dtype — apply element-wise parser
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


# ── Loaders ─────────────────────────────────────────────────────────────────────

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
        Partition numbers to load (1–4). Defaults to all four.

    Returns
    -------
    pd.DataFrame with canonical column names and coerced port columns,
    shape approximately (2.5M, 50).
    """
    raw_dir    = raw_dir    or RAW_DIR
    partitions = partitions or [1, 2, 3, 4]

    dfs = []
    for i in partitions:
        path = raw_dir / f"UNSW-NB15_{i}.csv"
        if not path.exists():
            logger.warning(f"Partition {i} not found at {path}. Skipping.")
            continue

        logger.info(f"Loading partition {i}: {path}")
        df = pd.read_csv(
            path,
            header=None,
            names=COLUMN_NAMES,
            low_memory=False,
            on_bad_lines="warn",
        )
        df["_source_partition"] = i
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

    The Kaggle version includes column headers and uses slightly different
    casing. Columns are normalised to COLUMN_NAMES casing.

    Returns
    -------
    (train_df, test_df) — both with coerced port columns.
    """
    raw_dir    = raw_dir or RAW_DIR
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path  = raw_dir / "UNSW_NB15_testing-set.csv"

    missing = [p for p in [train_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Kaggle split files not found: {missing}\n"
            "Download from: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
        )

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df  = pd.read_csv(test_path,  low_memory=False)

    for df in [train_df, test_df]:
        df.columns = df.columns.str.strip().str.lower()

    train_df = _coerce_port_columns(train_df)
    test_df  = _coerce_port_columns(test_df)

    logger.info(f"Kaggle train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df


def auto_load(raw_dir: Optional[Path] = None) -> Union[pd.DataFrame, tuple]:
    """
    Auto-detect the dataset format and load accordingly.

    Priority: Kaggle pre-split → raw partitions.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]  if Kaggle format detected
    pd.DataFrame                        if raw partition format detected
    """
    raw_dir = raw_dir or RAW_DIR

    if (raw_dir / "UNSW_NB15_training-set.csv").exists():
        logger.info("Detected Kaggle pre-split format.")
        return load_kaggle_splits(raw_dir)

    if (raw_dir / "UNSW-NB15_1.csv").exists():
        logger.info("Detected raw partition format.")
        return load_raw_partitions(raw_dir)

    raise FileNotFoundError(
        "No UNSW-NB15 data found in data/raw/.\n"
        "Run: python data/download_data.py"
    )


# ── Schema validation ────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> dict:
    """
    Run a basic integrity check on a loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys: shape, missing_columns, null_counts,
    object_columns_outside_expected, attack_cat_distribution,
    label_distribution, duplicate_rows.
    """
    allowed_object_cols = {"attack_cat", "proto", "state", "service"}
    object_cols = {
        col: df[col].dropna().unique()[:5].tolist()
        for col in df.select_dtypes(include="object").columns
        if col not in allowed_object_cols
    }

    return {
        "shape": df.shape,
        "missing_columns": [c for c in COLUMN_NAMES if c not in df.columns],
        "null_counts": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "object_columns_outside_expected": object_cols,
        "attack_cat_distribution": (
            df["attack_cat"].value_counts().to_dict()
            if "attack_cat" in df.columns else "N/A"
        ),
        "label_distribution": (
            df["label"].value_counts().to_dict()
            if "label" in df.columns else "N/A"
        ),
        "duplicate_rows": int(df.duplicated().sum()),
    }


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
        print("\n[Combined dataset]")
        for k, v in validate_schema(data).items():
            print(f"  {k}: {v}")

"""
UNSW-NB15 Dataset Download Script
==================================
The UNSW-NB15 dataset is hosted by the University of New South Wales (UNSW)
Canberra Cyber Security Lab. It contains 2,540,044 records with 49 features
describing network traffic, split into 4 CSV partitions.

Official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Direct download: https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys

This script downloads the 4 raw CSV files + the feature names CSV.
If the direct links are unavailable (they sometimes require registration),
see the MANUAL DOWNLOAD section below.
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# UNSW-NB15 files hosted on UNSW's SharePoint/CloudStor
# NOTE: These URLs may require updating if UNSW rotates them.
# Check https://research.unsw.edu.au/projects/unsw-nb15-dataset for the latest.
FILES = {
    "UNSW-NB15_1.csv": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_4.csv",
    "NUSW-NB15_features.csv": "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=NUSW-NB15_features.csv",
}

# Expected approximate sizes in MB (for validation)
EXPECTED_SIZES_MB = {
    "UNSW-NB15_1.csv": 160,
    "UNSW-NB15_2.csv": 160,
    "UNSW-NB15_3.csv": 160,
    "UNSW-NB15_4.csv": 160,
    "NUSW-NB15_features.csv": 0.01,
}


def download_file(url: str, dest: Path, filename: str) -> bool:
    """Stream-download a file with a progress bar."""
    print(f"\n📥 Downloading {filename} ...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def check_existing(dest: Path, filename: str) -> bool:
    """Skip download if file already exists and has reasonable size."""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        expected = EXPECTED_SIZES_MB.get(filename, 0)
        if size_mb > max(expected * 0.5, 0.001):
            print(f"  ✅ {filename} already present ({size_mb:.1f} MB) — skipping.")
            return True
    return False


def main():
    print("=" * 60)
    print("UNSW-NB15 Dataset Downloader")
    print("=" * 60)
    print(f"Target directory: {RAW_DIR.resolve()}\n")

    failed = []
    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        if check_existing(dest, filename):
            continue
        success = download_file(url, dest, filename)
        if not success:
            failed.append(filename)

    print("\n" + "=" * 60)
    if failed:
        print("⚠️  Some downloads failed. See MANUAL DOWNLOAD instructions below.")
        print("Failed files:", failed)
        print_manual_instructions()
    else:
        print("✅  All files downloaded successfully!")
        print(f"    Location: {RAW_DIR.resolve()}")
        print("\nNext step: run `python src/preprocessing/loader.py` to merge and validate.")


def print_manual_instructions():
    """Printed if auto-download fails — UNSW sometimes requires registration."""
    print("""
──────────────────────────────────────────────────────────────
MANUAL DOWNLOAD INSTRUCTIONS
──────────────────────────────────────────────────────────────
1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Click "Dataset Files" → "UNSW-NB15 - CSV Files"
3. Download all 4 CSV partitions (UNSW-NB15_1.csv through _4.csv)
   and NUSW-NB15_features.csv
4. Place them in:  data/raw/

Alternative mirror (Kaggle — requires free account):
  https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  → Download and extract to data/raw/
  (The Kaggle version includes pre-split train/test CSVs which
   this project can also use — see loader.py for both paths.)
──────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()

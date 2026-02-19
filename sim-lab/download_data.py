#!/usr/bin/env python3
"""Download and extract the IBM Double Pendulum Chaotic Dataset."""

import os
import tarfile
import urllib.request
import sys

DATASET_URL = "https://dax-cdn.cdn.appdomain.cloud/dax-double-pendulum-chaotic/2.0.1/double-pendulum-chaotic.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ARCHIVE_PATH = os.path.join(DATA_DIR, "double-pendulum-chaotic.tar.gz")


def download_dataset():
    """Download and extract the dataset. Skip if already present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if already extracted — look for CSV files in expected location
    extracted_dirs = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "processed"
    ]
    if extracted_dirs:
        # Check if any CSVs exist inside
        for d in extracted_dirs:
            csvs = [f for f in os.listdir(os.path.join(DATA_DIR, d)) if f.endswith(".csv")]
            if csvs:
                print(f"Dataset already extracted in {os.path.join(DATA_DIR, d)} ({len(csvs)} CSV files). Skipping.")
                return

    # Download if archive not present
    if not os.path.exists(ARCHIVE_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        print(f"Saving to {ARCHIVE_PATH}")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / 1e6
                mb_total = total_size / 1e6
                print(f"\r  {pct:5.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH, reporthook=_progress)
        print("\nDownload complete.")
    else:
        print(f"Archive already exists at {ARCHIVE_PATH}. Skipping download.")

    # Extract
    print(f"Extracting {ARCHIVE_PATH}...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print("Extraction complete.")

    # Clean up archive to save space
    os.remove(ARCHIVE_PATH)
    print(f"Removed archive {ARCHIVE_PATH}.")

    # Report what we got
    for d in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, d)
        if os.path.isdir(full) and d != "processed":
            csvs = [f for f in os.listdir(full) if f.endswith(".csv")]
            print(f"Found {len(csvs)} CSV files in {full}")


if __name__ == "__main__":
    download_dataset()

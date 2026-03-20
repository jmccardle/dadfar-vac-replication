#!/usr/bin/env python3
"""Download all files from Zenodo record 18614770 into zenodo/data/ and zenodo/scripts/."""

import os
import urllib.request
from pathlib import Path

RECORD_ID = "18614770"
BASE_URL = f"https://zenodo.org/records/{RECORD_ID}/files"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "zenodo" / "data"
SCRIPTS_DIR = PROJECT_ROOT / "zenodo" / "scripts"

DATA_FILES = [
    "llama_baseline_n50.json",
    "llama_descriptive_control.json",
    "llama_layer_sweep_70b.json",
    "llama_overnight_battery.json",
    "llama_paired_n70.json",
    "qwen_baseline_n50.json",
    "qwen_descriptive_control.json",
]

SCRIPT_FILES = [
    "analyze_llama_desc_control.py",
    "analyze_paired_n50.py",
    "analyze_qwen_n50_all.py",
    "compute_fdr.py",
    "generate_fig2_steering.py",
    "generate_fig5_loop_autocorr.py",
    "generate_fig6_descriptive_control.py",
    "generate_fig7_shimmer_paired.py",
    "generate_fig9_qwen_control.py",
    "verify_qwen.py",
    "verify_tables.py",
]

OTHER_FILES = [
    ("README.md", PROJECT_ROOT / "zenodo"),
]


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading: {dest.name} ... ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_kb = dest.stat().st_size / 1024
    print(f"{size_kb:.1f} KB")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading data files...")
    for fname in DATA_FILES:
        url = f"{BASE_URL}/{fname}?download=1"
        download_file(url, DATA_DIR / fname)

    print("\nDownloading analysis scripts...")
    for fname in SCRIPT_FILES:
        url = f"{BASE_URL}/{fname}?download=1"
        download_file(url, SCRIPTS_DIR / fname)

    print("\nDownloading other files...")
    for fname, dest_dir in OTHER_FILES:
        url = f"{BASE_URL}/{fname}?download=1"
        download_file(url, dest_dir / fname)

    print("\nDone. Files downloaded to:")
    print(f"  Data:    {DATA_DIR}")
    print(f"  Scripts: {SCRIPTS_DIR}")


if __name__ == "__main__":
    main()

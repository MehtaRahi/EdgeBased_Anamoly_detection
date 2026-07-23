"""
Download NASA SMAP/MSL anomaly detection dataset.
Downloads individual .npy files from the OmniAnomaly GitHub mirror.
"""
import os
import urllib.request
import json
import ast
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/khundman/telemanom/master"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "SMAP"

def download_labels():
    """Download labeled_anomalies.csv from telemanom repo."""
    labels_path = DATA_DIR / "labeled_anomalies.csv"
    if labels_path.exists():
        print("[OK] Labels already exist")
        return
    
    url = f"{BASE_URL}/labeled_anomalies.csv"
    print(f"Downloading labels from {url}...")
    urllib.request.urlretrieve(url, str(labels_path))
    print("[OK] Labels downloaded")


def get_smap_channels():
    """Get list of SMAP channel IDs from labels."""
    labels_path = DATA_DIR / "labeled_anomalies.csv"
    df = pd.read_csv(labels_path)
    smap_channels = df[df["spacecraft"] == "SMAP"]["chan_id"].tolist()
    return smap_channels


def download_npy_files(channels, max_channels=10):
    """
    Download train/test .npy files for selected SMAP channels.
    We limit to max_channels to keep download size manageable.
    """
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    selected = channels[:max_channels]
    
    for i, chan in enumerate(selected):
        for split in ["train", "test"]:
            filename = f"{chan}.npy"
            local_path = DATA_DIR / split / filename
            
            if local_path.exists():
                continue
            
            url = f"{BASE_URL}/data/{split}/{filename}"
            try:
                urllib.request.urlretrieve(url, str(local_path))
                print(f"  [{i+1}/{len(selected)}] Downloaded {split}/{filename}")
            except Exception as e:
                print(f"  [WARN] Failed to download {split}/{filename}: {e}")
    
    print(f"[OK] Downloaded {len(selected)} channels")
    return selected


def main():
    print("=== NASA SMAP Dataset Downloader ===\n")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Step 1: Download labels
    download_labels()
    
    # Step 2: Get SMAP channel IDs
    channels = get_smap_channels()
    print(f"Found {len(channels)} SMAP channels")
    
    # Step 3: Download .npy files for top 10 channels
    selected = download_npy_files(channels, max_channels=10)
    
    # Step 4: Verify downloads
    print("\n=== Verification ===")
    for chan in selected:
        train_path = DATA_DIR / "train" / f"{chan}.npy"
        test_path = DATA_DIR / "test" / f"{chan}.npy"
        if train_path.exists() and test_path.exists():
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            print(f"  {chan}: train={train_data.shape}, test={test_data.shape}")
        else:
            print(f"  {chan}: MISSING")
    
    print("\n[DONE] SMAP dataset ready")


if __name__ == "__main__":
    main()

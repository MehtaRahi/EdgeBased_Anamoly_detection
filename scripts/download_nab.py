"""
Download Numenta Anomaly Benchmark (NAB) dataset.
Downloads specifically the 'realKnownCause' subset for benchmarking.
"""
import os
import urllib.request
import pandas as pd
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause"
LABEL_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "NAB"

FILES_TO_DOWNLOAD = [
    "ambient_temperature_system_failure.csv",
    "cpu_utilization_asg_misconfiguration.csv",
    "ec2_request_latency_system_failure.csv",
    "machine_temperature_system_failure.csv",
    "nyc_taxi.csv",
    "rogue_agent_key_hold.csv",
    "rogue_agent_key_updown.csv"
]

def download_nab():
    print("=== NAB Dataset Downloader ===\n")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download labels
    labels_path = DATA_DIR / "combined_labels.json"
    if not labels_path.exists():
        print(f"Downloading labels from {LABEL_URL}...")
        try:
            urllib.request.urlretrieve(LABEL_URL, str(labels_path))
            print("  [OK] Labels downloaded")
        except Exception as e:
            print(f"  [ERROR] Failed to download labels: {e}")
            return
    else:
        print("  [OK] Labels already exist")
        
    # Download data files
    print("\nDownloading data files...")
    for filename in FILES_TO_DOWNLOAD:
        file_path = DATA_DIR / filename
        if file_path.exists():
            print(f"  [OK] {filename} already exists")
            continue
            
        url = f"{BASE_URL}/{filename}"
        print(f"  Fetching {filename}...")
        try:
            urllib.request.urlretrieve(url, str(file_path))
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")
            
    print("\n[DONE] NAB dataset ready")

if __name__ == "__main__":
    download_nab()

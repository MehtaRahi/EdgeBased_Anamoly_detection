import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "NAB"

FILES = [
    "ambient_temperature_system_failure.csv",
    "cpu_utilization_asg_misconfiguration.csv",
    "ec2_request_latency_system_failure.csv",
    "machine_temperature_system_failure.csv",
    "nyc_taxi.csv",
    "rogue_agent_key_hold.csv",
    "rogue_agent_key_updown.csv"
]

def load_nab(seq_len=50, stride=2):
    """
    Load NAB datasets as a collection of univariate time series.
    Each file acts as a separate entity/client.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"NAB directory not found. Please run scripts/download_nab.py")
        
    # Load labels
    with open(DATA_DIR / "combined_labels.json", "r") as f:
        labels_dict = json.load(f)

    all_train, all_test, all_labels = [], [], []

    for filename in FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
            
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Scale values
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[["value"]].values).astype(np.float32)
        
        # Get labels
        # The labels_dict uses paths like "realKnownCause/ambient_temperature_system_failure.csv"
        dict_key = f"realKnownCause/{filename}"
        anomaly_windows = labels_dict.get(dict_key, [])
        
        y_raw = np.zeros(len(data), dtype=int)
        for timestamp_str in anomaly_windows:
            ts = pd.to_datetime(timestamp_str)
            # Find the closest index (NAB labels are points, we can treat a small window around it as anomaly or just exact match)
            # Usually we expand the label a bit since it's a point anomaly
            idx = (df["timestamp"] - ts).abs().idxmin()
            # Mark a small window around the anomaly (e.g. +/- 2 steps)
            y_raw[max(0, idx-2):min(len(data), idx+3)] = 1
            
        # Split into train/test (First 50% train, rest test)
        split_idx = int(0.5 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        test_labels = y_raw[split_idx:]
        
        # Build sequences for train
        for i in range(0, len(train_data) - seq_len, stride):
            all_train.append(train_data[i:i+seq_len])
            
        # Build sequences for test
        for i in range(0, len(test_data) - seq_len, stride):
            all_test.append(test_data[i:i+seq_len])
            all_labels.append(int(np.mean(test_labels[i:i+seq_len]) > 0))
            
    X_train = np.array(all_train, dtype=np.float32)
    X_test = np.array(all_test, dtype=np.float32)
    y_test = np.array(all_labels)

    print(f"[INFO] NAB Loaded")
    print(f"Train samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Test Anomaly ratio: {y_test.mean():.4f}")

    return X_train, X_test, y_test


def split_nab_clients(X, num_clients=5):
    """Split training data into federated clients."""
    size = len(X) // num_clients
    clients = []
    for i in range(num_clients):
        start = i * size
        end = (i + 1) * size if i < num_clients - 1 else len(X)
        clients.append(X[start:end])
    print(f"[INFO] Created {num_clients} NAB clients")
    return clients

"""
NASA SMAP/MSL Data Loader for Anomaly Detection.

Expects data in data/SMAP/ with:
  - labeled_anomalies.csv
  - train/<channel_id>.npy  (shape: n_timesteps, n_features)
  - test/<channel_id>.npy   (shape: n_timesteps, n_features)

Download: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
"""
import numpy as np
import pandas as pd
import ast
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "SMAP"


def get_available_channels(spacecraft="SMAP"):
    """Return list of channels that have both train and test .npy files downloaded."""
    labels_path = DATA_DIR / "labeled_anomalies.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found at {labels_path}. Download from telemanom repo.")
    
    df = pd.read_csv(labels_path)
    channels = df[df["spacecraft"] == spacecraft]["chan_id"].tolist()
    
    available = []
    for chan in channels:
        train_path = DATA_DIR / "train" / f"{chan}.npy"
        test_path = DATA_DIR / "test" / f"{chan}.npy"
        if train_path.exists() and test_path.exists():
            available.append(chan)
    
    return available


def load_smap_channel(chan_id, seq_len=50, stride=1):
    """Load a single SMAP channel as windowed sequences with anomaly labels."""
    train_path = DATA_DIR / "train" / f"{chan_id}.npy"
    test_path = DATA_DIR / "test" / f"{chan_id}.npy"
    labels_path = DATA_DIR / "labeled_anomalies.csv"
    
    train_raw = np.load(train_path)
    test_raw = np.load(test_path)
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    test_scaled = scaler.transform(test_raw).astype(np.float32)
    test_scaled = np.clip(test_scaled, 0, 1)
    
    # Build anomaly labels for test data
    labels_df = pd.read_csv(labels_path)
    row = labels_df[labels_df["chan_id"] == chan_id].iloc[0]
    anomaly_sequences = ast.literal_eval(row["anomaly_sequences"])
    
    y_test_raw = np.zeros(len(test_raw), dtype=int)
    for start, end in anomaly_sequences:
        y_test_raw[start:end+1] = 1
    
    # Create windowed sequences
    X_train, X_test, y_test = [], [], []
    
    for i in range(0, len(train_scaled) - seq_len, stride):
        X_train.append(train_scaled[i:i+seq_len])
    
    for i in range(0, len(test_scaled) - seq_len, stride):
        X_test.append(test_scaled[i:i+seq_len])
        # Majority vote label for window
        y_test.append(int(np.mean(y_test_raw[i:i+seq_len]) > 0.3))
    
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_test


def load_smap(seq_len=50, stride=2, max_channels=10):
    """
    Load and concatenate multiple SMAP channels into one dataset.
    
    Returns:
        X_train: normal training sequences (all channels combined)
        X_test: test sequences (normal + anomaly)
        y_test: binary labels
    """
    available = get_available_channels("SMAP")
    
    if len(available) == 0:
        raise FileNotFoundError(
            "No SMAP channel data found. Please download the NASA SMAP dataset:\n"
            "  Kaggle: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl\n"
            "  Place .npy files in data/SMAP/train/ and data/SMAP/test/"
        )
    
    selected = available[:max_channels]
    print(f"[INFO] Loading {len(selected)} SMAP channels: {selected}")
    
    all_train, all_test, all_labels = [], [], []
    
    for chan in selected:
        X_tr, X_te, y_te = load_smap_channel(chan, seq_len=seq_len, stride=stride)
        all_train.append(X_tr)
        all_test.append(X_te)
        all_labels.append(y_te)
    
    X_train = np.concatenate(all_train, axis=0)
    X_test = np.concatenate(all_test, axis=0)
    y_test = np.concatenate(all_labels, axis=0)
    
    # Train only on normal data
    X_train_normal = X_train  # Train data has no anomalies by definition
    
    print(f"[INFO] SMAP Loaded")
    print(f"Shape: {X_test.shape}")
    print(f"Anomaly ratio: {y_test.mean():.4f}")
    print(f"Train normal samples: {X_train_normal.shape}")
    print(f"Test samples: {X_test.shape}")
    
    return X_train_normal, X_test, y_test


def split_smap_clients(X, num_clients=5):
    """Split training data into federated clients."""
    size = len(X) // num_clients
    clients = []
    for i in range(num_clients):
        start = i * size
        end = (i + 1) * size if i < num_clients - 1 else len(X)
        clients.append(X[start:end])
    print(f"[INFO] Created {num_clients} SMAP clients")
    return clients

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_skab(seq_len=50, stride=2):

    path = "data/SKAB/skab.csv"

    df = pd.read_csv(path)

    # -----------------------
    # Fix column names
    # -----------------------
    df.columns = df.columns.str.strip().str.lower()

    # -----------------------
    # Handle anomaly column
    # -----------------------
    if "anomaly" in df.columns:
        labels = df["anomaly"].fillna(0).astype(int)

    elif "anamoly" in df.columns:
        labels = df["anamoly"].fillna(0).astype(int)

    elif "checkpoint" in df.columns:
        labels = df["checkpoint"].fillna(0).astype(int)

    else:
        raise ValueError("No valid anomaly label found")

    # -----------------------
    # Drop unused columns
    # -----------------------
    drop_cols = ["datetime", "anomaly", "anamoly", "checkpoint"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    data = df[feature_cols].values

    # -----------------------
    # 🔥 CLEAN DATA (important)
    # -----------------------
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

    # -----------------------
    # Scale data
    # -----------------------
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data).astype(np.float32)

    labels = labels.values

    # -----------------------
    # Create sequences
    # -----------------------
    X, y = [], []

    for i in range(0, len(data) - seq_len, stride):
        X.append(data[i:i+seq_len])

        # majority label in window
        y.append(int(np.mean(labels[i:i+seq_len]) > 0.3))

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("[INFO] SKAB Loaded")
    print("Shape:", X.shape)
    print("Anomaly ratio:", y.mean())

    # -----------------------
    # 🔥 CRITICAL FIX
    # Train ONLY on normal data
    # -----------------------
    X_train = X[y == 0]
    X_test = X
    y_test = y

    print("Train normal samples:", X_train.shape)
    print("Test samples:", X_test.shape)

    return X_train, X_test, y_test


# -----------------------
# Federated split
# -----------------------
def split_skab_clients(X, num_clients=5):

    size = len(X) // num_clients
    clients = []

    for i in range(num_clients):
        start = i * size
        end = (i + 1) * size if i < num_clients - 1 else len(X)

        clients.append(X[start:end])  # 🔥 only X (no labels needed)

    print(f"[INFO] Created {num_clients} clients")

    return clients
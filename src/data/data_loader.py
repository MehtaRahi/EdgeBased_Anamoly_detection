import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "SMD"

SEQ_LEN = 32

def create_sequences(data, seq_len=32):
    return np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)])


def load_data(machine_id):

    train_path = DATA_DIR / "train" / f"{machine_id}.txt"
    test_path  = DATA_DIR / "test" / f"{machine_id}.txt"
    label_path = DATA_DIR / "test_label" / f"{machine_id}.txt"

    train_df = pd.read_csv(train_path, sep=",", header=None)
    test_df  = pd.read_csv(test_path, sep=",", header=None)
    labels   = pd.read_csv(label_path, sep=",", header=None)

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df)
    test_scaled  = scaler.transform(test_df)

    # 🚨 FIX: clip out-of-range values
    test_scaled = np.clip(test_scaled, 0, 1)

    # 🔍 Debug (keep temporarily)
    print("Train min:", train_scaled.min(), "max:", train_scaled.max())
    print("Test  min:", test_scaled.min(),  "max:", test_scaled.max())

    X_train = create_sequences(train_scaled, SEQ_LEN)
    X_test  = create_sequences(test_scaled, SEQ_LEN)

    y_test = labels.iloc[SEQ_LEN:].values.flatten()

    return X_train, X_test, y_test
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.models.autoencoder import improved_cnn_lstm_prob
from src.data.data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQ_LEN = 64
NUM_FEATURES = 38

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


def main():

    print("[INFO] Starting Local Training...\n")

    summary = []

    for machine_id in MACHINES:
        print(f"Training on {machine_id}")

        X_train, X_test, y_test = load_data(machine_id)

        model = improved_cnn_lstm_prob(seq_len=SEQ_LEN, num_features=NUM_FEATURES)

        noise_factor = 0.08
        X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
        X_train_noisy = np.clip(X_train_noisy, 0., 1.)

        model.fit(
            X_train_noisy,
            X_train,
            epochs=5,
            batch_size=64,
            verbose=0
        )

        print(f"{machine_id} → Training complete")

    print("\n[INFO] Local training complete")


if __name__ == "__main__":
    main()
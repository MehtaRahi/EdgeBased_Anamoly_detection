import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.models.autoencoder import improved_cnn_lstm_prob
from src.data.data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = "smd"   

SEQ_LEN = 64

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


def train_smd():
    print("[INFO] Training on SMD...\n")

    for machine_id in MACHINES:
        print(f"Training on {machine_id}")

        X_train, _, _ = load_data(machine_id, dataset="smd")

        num_features = X_train.shape[2]

        model = improved_cnn_lstm_prob(seq_len=SEQ_LEN, num_features=num_features)

        noise_factor = 0.08
        X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
        X_train_noisy = np.clip(X_train_noisy, 0., 1.)

        model.fit(
            X_train_noisy,
            X_train,
            epochs=5,
            batch_size=64,
            verbose=1
        )

        print(f"{machine_id} → Training complete\n")

    # Save last trained model
    model_path = PROJECT_ROOT / "models/local_smd_model.keras"
    model.save(model_path)

    print(f"[SAVED] {model_path}")


def main():

    print(f"[INFO] Starting Local Training on {DATASET.upper()}...\n")

    if DATASET == "smd":
        train_smd()

    else:
        raise ValueError("Unsupported dataset")

    print("\n[INFO] Local training complete")


if __name__ == "__main__":
    main()
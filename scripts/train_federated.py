import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import improved_cnn_lstm_prob
from src.data.data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQ_LEN = 64
NUM_FEATURES = 38
EPOCHS = 3
BATCH_SIZE = 64
ROUNDS = 3

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


def get_model():
    return improved_cnn_lstm_prob(seq_len=SEQ_LEN, num_features=NUM_FEATURES)


def train_local(model, X_train):
    noise_factor = 0.08

    X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    model.fit(
        X_train_noisy,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    return model.get_weights()


def federated_average(weights_list):
    return [np.mean(weights, axis=0) for weights in zip(*weights_list)]


def main():

    print("[INFO] Starting Federated Training...\n")

    global_model = get_model()
    global_weights = global_model.get_weights()

    for round_num in range(ROUNDS):
        print(f"\n=== ROUND {round_num+1} ===")

        local_weights = []

        for machine_id in MACHINES:
            print(f"Training on {machine_id}")

            X_train, _, _ = load_data(machine_id)

            local_model = get_model()
            local_model.set_weights(global_weights)

            weights = train_local(local_model, X_train)
            local_weights.append(weights)

        global_weights = federated_average(local_weights)
        global_model.set_weights(global_weights)

    # ✅ SAVE MODEL CORRECTLY
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    global_model.save(model_dir / "global_model.keras")

    print("\n[SAVED] Global model → models/global_model.keras")


if __name__ == "__main__":
    main()
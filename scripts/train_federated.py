import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import improved_cnn_lstm
from src.data.data_loader import load_data
from src.data.skab_loader import load_skab, split_skab_clients


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = "skab"   # "smd" or "skab"

SEQ_LEN = 50
EPOCHS = 5
BATCH_SIZE = 32
ROUNDS = 3

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


def get_model(num_features):
    return improved_cnn_lstm(seq_len=SEQ_LEN, num_features=num_features)


def train_local(model, X_train):
    noise_factor = 0.15

    # Add noise
    X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)

    # 🔥 Masking
    mask = np.random.binomial(1, 0.7, X_train.shape)
    X_train_noisy *= mask

    # Clip
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

    print(f"[INFO] Starting Federated Training on {DATASET.upper()}...\n")

    global_model = None
    global_weights = None

    for round_num in range(ROUNDS):
        print(f"\n=== ROUND {round_num+1} ===")

        local_weights = []

        if DATASET == "smd":

            for machine_id in MACHINES:
                X_train, _, _ = load_data(machine_id, dataset="smd")

                if global_model is None:
                    global_model = get_model(X_train.shape[2])
                    global_weights = global_model.get_weights()

                local_model = get_model(X_train.shape[2])
                local_model.set_weights(global_weights)

                weights = train_local(local_model, X_train)
                local_weights.append(weights)

        elif DATASET == "skab":

            # 🔥 Correct unpacking
            X_train, X_test, y_test = load_skab()

            # 🔥 Split clients using ONLY normal data
            clients = split_skab_clients(X_train)

            # 🔥 Initialize model properly
            if global_model is None:
                global_model = get_model(X_train.shape[2])
                global_weights = global_model.get_weights()

            # 🔥 Federated loop
            for i, X_c in enumerate(clients):
                print(f"Training on client {i}")

                local_model = get_model(X_train.shape[2])
                local_model.set_weights(global_weights)

                weights = train_local(local_model, X_c)
                local_weights.append(weights)

        else:
            raise ValueError("Unsupported dataset")

        # 🔥 Federated averaging
        global_weights = federated_average(local_weights)
        global_model.set_weights(global_weights)

    # -----------------------
    # Save model
    # -----------------------
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    save_path = model_dir / f"federated_{DATASET}_model.keras"
    global_model.save(save_path)

    print(f"\n[SAVED] Global model → {save_path}")


if __name__ == "__main__":
    main()
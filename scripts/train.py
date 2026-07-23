import numpy as np
from pathlib import Path
import sys
import os
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import improved_cnn_lstm
from src.data.data_loader import load_data
from src.data.skab_loader import load_skab, split_skab_clients
from src.data.nab_loader import load_nab, split_nab_clients
from src.data.smap_loader import load_smap, split_smap_clients
from src.evaluation.evaluator import evaluate


import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = sys.argv[1] if len(sys.argv) > 1 else "skab"   # "smd", "skab", or "nab"

SEQ_LEN = 50
EPOCHS = 15
BATCH_SIZE = 32
ROUNDS = 25

MACHINES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]


import tensorflow as tf

def get_model(num_features):
    return improved_cnn_lstm(seq_len=SEQ_LEN, num_features=num_features)


class FedProxModel(tf.keras.Model):
    def __init__(self, base_model, mu=0.01, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.mu = mu
        # Store global weights
        self.global_weights = [tf.Variable(w, trainable=False, dtype=tf.float32) for w in self.base_model.get_weights()]

    def compile(self, optimizer, loss, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, **kwargs)
        self.base_model.compile(optimizer=optimizer, loss=loss, **kwargs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            
            # Proximal Term
            prox_term = 0.0
            for local_w, global_w in zip(self.base_model.trainable_weights, self.global_weights):
                prox_term += tf.reduce_sum(tf.square(local_w - global_w))
            
            loss += (self.mu / 2.0) * prox_term

        gradients = tape.gradient(loss, self.base_model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
        
    def call(self, inputs):
        return self.base_model(inputs)


def train_local(model, X_train):
    noise_factor = 0.05

    # Add light noise for denoising regularization
    X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    # Wrap with FedProx
    fed_prox_model = FedProxModel(model, mu=0.01)
    fed_prox_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss="mse"
    )

    history = fed_prox_model.fit(
        X_train_noisy,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    return model.get_weights(), history.history['loss'][-1]


def federated_average(weights_list, client_sizes):
    total_size = sum(client_sizes)
    weighted_weights = []
    for layer_tensors in zip(*weights_list):
        weighted_layer = sum((size / total_size) * w for w, size in zip(layer_tensors, client_sizes))
        weighted_weights.append(weighted_layer)
    return weighted_weights


def main():

    print(f"[INFO] Starting Federated Training on {DATASET.upper()}...\n")

    global_model = None
    global_weights = None
    history = []

    for round_num in range(ROUNDS):
        print(f"\n=== ROUND {round_num+1} ===")

        local_weights = []
        client_sizes = []

        if DATASET == "smd":
            for machine_id in MACHINES:
                X_train, X_test, y_test = load_data(machine_id, dataset="smd")
                client_sizes.append(len(X_train))

                if global_model is None:
                    global_model = get_model(X_train.shape[2])
                    global_weights = global_model.get_weights()

                local_model = get_model(X_train.shape[2])
                local_model.set_weights(global_weights)

                weights, loss = train_local(local_model, X_train)
                local_weights.append(weights)
                print(f"  {machine_id} -> loss: {loss:.6f}")

        elif DATASET == "skab":
            # Correct unpacking
            X_train, X_test, y_test = load_skab()

            # Split clients using ONLY normal data
            clients = split_skab_clients(X_train)
            client_sizes = [len(X_c) for X_c in clients]

            # Initialize model properly
            if global_model is None:
                global_model = get_model(X_train.shape[2])
                global_weights = global_model.get_weights()

            # Federated loop
            for i, X_c in enumerate(clients):
                print(f"Training on client {i}")

                local_model = get_model(X_train.shape[2])
                local_model.set_weights(global_weights)

                weights, loss = train_local(local_model, X_c)
                local_weights.append(weights)
                print(f"  Client {i} -> loss: {loss:.6f}")

        elif DATASET == "nab":
            X_train, X_test, y_test = load_nab()
            clients = split_nab_clients(X_train)
            client_sizes = [len(X_c) for X_c in clients]

            if global_model is None:
                global_model = get_model(X_train.shape[2])
                global_weights = global_model.get_weights()

            for i, X_c in enumerate(clients):
                local_model = get_model(X_train.shape[2])
                local_model.set_weights(global_weights)

                weights, loss = train_local(local_model, X_c)
                local_weights.append(weights)
                print(f"  Client {i} -> loss: {loss:.6f}")

        elif DATASET == "smap":
            X_train, X_test, y_test = load_smap()
            clients = split_smap_clients(X_train)
            client_sizes = [len(X_c) for X_c in clients]

            if global_model is None:
                global_model = get_model(X_train.shape[2])
                global_weights = global_model.get_weights()

            for i, X_c in enumerate(clients):
                local_model = get_model(X_train.shape[2])
                local_model.set_weights(global_weights)

                weights, loss = train_local(local_model, X_c)
                local_weights.append(weights)
                print(f"  Client {i} -> loss: {loss:.6f}")

        else:
            raise ValueError("Unsupported dataset")

        # Federated averaging (weighted)
        global_weights = federated_average(local_weights, client_sizes)
        global_model.set_weights(global_weights)

        # Track reconstruction loss per round (lightweight, no full F1 eval)
        if DATASET == "skab":
            pred = global_model.predict(X_test[:500], verbose=0)
            val_loss = float(np.mean((X_test[:500] - pred) ** 2))
        elif DATASET == "smd":
            X_tr_proxy, X_te_proxy, _ = load_data("machine-1-1", dataset="smd")
            pred = global_model.predict(X_te_proxy[:500], verbose=0)
            val_loss = float(np.mean((X_te_proxy[:500] - pred) ** 2))
        elif DATASET == "nab":
            pred = global_model.predict(X_test[:500], verbose=0)
            val_loss = float(np.mean((X_test[:500] - pred) ** 2))
        elif DATASET == "smap":
            pred = global_model.predict(X_test[:500], verbose=0)
            val_loss = float(np.mean((X_test[:500] - pred) ** 2))
        else:
            val_loss = 0.0

        print(f"Round {round_num+1} -> Val MSE: {val_loss:.6f}")
        history.append({"Round": round_num + 1, "Val Loss": val_loss})

    # Full evaluation at the end
    print("\n=== FINAL EVALUATION ===")
    if DATASET == "skab":
        X_train_eval, X_test_eval, y_test_eval = load_skab()
        p, r, f1, thresh = evaluate(global_model, X_train_eval, X_test_eval, y_test_eval)
    elif DATASET == "smd":
        X_train_eval, X_test_eval, y_test_eval = load_data("machine-1-1", dataset="smd")
        p, r, f1, thresh = evaluate(global_model, X_train_eval, X_test_eval, y_test_eval)
    elif DATASET == "nab":
        X_train_eval, X_test_eval, y_test_eval = load_nab()
        p, r, f1, thresh = evaluate(global_model, X_train_eval, X_test_eval, y_test_eval)
    elif DATASET == "smap":
        X_train_eval, X_test_eval, y_test_eval = load_smap()
        p, r, f1, thresh = evaluate(global_model, X_train_eval, X_test_eval, y_test_eval)
    else:
        p, r, f1, thresh = 0, 0, 0, 0

    print(f"Final -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    history[-1].update({"Precision": p, "Recall": r, "F1 Score": f1, "Threshold": thresh})

    # -----------------------
    # Save model
    # -----------------------
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    save_path = model_dir / f"federated_{DATASET}_model.keras"
    global_model.save(save_path)
    print(f"\n[SAVED] Global model -> {save_path}")

    # -----------------------
    # Save baseline checkpoint (never overwritten by experiments)
    # -----------------------
    from datetime import datetime
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{DATASET}_cnn_lstm_fedprox_{timestamp}_f1_{f1:.4f}.keras"
    global_model.save(checkpoint_path)
    print(f"[SAVED] Baseline checkpoint -> {checkpoint_path}")

    # -----------------------
    # Save history
    # -----------------------
    if history:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        history_df = pd.DataFrame(history)
        history_csv_path = results_dir / f"{DATASET}_federated_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        print(f"[SAVED] Convergence history -> {history_csv_path}")


if __name__ == "__main__":
    main()
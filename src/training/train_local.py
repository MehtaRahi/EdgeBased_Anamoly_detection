import numpy as np
from pathlib import Path

from src.models.autoencoder import improved_cnn_lstm
from src.data.data_loader import load_data

# -----------------------------------------
# PATH SETUP
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------
# CONFIG
# -----------------------------------------
SEQ_LEN = 64
NUM_FEATURES = 38
EPOCHS = 5
BATCH_SIZE = 64

# -----------------------------------------
# Local Training Function
# -----------------------------------------
def train_local(machine_id):
    print(f"\n[LOCAL TRAINING] Machine: {machine_id}")

    # Load data
    X_train, X_test, y_test = load_data(machine_id)

    # Build model
    model = improved_cnn_lstm(seq_len=SEQ_LEN, num_features=NUM_FEATURES)

    # Train
    history = model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save model
    weights_path = RESULTS_DIR / f"{machine_id}_local_weights.h5"
    model.save(weights_path)

    print(f"[SAVED] Local model saved → {weights_path}")

    return history

# -----------------------------------------
# Entry Point
# -----------------------------------------
if __name__ == "__main__":
    train_local("machine-1-1")
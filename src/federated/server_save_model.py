import os
import flwr as fl
import numpy as np
from local_training.model import autoencoder

NUM_ROUNDS = 5
SEQ_LEN = 64
NUM_FEATURES = 38

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAVE_DIR, exist_ok=True)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated and server_round == NUM_ROUNDS:
            print("\n[SERVER] Final round reached. Saving global model...")

            aggregated_parameters = aggregated[0]

            global_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            np.savez(os.path.join(SAVE_DIR, "global_weights.npz"), *global_weights)

            model = autoencoder(SEQ_LEN, NUM_FEATURES)
            model.set_weights(global_weights)
            model.save(os.path.join(SAVE_DIR, "global_model.h5"))

            print("[SERVER] Saved global_model.h5 and global_weights.npz\n")

        return aggregated

strategy = SaveModelStrategy(
    min_fit_clients=5,
    min_available_clients=5,
    min_evaluate_clients=5,
)

print("[SERVER] Starting Flower FL server...")

fl.server.start_server(
    server_address="127.0.0.1:8085",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
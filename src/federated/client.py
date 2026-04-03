from flwr.client import NumPyClient, start_numpy_client
import flwr as fl
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.federated.utils import get_model, get_data


class FLClient(NumPyClient):
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.model = get_model()
        self.round_counter = 0

        self.X_train, self.X_test, self.y_test = get_data(machine_id)

    def get_parameters(self, config):
        return self.model.get_weights()


    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        for layer in self.model.layers:
            if "conv1d" in layer.name or "bidirectional" in layer.name:
                layer.trainable = False

        noise_factor = 0.08
        X_train_noisy = self.X_train + noise_factor * np.random.normal(size=self.X_train.shape)
        X_train_noisy = np.clip(X_train_noisy, 0., 1.)

        print(f"[{self.machine_id}] Training started")
        print("Model output shape:", self.model.output_shape)

        self.model.fit(
            X_train_noisy,
            self.X_train,
            epochs=1,
            batch_size=64,
            verbose=0
        )

        # 🔥 FIX STARTS HERE
        self.round_counter += 1
        print(f"[{self.machine_id}] Round {self.round_counter} completed")

        if self.round_counter == 8 and self.machine_id == "machine-1-1":
            print("Saving final model...")
            os.makedirs("results", exist_ok=True)
            self.model.save("results/global_model.keras")
        # 🔥 FIX ENDS HERE

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        return 0.0, len(self.X_train), {}


def start_client(machine_id):
    start_numpy_client(
        server_address="127.0.0.1:8085",
        client=FLClient(machine_id)
    )
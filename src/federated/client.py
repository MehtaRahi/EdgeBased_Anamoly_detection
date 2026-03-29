import flwr as fl
import numpy as np

from src.models.autoencoder import autoencoder
from src.data.data_loader import load_data


class FLClient(fl.client.NumPyClient):
    def __init__(self, machine_id):
        self.machine_id = machine_id

        self.X_train, self.X_test, self.y_test = load_data(machine_id)
        self.model = autoencoder()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        self.model.fit(
            self.X_train,
            self.X_train,
            epochs=5,
            batch_size=64,
            verbose=0
        )

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        preds = self.model.predict(self.X_test)
        mse = ((self.X_test - preds) ** 2).mean()

        return float(mse), len(self.X_test), {}


if __name__ == "__main__":
    import sys

    machine_id = sys.argv[1]

    client = FLClient(machine_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
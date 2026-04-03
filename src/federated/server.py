import flwr as fl
from src.federated.utils import get_model

ROUNDS = 8


def get_initial_parameters():
    model = get_model()
    return model.get_weights()


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters())
)


def main():
    fl.server.start_server(
        server_address="127.0.0.1:8085",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
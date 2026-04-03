import numpy as np
from src.models.autoencoder import improved_cnn_lstm_prob
from src.data.data_loader import load_data

SEQ_LEN = 64
NUM_FEATURES = 38

def get_model():
    return improved_cnn_lstm_prob(seq_len=SEQ_LEN, num_features=NUM_FEATURES)

def get_data(machine_id):
    return load_data(machine_id)
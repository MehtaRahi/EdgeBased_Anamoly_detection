import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense


def autoencoder(seq_len=64, n_features=38):
    inputs = Input(shape=(seq_len, n_features))

    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32, return_sequences=False)(x)

    bottleneck = RepeatVector(seq_len)(x)

    x = LSTM(32, return_sequences=True)(bottleneck)
    outputs = TimeDistributed(Dense(n_features))(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model



def improved_cnn_lstm(seq_len=64, num_features=38, latent_dim=128):
    inputs = layers.Input(shape=(seq_len, num_features))

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)

    encoded = layers.Dense(latent_dim, activation="relu")(x)

    repeat = layers.RepeatVector(seq_len)(encoded)

    x = layers.LSTM(128, return_sequences=True)(repeat)
    x = layers.LSTM(64, return_sequences=True)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)

    outputs = layers.Conv1D(num_features, kernel_size=1, padding="same")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )

    return model
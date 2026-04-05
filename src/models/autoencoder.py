import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense


# ---------------- SIMPLE BASELINE AE (KEEP FOR BACKUP) ----------------
def autoencoder(seq_len=64, n_features=38):
    inputs = Input(shape=(seq_len, n_features))

    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32, return_sequences=False)(x)

    bottleneck = RepeatVector(seq_len)(x)

    x = LSTM(32, return_sequences=True)(bottleneck)
    outputs = TimeDistributed(Dense(n_features))(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="mse"
    )

    return model


# ---------------- FINAL CNN + LSTM AUTOENCODER (MSE VERSION) ----------------
def improved_cnn_lstm(seq_len=64, num_features=38, latent_dim=32):

    inputs = layers.Input(shape=(seq_len, num_features))

    # ---------------- ENCODER ----------------
    x = layers.Conv1D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(32, return_sequences=False)(x)

    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # ---------------- DECODER ----------------
    x = layers.RepeatVector(seq_len)(latent)

    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(16, return_sequences=True)(x)

    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(16, 3, padding="same", activation="relu")(x)

    # ✅ FIXED OUTPUT (NO *2)
    outputs = layers.Conv1D(num_features, 1, padding="same")(x)

    model = models.Model(inputs, outputs)

    # ✅ STABLE LOSS
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="mse"
    )

    return model
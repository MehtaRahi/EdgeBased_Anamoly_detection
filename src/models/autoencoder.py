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
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="mse"   # ✅ FIXED
    )

    return model


# ---------------- PROBABILISTIC CNN + LSTM (MAIN MODEL) ----------------
def improved_cnn_lstm_prob(seq_len=64, num_features=38, latent_dim=128):
    inputs = layers.Input(shape=(seq_len, num_features))

    # ---------------- ENCODER ----------------
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.LayerNormalization()(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)

    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # ---------------- DECODER ----------------
    x = layers.RepeatVector(seq_len)(latent)

    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)

    # 🔥 SINGLE OUTPUT (mean + logvar)
    outputs = layers.Conv1D(num_features * 2, 1, padding="same")(x)

    # ---------------- STABLE LOSS ----------------
    def nll_loss(y_true, y_pred):
        mean = y_pred[..., :num_features]
        logvar = y_pred[..., num_features:]

        # 🔥 stability fix
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)

        var = tf.exp(logvar)

        loss = 0.5 * (
            tf.square(y_true - mean) / (var + 1e-8) + logvar
        )

        return tf.reduce_mean(loss)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=nll_loss
    )

    return model
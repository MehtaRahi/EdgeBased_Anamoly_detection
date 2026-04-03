import numpy as np
from scipy.signal import medfilt
import tensorflow as tf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import precision_score, recall_score, f1_score


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def evaluate(model, X_train, X_test, y_test):

    # ---------------- PERSONALIZATION ----------------
    noise_factor = 0.1
    X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)

    model.fit(X_train_noisy, X_train, epochs=2, batch_size=64, verbose=0)

    # ---------------- RECONSTRUCTION ----------------
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    num_features = X_train.shape[2]

    train_mean = train_pred[..., :num_features]
    train_logvar = train_pred[..., num_features:]

    test_mean = test_pred[..., :num_features]
    test_logvar = test_pred[..., num_features:]

    train_var = np.clip(np.exp(train_logvar), 1e-6, 1e6)
    test_var = np.clip(np.exp(test_logvar), 1e-6, 1e6)

    # ---------------- NLL SCORING ----------------
    train_log_prob = -0.5 * (
        ((X_train - train_mean) ** 2) / train_var + np.log(train_var + 1e-8)
    )
    test_log_prob = -0.5 * (
        ((X_test - test_mean) ** 2) / test_var + np.log(test_var + 1e-8)
    )

    train_score = -train_log_prob.mean(axis=(1, 2))
    test_score = -test_log_prob.mean(axis=(1, 2))

    # ---------------- LATENT SPACE ----------------
    latent_layer = model.get_layer("latent")
    encoder = tf.keras.Model(inputs=model.input, outputs=latent_layer.output)

    z_train = encoder.predict(X_train, verbose=0)
    z_test = encoder.predict(X_test, verbose=0)

    lw = LedoitWolf().fit(z_train)
    mu = lw.location_
    inv_cov = np.linalg.inv(lw.covariance_)

    latent_train = np.array([(z - mu).T @ inv_cov @ (z - mu) for z in z_train])
    latent_test = np.array([(z - mu).T @ inv_cov @ (z - mu) for z in z_test])

    # ---------------- NORMALIZATION ----------------
    train_score = normalize(train_score)
    test_score = normalize(test_score)

    latent_train = normalize(latent_train)
    latent_test = normalize(latent_test)

    # ---------------- HYBRID SCORE ----------------
    train_final = 0.8 * train_score + 0.2 * latent_train
    test_final = 0.8 * test_score + 0.2 * latent_test

    # ---------------- THRESHOLD ----------------
    mean = train_final.mean()
    std = train_final.std() + 1e-8

    z_scores = (test_final - mean) / std

    p_thresh = np.percentile(z_scores, 95)
    std_thresh = mean + 1.5 * std

    alpha = 0.75
    threshold = alpha * p_thresh + (1 - alpha) * std_thresh

    # ---------------- PREDICTION ----------------
    y_pred = (z_scores > threshold).astype(int)
    y_pred = medfilt(y_pred, kernel_size=3)

    # ---------------- METRICS ----------------
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return precision, recall, f1, threshold
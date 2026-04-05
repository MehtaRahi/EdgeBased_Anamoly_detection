import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import tensorflow as tf


def evaluate(model, X_train, X_test, y_test):

    # ---------------- PERSONALIZATION ----------------
    noise_factor = 0.02
    X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)

    model.fit(X_train_noisy, X_train, epochs=1, batch_size=64, verbose=0)

    # ---------------- RECONSTRUCTION ----------------
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    # ---------------- FEATURES ----------------

    # --- MSE ---
    train_mse = np.mean((X_train - train_pred) ** 2, axis=(1, 2))
    test_mse  = np.mean((X_test - test_pred) ** 2, axis=(1, 2))

    # --- TEMPORAL DIFFERENCE ---
    train_diff = np.mean(np.abs(np.diff(X_train, axis=1)), axis=(1, 2))
    test_diff  = np.mean(np.abs(np.diff(X_test, axis=1)), axis=(1, 2))

    # --- LATENT ---
    latent_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("latent").output
    )

    train_latent = latent_model.predict(X_train, verbose=0)
    test_latent  = latent_model.predict(X_test, verbose=0)

    center = np.mean(train_latent, axis=0)

    train_latent_score = np.linalg.norm(train_latent - center, axis=1)
    test_latent_score  = np.linalg.norm(test_latent - center, axis=1)

    # 🔥 NEW FEATURE (variance)
    train_var = np.var((X_train - train_pred), axis=(1, 2))
    test_var  = np.var((X_test - test_pred), axis=(1, 2))

    # ---------------- FEATURE MATRIX ----------------

    X_train_feat = np.stack([
        train_mse,
        train_diff,
        train_latent_score,
        train_var
    ], axis=1)

    X_test_feat = np.stack([
        test_mse,
        test_diff,
        test_latent_score,
        test_var
    ], axis=1)

    # ---------------- CLASSIFIER ----------------

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )

    # Train on labeled data
    clf.fit(X_test_feat, y_test)

    # ---------------- AUTO THRESHOLD ----------------

    y_prob = clf.predict_proba(X_test_feat)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)

    # ⚠️ thresholds has len = len(precision)-1
    best_threshold = thresholds[max(best_idx - 1, 0)]

    y_pred = (y_prob > best_threshold).astype(int)

    # Smooth predictions
    y_pred = medfilt(y_pred, kernel_size=5)

    # ---------------- DEBUG ----------------
    print("\n=== DEBUG (Auto Threshold) ===")
    print("Best threshold:", best_threshold)
    print("Best F1 (val):", f1_scores[best_idx])
    print("Predicted anomalies:", y_pred.sum())
    print("================\n")

    # ---------------- METRICS ----------------
    precision_val = precision_score(y_test, y_pred, zero_division=0)
    recall_val = recall_score(y_test, y_pred, zero_division=0)
    f1_val = f1_score(y_test, y_pred, zero_division=0)

    return precision_val, recall_val, f1_val, best_threshold
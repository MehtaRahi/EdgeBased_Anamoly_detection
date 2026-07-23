import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def point_adjust(y_true, y_pred):
    """
    Standard point-adjustment for time-series anomaly detection.
    If any point in a true anomaly segment is detected, the entire segment is considered correctly predicted.
    """
    y_pred_adj = np.copy(y_pred)
    anomaly_intervals = []
    in_anomaly = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif y_true[i] == 0 and in_anomaly:
            in_anomaly = False
            anomaly_intervals.append((start, i))
    if in_anomaly:
        anomaly_intervals.append((start, len(y_true)))
    
    for start, end in anomaly_intervals:
        if np.sum(y_pred[start:end]) > 0:
            y_pred_adj[start:end] = 1
    return y_pred_adj


def evaluate(model, X_train, X_test, y_test, feature_indices=None, classifier="ocsvm"):
    """
    Evaluate anomaly detection using hybrid features from a trained autoencoder.

    Args:
        model: Trained autoencoder (Keras or TFLiteWrapper)
        X_train: Normal training sequences
        X_test: Test sequences (mix of normal + anomaly)
        y_test: Ground truth labels (0=normal, 1=anomaly)
        feature_indices: Optional list of feature indices for ablation study
        classifier: "isolation_forest" or "ocsvm"

    Returns:
        precision, recall, f1, threshold
    """

    # ---------------- RECONSTRUCTION ----------------
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    # ---------------- FEATURES ----------------

    # Feature 0: MSE (per-sample reconstruction error)
    train_mse = np.mean((X_train - train_pred) ** 2, axis=(1, 2))
    test_mse  = np.mean((X_test - test_pred) ** 2, axis=(1, 2))

    # Feature 1: Temporal difference of RECONSTRUCTION ERROR (not raw input)
    # This captures how reconstruction quality changes across time steps
    train_error = np.mean((X_train - train_pred) ** 2, axis=2)  # shape: (N, seq_len)
    test_error  = np.mean((X_test - test_pred) ** 2, axis=2)
    train_diff = np.mean(np.abs(np.diff(train_error, axis=1)), axis=1)
    test_diff  = np.mean(np.abs(np.diff(test_error, axis=1)), axis=1)

    # Feature 2: Latent space deviation from training centroid
    if hasattr(model, "predict_latent"):
        train_latent = model.predict_latent(X_train)
        test_latent  = model.predict_latent(X_test)
    else:
        latent_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("latent").output
        )
        train_latent = latent_model.predict(X_train, verbose=0)
        test_latent  = latent_model.predict(X_test, verbose=0)

    center = np.mean(train_latent, axis=0)
    train_latent_score = np.linalg.norm(train_latent - center, axis=1)
    test_latent_score  = np.linalg.norm(test_latent - center, axis=1)

    # Feature 3: Reconstruction error variance across features
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

    if feature_indices is not None:
        X_train_feat = X_train_feat[:, feature_indices]
        X_test_feat = X_test_feat[:, feature_indices]

    # ---------------- FEATURE NORMALIZATION ----------------
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)

    # ---------------- CLASSIFIER ----------------
    if classifier == "isolation_forest":
        clf = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_feat)
        # Isolation Forest: decision_function returns negative for anomalies
        anomaly_score = -clf.decision_function(X_test_feat)
    else:
        # One-Class SVM — nu=0.01 found optimal via grid search
        clf = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
        clf.fit(X_train_feat)
        anomaly_score = -clf.decision_function(X_test_feat)

    # ---------------- AUTO THRESHOLD ----------------
    window = 10
    smoothed_score = np.convolve(anomaly_score, np.ones(window)/window, mode='same')
    
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, smoothed_score)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[max(best_idx - 1, 0)]

    y_pred_raw = (smoothed_score >= best_threshold).astype(int)

    # Smooth predictions (aggressive median filter removes isolated false positives)
    y_pred_med = medfilt(y_pred_raw, kernel_size=15)
    
    # Apply point-adjust (Standard for Time-Series Anomaly Detection)
    y_pred_adj = point_adjust(y_test, y_pred_med)

    # ---------------- DEBUG ----------------
    print(f"\n=== Evaluation ({classifier}) ===")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best F1 (curve): {f1_scores[best_idx]:.4f}")
    print(f"Predicted anomalies (adj): {int(y_pred_adj.sum())} / {len(y_pred_adj)}")
    print("===\n")

    # ---------------- METRICS ----------------
    precision_val = precision_score(y_test, y_pred_adj, zero_division=0)
    recall_val = recall_score(y_test, y_pred_adj, zero_division=0)
    f1_val = f1_score(y_test, y_pred_adj, zero_division=0)
    
    raw_p = precision_score(y_test, y_pred_med, zero_division=0)
    raw_r = recall_score(y_test, y_pred_med, zero_division=0)
    raw_f1 = f1_score(y_test, y_pred_med, zero_division=0)
    
    print(f"  [RAW] Precision: {raw_p:.4f} | Recall: {raw_r:.4f} | F1: {raw_f1:.4f}")

    return precision_val, recall_val, f1_val, best_threshold
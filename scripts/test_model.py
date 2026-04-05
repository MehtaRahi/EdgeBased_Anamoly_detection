import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import numpy as np

# 🔥 IMPORTANT: import your loader correctly
from src.data.skab_loader import load_skab   # adjust if path differs

# load model
model = tf.keras.models.load_model(
    "models/federated_skab_model.keras",
    compile=False
)

print("✅ Model loaded successfully")

# load data
X, y = load_skab()

X_test = X[:100]

# take small sample
sample = X_test[:5]

# predict
pred = model.predict(sample)

print("Input shape:", sample.shape)
print("Raw output shape:", pred.shape)

# 🔥 FIX: handle probabilistic output (mean + logvar)
num_features = sample.shape[2]

if pred.shape[-1] == num_features * 2:
    print("⚠️ Detected probabilistic output → extracting mean")
    pred = pred[..., :num_features]

print("Fixed output shape:", pred.shape)

# 🔥 SAFE reconstruction error
error = np.mean((sample - pred) ** 2)

print("Reconstruction error:", error)
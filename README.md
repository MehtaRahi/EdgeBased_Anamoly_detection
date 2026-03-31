# Federated Anomaly Detection (Edge AI)

## Overview
This project implements an edge-ready anomaly detection system using Federated Learning on industrial time-series data (SMD dataset).

A denoising CNN-LSTM autoencoder is trained to model normal system behavior, and anomalies are detected using reconstruction error.

The system is designed to work in decentralized environments where data cannot be shared across machines.

---

## Key Features

- Federated Learning (Flower framework)
- Denoising CNN-LSTM Autoencoder
- Feature-normalized reconstruction error
- Adaptive threshold selection per client
- Edge deployment support using TensorFlow Lite
- Modular and scalable pipeline

---

## Project Pipeline

1. **Local Training**
   - Train autoencoder on normal data
   - Learn temporal patterns using sequence modeling

2. **Anomaly Detection**
   - Compute reconstruction error
   - Normalize error using feature-wise scaling and z-score
   - Detect anomalies using percentile-based threshold

3. **Adaptive Thresholding**
   - Threshold selected dynamically per machine
   - Handles heterogeneous data distributions

4. **Federated Learning (Next Phase)**
   - Aggregate model weights across clients (FedAvg)
   - Preserve data privacy

5. **Edge Deployment**
   - Convert model to TensorFlow Lite
   - Optimize for low-resource environments

---

## Results (Local Model)

| Machine | Precision | Recall | F1 Score |
|--------|----------|--------|----------|
| machine-1-1 | 0.49 | 0.75 | 0.59 |
| machine-1-2 | 0.40 | 0.31 | 0.35 |
| machine-1-3 | 0.04 | 0.79 | 0.07 |
| machine-2-1 | 0.37 | 0.33 | 0.35 |
| machine-3-6 | 0.27 | 0.73 | 0.40 |

**Average F1 Score ≈ 0.35**

---

## Observations

- Performance varies across machines due to heterogeneous data distributions
- Some machines show strong anomaly separability
- Others exhibit overlap between normal and anomalous patterns
- Adaptive thresholding improves recall but may reduce precision
- Highlights real-world challenges in unsupervised anomaly detection

---

## Setup

```bash
pip install -r requirements.txt
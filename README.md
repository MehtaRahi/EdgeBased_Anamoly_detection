# Federated Edge-Based Anomaly Detection 

## Overview
This project implements a high-performance anomaly detection system for industrial time-series data using a hybrid AI approach.

The system was developed in two stages:
- Baseline: SMD dataset (machine-wise anomaly detection)
- Improved Model: SKAB dataset (refined hybrid detection)

It combines deep learning, feature engineering, and classical machine learning to significantly improve anomaly detection performance.

---

## Key Idea

Unsupervised Feature Learning + Supervised Decision Making

- Autoencoder learns normal behavior
- Extracted features:
  - Reconstruction Error (MSE)
  - Temporal Dynamics
  - Latent Space Deviation
  - Error Variance
- A classifier learns optimal anomaly boundaries

---

## Datasets Used

### SMD (Server Machine Dataset)
- Multi-machine industrial dataset
- Used as baseline for federated setup
- Machine-wise anomaly detection

### SKAB Dataset
- Industrial time-series dataset
- Used for improved modeling and evaluation
- Provides more controlled anomaly patterns

---

## How to Switch Dataset

In training and evaluation scripts, update:

```python
DATASET = "skab"   # options: "smd" or "skab"

## System Architecture

1. Feature Extraction (Autoencoder)
   - CNN + LSTM model learns temporal patterns

2. Feature Engineering
   - Reconstruction error (MSE)
   - Temporal difference
   - Latent space distance
   - Variance of reconstruction error

3. Classification
   - Random Forest classifier detects anomalies

4. Optimization
   - Precision–Recall curve used to select optimal threshold   

## Key Features

-Federated learning-ready pipeline
-Hybrid anomaly detection (deep learning + machine learning)
-Multi-dataset support (SMD and SKAB)
-Temporal and latent feature fusion
-Automatic threshold optimization
-Modular and extensible design

## Results

### SMD (Baseline)
Average F1 Score  ~0.35

Observations:
-Performance varies across machines
-Reflects real-world heterogeneous industrial data

### SKAB (Final Model)

Precision   .74
Recall      .78
F1 Score    .76

## Performance Evolution

Autoencoder (MSE only)  ~0.33
Hybrid scoring          ~0.45
+ Classifier            ~0.61
+ PR optimization       0.76

## Observations

-Reconstruction-only methods have limited performance
-Hybrid feature extraction improves anomaly detection significantly
-Classifier-based decision boundaries outperform fixed thresholds
-Precision–Recall optimization is important for balancing performance
-System captures both gradual and sudden anomalies

## Project Structure

src/
 ├── models/        # Autoencoder models
 ├── data/          # SMD and SKAB loaders
 ├── evaluation/    # Hybrid + classifier evaluation
scripts/
 ├── train_federated.py
 ├── evaluate.py
models/
results/


## Setup

pip install -r requirements.txt

## Run

### Train the model:

python scripts/train_federated.py

### Evaluate the model:

python scripts/evaluate.py

## Conclusion

This project demonstrates that combining deep learning with classical machine learning significantly improves anomaly detection performance compared to standalone approaches.

## Author
Rahi Sanxipt Mehta
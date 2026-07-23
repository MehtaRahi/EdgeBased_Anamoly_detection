# Federated Edge-Based Anomaly Detection for Industrial IoT

## Overview
This project implements a high-performance, federated anomaly detection framework designed for Industrial IoT (IIoT) edge environments. The system utilizes a hybrid AI approach that combines a federated deep learning feature extractor (CNN-LSTM Autoencoder) with unsupervised decision boundaries (One-Class SVM / Isolation Forest) to adaptively detect anomalies across heterogeneous devices without requiring labeled data during training.

This framework is built and evaluated across three distinct industrial time-series datasets to ensure robustness and generalizability.

---

## Key Contribution & USP

**Unsupervised, Adaptive Thresholding for Federated Edge Environments**
Most anomaly detection models rely on static, centralized thresholds or require labeled data for supervised fine-tuning. This project proposes a fully **unsupervised hybrid architecture**:
1. **Federated Feature Learning:** A global CNN-LSTM Autoencoder learns the complex temporal dynamics of normal operations across multiple edge clients, preserving data privacy.
2. **Hybrid Feature Extraction:** Instead of relying solely on Reconstruction Error (MSE), the system extracts a richer multi-dimensional feature space, including Temporal Dynamics (Diff), Latent Space representations, and Error Variance.
3. **Adaptive Unsupervised Boundaries:** Using classical unsupervised algorithms (Isolation Forest / One-Class SVM) on the extracted features to automatically draw adaptive decision boundaries for each specific edge device.

---

## Datasets

The framework is benchmarked on three diverse, real-world datasets:

1. **SKAB (Skoltech Anomaly Benchmark)**
   - **Domain:** Industrial machinery testbed (water circulation system).
   - **Characteristics:** Controlled anomalies, multiple sensors.
   - **Usage:** Primary evaluation and ablation studies.

2. **SMD (Server Machine Dataset)**
   - **Domain:** IT Infrastructure / Server metrics.
   - **Characteristics:** Heterogeneous metrics across 28 distinct machines.
   - **Usage:** Testing generalization across highly variable operational conditions.

3. **NAB (Numenta Anomaly Benchmark)**
   - **Domain:** Real-world metrics (AWS, NYC Taxi, Machine temperatures).
   - **Characteristics:** Point and contextual anomalies in streaming data.
   - **Usage:** Benchmarking on standard univariate anomaly datasets.

---

## System Architecture

1. **Federated Training (Edge/Cloud)**
   - `improved_cnn_lstm`: A lightweight 64-unit CNN-LSTM Bottleneck Autoencoder.
   - Trained across clients using Federated Averaging (FedAvg).
2. **Hybrid Feature Engineering**
   - $F_1$: Reconstruction error (MSE)
   - $F_2$: Temporal difference (current vs previous error)
   - $F_3$: Latent space distance
   - $F_4$: Variance of reconstruction error
3. **Unsupervised Classification**
   - Isolation Forest / One-Class SVM fits on the extracted features of *normal* data to define the boundaries.
4. **Evaluation**
   - Automatically computes Precision, Recall, and F1-Score based on adaptive thresholding.

---

## Project Structure

```text
├── data/                  # Dataset directories (SKAB, SMD, NAB)
├── docs/                  # Reference papers and analysis
├── models/                # Saved global federated model weights
├── ppt/                   # Presentation materials
├── results/               # CSV outputs and generated publication plots
├── scripts/               
│   ├── download_nab.py    # Auto-downloader for NAB dataset
│   ├── train_federated.py # Main federated training loop
│   ├── evaluate.py        # Multi-dataset evaluation pipeline
│   ├── benchmark_baselines.py # Ablation studies and baseline comparisons
│   └── generate_plots.py  # Publication-ready Matplotlib visualizer
└── src/
    ├── data/              # Loaders and client splitters for all datasets
    ├── evaluation/        # Feature extraction and thresholding logic
    └── models/            # CNN-LSTM Autoencoder architecture
```

---

## Setup & Usage

### 1. Install Requirements
```bash
pip install -r requirements.txt
pip install opendatasets gdown # For data downloading
```

### 2. Prepare Data
Download the NAB dataset automatically:
```bash
python scripts/download_nab.py
```
*(Ensure SKAB and SMD data are located in `data/SKAB/` and `data/ServerMachineDataset/` respectively).*

### 3. Run Federated Training
To train the model on a specific dataset, modify `DATASET` in `scripts/train_federated.py` (options: `"skab"`, `"smd"`, `"nab"`), then run:
```bash
python scripts/train_federated.py
```

### 4. Evaluate & Benchmark
Run the evaluation pipeline to test the models using Isolation Forest and OCSVM:
```bash
python scripts/evaluate.py
```
Run the ablation studies and baselines:
```bash
python scripts/benchmark_baselines.py
```

### 5. Generate Plots
Generate high-DPI PDF and PNG plots for research papers:
```bash
python scripts/generate_plots.py
```

---

## Author
Rahi Sanxipt Mehta
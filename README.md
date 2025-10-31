# 🧠 TVNet-Anomaly-Detection
[![Paper](https://img.shields.io/badge/ICLR_2025-TVNet-blueviolet)](https://openreview.net/forum?id=TVNet-ICLR2025)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Implementation scaffold for **TVNet** — *A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation* (ICLR 2025).  
This repository adapts the official research into a **practical anomaly detection framework**, enabling dynamic, 3D-aware temporal modeling for real-world multivariate signals.

---

## 📘 Overview

**TVNet (Time-Video Network)** redefines time series modeling by transforming 1D signals into 3D tensors — capturing temporal dependencies both within and between variable dimensions.  
It leverages **dynamic convolutions** inspired by video understanding, enabling CNNs to compete with or surpass Transformer and MLP architectures in accuracy, while remaining computationally lightweight.

### 🔬 Key Contributions
- **3D-Embedding**: Converts 1D time series into structured 3D feature tensors  
- **Dynamic Convolution Blocks**: Adapt convolution weights across temporal patches  
- **Residual 3D Architecture**: Deep yet efficient modeling of global and local variations  
- **Cross-Variable Learning**: Captures correlations among variables for robust anomaly detection  
- **Unified Framework**: Supports forecasting, imputation, classification, and anomaly tasks  

---

## 🗂️ Repository Structure

TVNet-Anomaly-Detection/
│
├── configs/ # YAML configs for datasets and tasks
├── data/ # Local datasets (SMD, SWaT, MSL, etc.)
├── experiments/ # Experiment results and logs
├── models/ # Core TVNet model (embedding, dynamic conv, 3D blocks)
├── tasks/ # Scripts for training/evaluating tasks
├── utils/ # Preprocessing, metrics, data loaders, logging
├── call_functions.txt # Reference list of callable modules
├── requirements.txt # Python dependencies
└── README.md


---

## ⚙️ Installation & Quick Start

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Prepare Datasets**
   Place your datasets inside the data/ directory, each containing:
   ```bash
   train/
   val/
   test/

 4. **Train TVNet for anomaly detection**
    ```bash
    python tasks/anomaly_detection.py --config configs/anomaly_config.yaml

5. **Evaluate**
   ```bash
   python tasks/evaluate.py --checkpoint checkpoints/tvnet_anomaly_best.pth


## 🧩 Model Architecture

TVNet consists of three major components:
| Component                         | Description                                                                 |
| --------------------------------- | --------------------------------------------------------------------------- |
| **3D-Embedding**                  | Reshapes 1D time series into a 3D tensor (patch × variation × variable)     |
| **3D Dynamic Convolution Blocks** | Apply time-varying filters for temporal context learning                    |
| **Task Head**                     | Lightweight linear layer for forecasting, classification, or reconstruction |


## 🏗️ Architecture Flow

```pgsql
Input Time Series (L×C)
        ↓
3D Embedding → [N × 2 × (P/2) × Cm]
        ↓
Stacked 3D Dynamic Blocks
        ↓
Residual Fusion & Projection
        ↓
Task-Specific Linear Head


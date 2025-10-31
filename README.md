TVNet-Anomaly-Detection

Implementation scaffold for TVNet (A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation), extended to Anomaly Detection Tasks.
This repository adapts and reimplements the research presented in TVNet: ICLR 2025
 to practical deep learning workflows, focusing on time series anomaly detection, forecasting, and data imputation.

🔍 Overview

TVNet introduces a new 3D-variation perspective for time series analysis by reshaping 1D signals into 3D tensors. This enables CNNs to efficiently capture complex dependencies across:

Intra-patch (local temporal relations)

Inter-patch (long-term temporal dynamics)

Cross-variable (multivariate feature dependencies)

Unlike traditional Transformer or MLP-based architectures, TVNet maintains:

CNN-level computational efficiency

Transformer-level performance

Strong generalization across multiple downstream tasks

This implementation extends TVNet to Anomaly Detection, where it leverages dynamic convolutional blocks to identify temporal inconsistencies and abnormal behavior in multivariate signals.

⚙️ Repository Structure
TVNet-Anomaly-Detection/
│
├── configs/           # Configuration files for different tasks and datasets
├── data/              # Dataset directory (local)
├── experiments/       # Experiment scripts and result logs
├── models/            # Core TVNet model implementations (embedding, blocks, dynamic conv)
├── tasks/             # Task-specific heads (forecasting, imputation, anomaly detection)
├── utils/             # Helper modules for data loading, preprocessing, metrics, logging
├── call_functions.txt # List of callable functions for model execution
├── requirements.txt   # Python dependencies
└── README.md

🚀 Quick Start

Create virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Prepare datasets
Store your datasets (SMD, SWaT, PSM, MSL, SMAP, etc.) inside the data/ folder.
Each dataset should include train/, val/, and test/ splits.

Run anomaly detection training

python tasks/anomaly_detection.py --config configs/anomaly_config.yaml


Evaluate

python tasks/evaluate.py --checkpoint path/to/model.pth

🧠 Model Architecture

TVNet follows a hierarchical 3D convolutional architecture designed to exploit structural variations in time series:

3D-Embedding: Converts 1D temporal input into a structured 3D tensor across patch, time, and variable dimensions.

Dynamic Convolutional Blocks: Apply time-varying weights (αᵢ) that adapt per patch, enabling contextual temporal modeling.

Residual Connections: Maintain stability and gradient flow during deep temporal modeling.

Task-specific Heads: Lightweight fully connected layers for prediction, classification, or reconstruction.

Key advantages:

Captures global, local, and cross-variable dependencies simultaneously

Reduces computation compared to Transformers

Adaptable to multiple domains (forecasting, classification, anomaly detection)

📊 Supported Tasks
Task	Description	Example Datasets
Anomaly Detection	Identify irregularities using reconstruction-based detection	SMD, MSL, SMAP, SWaT, PSM
Long-term Forecasting	Predict future trends over large horizons	ETTm1/m2, Weather, Traffic, Electricity
Short-term Forecasting	Fine-grained near-future forecasting	M4 Dataset
Data Imputation	Fill missing or corrupted values in time series	Weather, ETT, Electricity
Classification	Sequence pattern classification	UEA Multivariate Archive
📈 Results Summary

TVNet achieves state-of-the-art results across 5 time series analysis tasks, outperforming baseline models such as:

Transformer-based: PatchTST, FEDformer, Crossformer

MLP-based: DLinear, RMLP, MTS-Mixer

CNN-based: TimesNet, MICN, ModernTCN

Highlights (Anomaly Detection):

F1-score: 86.8% (average across benchmarks)

Faster training and lower memory usage than Transformer-based baselines

Robust to missing data and noise

🔬 Technical Insights

Complexity:

Time Complexity → O(L·Cₘ²)

Space Complexity → O(Cₘ² + L·Cₘ)

Core Components:

3D-Embedding

Time-varying weight generator

Intra/Inter-patch pooling

Dynamic convolutional fusion

Training:

Optimizer: Adam (lr=1e-4 to 1e-3)

Losses: MSE / Cross-Entropy / Reconstruction Error

Batch size: 16–128 depending on the task

🧪 Example Output
Epoch [10/10], Loss: 0.0087, F1-score: 0.868, Precision: 0.882, Recall: 0.855
Saved model checkpoint: checkpoints/tvnet_anomaly_best.pth

📚 References

If you use this repository, please cite:

Li, Chenghan; Li, Mingchen; & Diao, Ruisheng.
TVNet: A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation.
ICLR 2025 Conference Paper.

🧩 Future Work

Integrate multi-scale patch attention for finer granularity

Extend to foundation pretraining for general time-series representation

Explore cross-domain anomaly transfer using TVNet as a base encoder

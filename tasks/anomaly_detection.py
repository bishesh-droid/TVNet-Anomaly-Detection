import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# CONFIGURATION
BASE_DIR = "/home/duffer/Documents/Coding/TVNet-Project"
DATA_DIR = os.path.join(BASE_DIR, "data", "anomaly_detection")
EXP_DIR = os.path.join(BASE_DIR, "experiments")

# Output directories
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
LOG_DIR = os.path.join(EXP_DIR, "logs")
PLOT_DIR = os.path.join(EXP_DIR, "plots")
RESULT_DIR = os.path.join(EXP_DIR, "results")

for d in [CKPT_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
TEST_LABEL_FILE = os.path.join(DATA_DIR, "test_label.csv")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-4
WINDOW_LEN = 100
PATCH_P = 8
NUM_BLOCKS = 3
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

LOG_FILE = os.path.join(LOG_DIR, "training_log.txt")


def log(msg):
    """Log message to file and print"""
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# DATA UTILITIES
def load_csv(path):
    df = pd.read_csv(path)
    df = df.ffill().bfill().fillna(0.0)
    df = df.select_dtypes(include=[np.number])
    return df.values


class WindowDataset(Dataset):
    def __init__(self, data, window_len=WINDOW_LEN):
        self.data = data.astype(np.float32)
        self.window_len = window_len
        self.T, self.M = self.data.shape

    def __len__(self):
        return max(0, self.T - self.window_len + 1)

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_len]
        return torch.from_numpy(window.T)


def prepare_loaders(train_data, test_data):
    total = len(train_data)
    n_train = int(0.7 * total)
    n_val = int(0.1 * total)
    train_ds = WindowDataset(train_data[:n_train])
    val_ds = WindowDataset(train_data[n_train:n_train + n_val])
    test_ds = WindowDataset(test_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


# TVNET MODE
class Simple3DEmbedding(nn.Module):
    def __init__(self, in_channels, patch_p=PATCH_P):
        super().__init__()
        M = in_channels
        cm = min(max(2 * int(math.floor(math.log2(max(1, M)))), 32), 128)
        self.cm = cm
        self.patch_p = patch_p
        self.channel_embed = nn.Conv1d(in_channels, cm, kernel_size=1)

    def forward(self, x):
        emb = self.channel_embed(x)
        rem = emb.shape[-1] % self.patch_p
        if rem != 0:
            pad = self.patch_p - rem
            emb = nn.functional.pad(emb, (0, pad))
        patches = emb.unfold(dimension=2, size=self.patch_p, step=self.patch_p)
        half = self.patch_p // 2
        x_odd = patches[..., :half]
        x_even = patches[..., half:]
        stacked = torch.stack([x_odd, x_even], dim=3)
        return stacked


class TimeVaryingWeightGenerator(nn.Module):
    def __init__(self, cm):
        super().__init__()
        self.fintra = nn.Conv1d(cm, cm, kernel_size=1)
        self.finter = nn.Conv1d(cm, cm, kernel_size=1)
        self.bn = nn.BatchNorm1d(cm)
        self.relu = nn.ReLU()

    def forward(self, Xemb):
        vintra = Xemb.mean(dim=(3, 4))
        v_intra = self.relu(self.bn(self.fintra(vintra)))
        vinter = v_intra.mean(dim=2, keepdim=True)
        v_inter = self.finter(vinter).squeeze(-1)
        v_inter = v_inter.unsqueeze(-1).expand_as(v_intra)
        return 1.0 + v_inter + v_intra


class Simple3DBlock(nn.Module):
    def __init__(self, cm):
        super().__init__()
        self.conv2d = nn.Conv2d(cm, cm, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(cm)
        self.relu = nn.ReLU()
        self.weight_gen = TimeVaryingWeightGenerator(cm)

    def forward(self, x):
        batch, cm, N, two, P2 = x.shape
        spatial = x.view(batch, cm, N, two * P2)
        alpha = self.weight_gen(x)
        y = self.conv2d(spatial)
        y = self.norm(y)
        alpha_exp = alpha.unsqueeze(-1)
        y = y * alpha_exp
        y = self.relu(y)
        y = y.view(batch, cm, N, two, P2)
        return y + x


class TVNetReconstructor(nn.Module):
    def __init__(self, in_channels, num_blocks=NUM_BLOCKS, patch_p=PATCH_P):
        super().__init__()
        self.embed = Simple3DEmbedding(in_channels, patch_p)
        self.cm = self.embed.cm
        self.blocks = nn.ModuleList([Simple3DBlock(self.cm) for _ in range(num_blocks)])
        self.reconstruct_conv = nn.Conv1d(self.cm, in_channels, kernel_size=1)

    def forward(self, x):
        emb = self.embed(x)
        y = emb
        for block in self.blocks:
            y = block(y)
        batch, cm, N, two, P2 = y.shape
        y_flat = y.view(batch, cm, N * two * P2)
        out = self.reconstruct_conv(y_flat)
        return out[:, :, :x.shape[-1]]


# TRAINING
def train_model(model, train_loader, val_loader):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val = float("inf")
    no_improve = 0
    history = {"train": [], "val": []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = criterion(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                loss = criterion(out, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        log(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_tvnet.pth"))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log("Early stopping triggered.")
                break

    # Save loss plot
    plt.figure()
    plt.plot(history["train"], label="Train MSE")
    plt.plot(history["val"], label="Val MSE")
    plt.legend()
    plt.title("Training vs Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "train_val_loss.png"))

    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_tvnet.pth")))
    return model


# EVALUATION
def compute_recon_errors(model, loader):
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    all_errs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            err = criterion(out, batch).mean(dim=(1, 2))
            all_errs.append(err.cpu().numpy())
    return np.concatenate(all_errs, axis=0)


def main():
    log("=== TVNet Anomaly Detection ===")
    train_data = load_csv(TRAIN_FILE)
    test_data = load_csv(TEST_FILE)
    train_loader, val_loader, test_loader = prepare_loaders(train_data, test_data)
    model = TVNetReconstructor(in_channels=train_data.shape[1])
    model = train_model(model, train_loader, val_loader)
    test_errs = compute_recon_errors(model, test_loader)

    if os.path.exists(TEST_LABEL_FILE):
        labels = pd.read_csv(TEST_LABEL_FILE).values.flatten()
        labels = pd.Series(labels).astype(str).str.strip().replace({
            "normal": 0, "Normal": 0, "NORMAL": 0,
            "abnormal": 1, "Anomaly": 1, "anomaly": 1, "ANOMALY": 1
        }).astype(float)
        labels = np.where(labels > 0.5, 1, 0).astype(int)
        labels = labels[:len(test_errs)]

        thresholds = np.linspace(test_errs.min(), test_errs.max(), 200)
        best_f1, best_t = 0, thresholds[0]
        f1_scores = []
        for t in thresholds:
            preds = (test_errs > t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
            f1_scores.append(f1)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        preds = (test_errs > best_t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        log(f"\nBest Threshold = {best_t:.6f}")
        log(f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

        results_df = pd.DataFrame({
            "reconstruction_error": test_errs,
            "predicted_anomaly": preds,
            "true_label": labels
        })
        results_df.to_csv(os.path.join(RESULT_DIR, "anomaly_results_with_labels.csv"), index=False)

        # Save plots
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores, label="F1 vs Threshold")
        plt.axvline(best_t, color="r", linestyle="--", label="Best Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "f1_vs_threshold.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(test_errs, label="Reconstruction Error")
        plt.axhline(best_t, color="r", linestyle="--", label="Best Threshold")
        plt.scatter(np.where(labels == 1)[0], test_errs[labels == 1], color="orange", label="True Anomalies")
        plt.legend()
        plt.title("Reconstruction Error and True Anomalies")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "anomaly_error_plot.png"))

        log("✅ Saved plots and results.")
    else:
        log("⚠️ test_label.csv not found — skipping F1 evaluation.")


if __name__ == "__main__":
    main()

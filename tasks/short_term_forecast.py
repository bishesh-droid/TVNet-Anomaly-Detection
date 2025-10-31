import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.tvnet import TVNet
from utils.metrics import mse, mae
from utils.logger import get_logger, save_checkpoint, save_json
from utils.seed import set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ForecastDataset(Dataset):
    """Create sliding windows for time-series forecasting."""
    def __init__(self, data, input_len, pred_len):
        self.data = data.astype(np.float32)
        self.input_len = input_len
        self.pred_len = pred_len
        self.length = len(data) - input_len - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq_x = self.data[idx: idx + self.input_len]
        seq_y = self.data[idx + self.input_len: idx + self.input_len + self.pred_len]
        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)
        return torch.from_numpy(seq_x), torch.from_numpy(seq_y)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for seq_x, seq_y in loader:
        seq_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        pred = model(seq_x)
        if pred.shape != seq_y.shape:
            pred = pred[:, -seq_y.shape[1]:, :]
        loss = criterion(pred, seq_y)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * seq_x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for seq_x, seq_y in loader:
        seq_x = seq_x.to(device)
        pred = model(seq_x)
        if pred.shape != seq_y.shape:
            pred = pred[:, -seq_y.shape[1]:, :]
        all_pred.append(pred.cpu())
        all_true.append(seq_y)
    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)
    return {"mse": mse(preds, trues).item(), "mae": mae(preds, trues).item()}


def read_time_series_file(path):
    """Reads .csv or .tsf file and returns numeric DataFrame (T, C)."""
    if path.endswith(".csv"):
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="ISO-8859-1")

    elif path.endswith(".tsf"):
        encodings = ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"]
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, errors="ignore") as f:
                    raw = f.readlines()
                break
            except Exception:
                continue

        lines = []
        for l in raw:
            line = l.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue
            lines.append(line)

        series_list = []
        for l in lines:
            if ":" in l:
                vals_str = l.split(":")[-1]
            else:
                vals_str = l
            try:
                vals = [float(v) for v in vals_str.split(",") if v.strip() != ""]
                series_list.append(vals)
            except ValueError:
                continue

        if not series_list:
            raise ValueError(f"No valid numeric series found in {path}")

        df = pd.DataFrame(series_list).T

    else:
        raise ValueError("Unsupported file format. Please use .csv or .tsf")

    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis=1, how="all")
    return df


# Normalization
def normalize_per_series(data, train_ratio=0.7):
    """Normalize each series (column) independently."""
    means, stds = [], []
    scaled = np.zeros_like(data)
    n = len(data)
    train_end = int(n * train_ratio)

    for i in range(data.shape[1]):
        series = data[:, i]
        train_part = series[:train_end]
        mean = np.nanmean(train_part)
        std = np.nanstd(train_part)
        if std < 1e-8:
            std = 1.0
        scaled[:, i] = (series - mean) / std
        means.append(mean)
        stds.append(std)
    return scaled, np.array(means), np.array(stds)


def main(args):
    set_seed(args.seed)
    logger = get_logger("short_forecast", log_dir=args.log_dir)
    device = torch.device(args.device)

    df = read_time_series_file(args.dataset_path)
    logger.info(f"Loaded {args.dataset_path} shape={df.shape}")

    # 2️⃣ Auto-detect large datasets
    if not args.full_dataset:
        max_series = min(200, df.shape[1])
        if df.shape[1] > max_series:
            logger.info(f"Dataset too large ({df.shape[1]} series). Using first {max_series} for faster training.")
            df = df.iloc[:, :max_series]
        else:
            logger.info("Using full dataset (small enough).")
    else:
        logger.info("Full dataset mode enabled. Using all series.")

    # 3️⃣ Split and normalize
    values = df.values.astype(np.float32)
    n = len(values)
    n_train, n_val = int(n * 0.7), int(n * 0.8)

    scaled_data, means, stds = normalize_per_series(values)
    train_data = scaled_data[:n_train]
    val_data = scaled_data[n_train:n_val]
    test_data = scaled_data[n_val:]

    # 4️⃣ Datasets
    train_ds = ForecastDataset(train_data, args.input_len, args.pred_len)
    val_ds = ForecastDataset(val_data, args.input_len, args.pred_len)
    test_ds = ForecastDataset(test_data, args.input_len, args.pred_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(f"Train/Val/Test sizes: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")

    # 5️⃣ Model
    _, C = values.shape
    model = TVNet(
        in_dim=C,
        Cm=args.Cm,
        num_blocks=args.num_blocks,
        patch_len=args.patch_len,
        kernel_size=args.kernel_size,
        task="forecast",
        out_len=args.pred_len,
        out_dim=C,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    val_mses, val_maes, train_losses = [], [], []

    # 6️⃣ Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["mse"])

        train_losses.append(train_loss)
        val_mses.append(val_metrics["mse"])
        val_maes.append(val_metrics["mae"])

        logger.info(f"[Epoch {epoch}] Train={train_loss:.6f} ValMSE={val_metrics['mse']:.6f}")
        if val_metrics["mse"] < best_val and not np.isnan(val_metrics["mse"]):
            best_val = val_metrics["mse"]
            save_path = os.path.join(
                args.save_dir, f"tvnet_short_best_{os.path.basename(args.dataset_path)}.pth"
            )
            save_checkpoint({"model_state": model.state_dict()}, save_path)
            logger.info(f"Saved best model → {save_path}")

    # 🔹 Plot validation error graphs
    plt.figure(figsize=(8,5))
    plt.plot(val_mses, label='Validation MSE', color='red')
    plt.plot(val_maes, label='Validation MAE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Short-Term Forecasting Error Graphs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("experiments/plots/short_term", exist_ok=True)
    plt.savefig(f"experiments/plots/short_term/{os.path.basename(args.dataset_path)}_error_graph.png")
    plt.close()

    # 7️⃣ Test
    best_ckpt = os.path.join(args.save_dir, f"tvnet_short_best_{os.path.basename(args.dataset_path)}.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f}")

    # 8️⃣ Save results
    save_json(
        {
            "dataset": args.dataset_path,
            "input_len": args.input_len,
            "pred_len": args.pred_len,
            "epochs": args.epochs,
            "val_mse": best_val,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "series_used": df.shape[1],
        },
        os.path.join(args.save_dir, f"short_forecast_results_{os.path.basename(args.dataset_path)}.json"),
    )
    logger.info("✅ Done!")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--input_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--Cm", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--patch_len", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints/short_term")
    parser.add_argument("--log_dir", type=str, default="experiments/logs/short_term")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--full_dataset", action="store_true")
    args = parser.parse_args()
    main(args)

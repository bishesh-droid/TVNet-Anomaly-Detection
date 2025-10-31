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
from utils.preprocess import StandardScaler1D
from utils.metrics import mse, mae
from utils.logger import get_logger, save_checkpoint, save_json
from utils.seed import set_seed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ForecastDataset(Dataset):
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
        return torch.from_numpy(seq_x), torch.from_numpy(seq_y)


# Train / Evaluate
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
        loss.backward()
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


# Main
def main(args):
    set_seed(args.seed)
    logger = get_logger("forecast", log_dir=args.log_dir)
    device = torch.device(args.device)

    # 1️⃣ Load dataset
    df = pd.read_csv(args.dataset_path)
    logger.info(f"Loaded {args.dataset_path} shape={df.shape}")

    df = df.drop(columns=["date", "Date", "time", "Time"], errors="ignore")
    values = df.values.astype(np.float32)

    n = len(values)
    n_train, n_val = int(n * 0.7), int(n * 0.8)
    train_data, val_data, test_data = values[:n_train], values[n_train:n_val], values[n_val:]

    scaler = StandardScaler1D()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    train_ds = ForecastDataset(train_data, args.input_len, args.pred_len)
    val_ds = ForecastDataset(val_data, args.input_len, args.pred_len)
    test_ds = ForecastDataset(test_data, args.input_len, args.pred_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(f"Train/Val/Test sizes: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")

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

    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    val_mses, val_maes, train_losses = [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_mses.append(val_metrics["mse"])
        val_maes.append(val_metrics["mae"])

        logger.info(f"[Epoch {epoch}] Train={train_loss:.6f}  ValMSE={val_metrics['mse']:.6f}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            path = os.path.join(args.save_dir, f"tvnet_forecast_best_{os.path.basename(args.dataset_path)}.pth")
            save_checkpoint({"model_state": model.state_dict()}, path)
            logger.info(f"Saved best checkpoint → {path}")

    plt.figure(figsize=(8,5))
    plt.plot(val_mses, label='Validation MSE', color='red')
    plt.plot(val_maes, label='Validation MAE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Long-Term Forecasting Error Graphs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("experiments/plots/long_term", exist_ok=True)
    plt.savefig(f"experiments/plots/long_term/{os.path.basename(args.dataset_path)}_error_graph.png")
    plt.close()

    best_ckpt = os.path.join(args.save_dir, f"tvnet_forecast_best_{os.path.basename(args.dataset_path)}.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test  MSE={test_metrics['mse']:.6f}  MAE={test_metrics['mae']:.6f}")

    save_json(
        {
            "dataset": args.dataset_path,
            "input_len": args.input_len,
            "pred_len": args.pred_len,
            "epochs": args.epochs,
            "val_mse": best_val,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
        },
        os.path.join(args.save_dir, f"forecast_results_{os.path.basename(args.dataset_path)}.json"),
    )
    logger.info("Done ✅")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=336)
    parser.add_argument("--Cm", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--patch_len", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints/long_term")
    parser.add_argument("--log_dir", type=str, default="experiments/logs/long_term")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()
    main(args)

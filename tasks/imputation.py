import os
import argparse
import math
import random
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


BASE_DIR = "/home/duffer/Documents/Coding/TVNet-Project"
DEFAULT_DATA_FOLDER = os.path.join("data", "imputation")  # relative to BASE_DIR
EXP_DIR = os.path.join(BASE_DIR, "experiments")
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
LOG_DIR = os.path.join(EXP_DIR, "logs")
PLOT_DIR = os.path.join(EXP_DIR, "plots")
RESULT_DIR = os.path.join(EXP_DIR, "results")

for d in (CKPT_DIR, LOG_DIR, PLOT_DIR, RESULT_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_PATIENCE = 3
DEFAULT_NUM_BLOCKS = 3
DEFAULT_Cm = 64
DEFAULT_PATCH_LEN = 1  
DEFAULT_BATCH = 64
DEFAULT_MASK_RATIO = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ett_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] > 1:
        # Drop first col (time) if non-numeric
        first_col = df.columns[0]
        if not np.issubdtype(df[first_col].dtype, np.number):
            df = df.drop(columns=[first_col])
    # keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    # forward/backfill then fill remaining with 0
    df = df.ffill().bfill().fillna(0.0)
    return df


class ImputationWindowDataset(Dataset):
   
    def __init__(self, arr: np.ndarray, window_len: int = 96, mask_ratio: float = 0.25, mode: str = "train"):
        # arr: (T, C)
        self.arr = arr.astype(np.float32)
        self.window_len = window_len
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.T, self.C = self.arr.shape
        self.num_windows = max(0, self.T - self.window_len + 1)

    def __len__(self):
        return self.num_windows

    def _random_mask(self, shape):
        m = np.ones(shape, dtype=np.float32)
        mask = (np.random.rand(*shape) >= self.mask_ratio).astype(np.float32)
        return mask

    def __getitem__(self, idx):
        start = idx
        win = self.arr[start: start + self.window_len]  # (L, C)
        win = win.T  # (C, L)
        if self.mode == "train":
            mask = self._random_mask(win.shape)
        else:
            mask = self._random_mask(win.shape)
        x_full = torch.from_numpy(win.copy())
        mask_t = torch.from_numpy(mask)
        x_masked = x_full * mask_t
        return x_full, x_masked, mask_t


class Simple3DEmbedding(nn.Module):
    
    def __init__(self, in_channels: int, Cm: int = DEFAULT_Cm, patch_len: int = DEFAULT_PATCH_LEN):
        super().__init__()
        self.in_channels = in_channels
        self.Cm = Cm
        self.patch_len = patch_len
        # channel embed as conv1d with kernel 1 (per-time linear)
        self.channel_embed = nn.Conv1d(in_channels, Cm, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        emb = self.channel_embed(x)  # (B, Cm, L)
        B, Cm, L = emb.shape
        # patch_len == 1 -> N = L, each patch length = 1
        # create shape (B, Cm, N, 1, 1)
        patches = emb.unsqueeze(2).unsqueeze(-1)  # (B, Cm, 1, L, 1)
        # we want (B, Cm, N, 1, 1) where N = L -> permute
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()  # (B, Cm, L, 1, 1)
        return patches


class TimeVaryingWeightGenerator(nn.Module):
    def __init__(self, Cm: int):
        super().__init__()
        self.fintra = nn.Conv1d(Cm, Cm, kernel_size=1)
        self.finter = nn.Conv1d(Cm, Cm, kernel_size=1)
        self.bn = nn.BatchNorm1d(Cm)
        self.act = nn.ReLU()

    def forward(self, Xemb):
        vintra = Xemb.mean(dim=(3, 4))  # (B, Cm, N)
        v_intra = self.act(self.bn(self.fintra(vintra)))
        vinter = v_intra.mean(dim=2, keepdim=True)  # (B, Cm, 1)
        v_inter = self.finter(vinter).squeeze(-1)  # (B, Cm)
        v_inter = v_inter.unsqueeze(-1).expand_as(v_intra)  # (B, Cm, N)
        alpha = 1.0 + v_intra + v_inter
        return alpha  # (B, Cm, N)


class Simple3DBlock(nn.Module):
    def __init__(self, Cm: int):
        super().__init__()
        self.Cm = Cm
        self.conv2d = nn.Conv2d(Cm, Cm, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(Cm)
        self.act = nn.ReLU()
        self.alpha_gen = TimeVaryingWeightGenerator(Cm)

    def forward(self, x):
        # x: (B, Cm, N, 1, 1) -> spatial = (N, 1)
        B, Cm, N, two, P2 = x.shape  # two=1, P2=1 here
        spatial = x.view(B, Cm, N, two * P2)  # (B, Cm, N, 1)
        y = self.conv2d(spatial)  # (B, Cm, N, 1)
        y = self.bn(y)
        alpha = self.alpha_gen(x).unsqueeze(-1)  # (B, Cm, N, 1)
        y = y * alpha
        y = self.act(y)
        y = y.view(B, Cm, N, two, P2)
        return x + y


class TVNetImputer(nn.Module):
    def __init__(self, in_channels: int, Cm: int = DEFAULT_Cm, num_blocks: int = DEFAULT_NUM_BLOCKS, patch_len: int = DEFAULT_PATCH_LEN):
        super().__init__()
        self.embed = Simple3DEmbedding(in_channels, Cm=Cm, patch_len=patch_len)
        self.Cm = Cm
        self.blocks = nn.ModuleList([Simple3DBlock(Cm) for _ in range(num_blocks)])
        # head: map Cm back to input channels per time step
        self.head = nn.Conv1d(Cm, in_channels, kernel_size=1)

    def forward(self, x_masked):
        # x_masked: (B, C, L)
        emb = self.embed(x_masked)  # (B, Cm, N, 1, 1)
        y = emb
        for blk in self.blocks:
            y = blk(y)
        B, Cm, N, _, _ = y.shape
        y_flat = y.view(B, Cm, N)  # (B, Cm, L)
        out = self.head(y_flat)  # (B, C, L)
        return out


# Utilities: metrics and saving
def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


# Training loop
def train_imputer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    ckpt_path: str,
    log_path: str,
    plot_path: str,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    no_improve = 0

    train_history = {"mse": [], "mae": []}
    val_history = {"mse": [], "mae": []}

    with open(log_path, "w") as flog:
        flog.write("epoch,train_mse,train_mae,val_mse,val_mae\n")

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        n_samples = 0
        for x_full, x_masked, mask in train_loader:
            x_full = x_full.to(DEVICE)  # (B, C, L)
            x_masked = x_masked.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(x_masked)  # (B, C, L)
            # compute loss only on masked positions: we want to impute them correctly
            loss = criterion(pred * (1 - mask), x_full * (1 - mask))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_mse = ((pred.detach().cpu().numpy() - x_full.detach().cpu().numpy()) ** 2).mean()
            batch_mae = (np.abs(pred.detach().cpu().numpy() - x_full.detach().cpu().numpy())).mean()
            bsz = x_full.shape[0]
            running_loss += batch_mse * bsz
            running_mae += batch_mae * bsz
            n_samples += bsz

        train_mse = running_loss / n_samples
        train_mae = running_mae / n_samples

        # validation
        model.eval()
        v_loss = 0.0
        v_mae = 0.0
        v_samples = 0
        with torch.no_grad():
            for x_full, x_masked, mask in val_loader:
                x_full = x_full.to(DEVICE)
                x_masked = x_masked.to(DEVICE)
                mask = mask.to(DEVICE)
                pred = model(x_masked)
                batch_mse = ((pred.cpu().numpy() - x_full.cpu().numpy()) ** 2).mean()
                batch_mae = (np.abs(pred.cpu().numpy() - x_full.cpu().numpy())).mean()
                bsz = x_full.shape[0]
                v_loss += batch_mse * bsz
                v_mae += batch_mae * bsz
                v_samples += bsz
        val_mse = v_loss / v_samples
        val_mae = v_mae / v_samples

        train_history["mse"].append(train_mse)
        train_history["mae"].append(train_mae)
        val_history["mse"].append(val_mse)
        val_history["mae"].append(val_mae)

        # log
        line = f"{ep},{train_mse:.6e},{train_mae:.6e},{val_mse:.6e},{val_mae:.6e}"
        print(f"[Epoch {ep}] train_mse={train_mse:.6e}, train_mae={train_mae:.6e} | val_mse={val_mse:.6e}, val_mae={val_mae:.6e}")
        with open(log_path, "a") as flog:
            flog.write(line + "\n")

        # early stop on val_mse
        if val_mse < best_val - 1e-9:
            best_val = val_mse
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    plt.figure(figsize=(8, 4))
    plt.plot(train_history["mse"], label="train_mse")
    plt.plot(val_history["mse"], label="val_mse")
    plt.title("MSE (Train / Val)")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path.replace(".png", "_mse.png"))

    plt.figure(figsize=(8, 4))
    plt.plot(train_history["mae"], label="train_mae")
    plt.plot(val_history["mae"], label="val_mae")
    plt.title("MAE (Train / Val)")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path.replace(".png", "_mae.png"))

    return model, train_history, val_history


# Evaluation on test set
def evaluate_on_test(model: nn.Module, test_loader: DataLoader, ckpt_path: str, results_csv: str, plot_path: str):
    # load best
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    all_errs = []
    rows = []
    with torch.no_grad():
        for x_full, x_masked, mask in test_loader:
            x_full = x_full.to(DEVICE)
            x_masked = x_masked.to(DEVICE)
            pred = model(x_masked)  # (B, C, L)
            pred_np = pred.cpu().numpy()
            full_np = x_full.cpu().numpy()
            bsz = pred_np.shape[0]
            for i in range(bsz):
                mse_val = ((pred_np[i] - full_np[i]) ** 2).mean()
                mae_val = np.abs(pred_np[i] - full_np[i]).mean()
                all_errs.append((mse_val, mae_val))
                rows.append({
                    "mse": mse_val,
                    "mae": mae_val
                })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(results_csv, index=False)

    mse_vals = df_res["mse"].values
    mae_vals = df_res["mae"].values
    plt.figure(figsize=(8, 4))
    plt.plot(mse_vals, label="window_mse")
    plt.title("Test window MSE (per window)")
    plt.xlabel("window index")
    plt.ylabel("mse")
    plt.tight_layout()
    plt.savefig(plot_path.replace(".png", "_test_mse.png"))

    plt.figure(figsize=(8, 4))
    plt.plot(mae_vals, label="window_mae")
    plt.title("Test window MAE (per window)")
    plt.xlabel("window index")
    plt.ylabel("mae")
    plt.tight_layout()
    plt.savefig(plot_path.replace(".png", "_test_mae.png"))

    return df_res


# CLI / main
def find_dataset_file(dataset_folder: str, dataset_name: str) -> str:
    # try folder/dataset_name.csv, then dataset_name.csv in current dir
    candidate = os.path.join(dataset_folder, f"{dataset_name}.csv")
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join(dataset_folder, f"{dataset_name}.CSV")
    if os.path.exists(candidate2):
        return candidate2
    # fallback search
    for f in os.listdir(dataset_folder):
        if dataset_name.lower() in f.lower() and f.lower().endswith(".csv"):
            return os.path.join(dataset_folder, f)
    raise FileNotFoundError(f"Dataset {dataset_name} not found under {dataset_folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default=DEFAULT_DATA_FOLDER, help="folder with ETTh1.csv etc (relative to project root)")
    parser.add_argument("--dataset_name", type=str, default="ETTh1", help="ETTh1, ETTh2, ETTm1, ETTm2")
    parser.add_argument("--window_len", type=int, default=96, help="window length (L)")
    parser.add_argument("--mask_ratio", type=float, default=DEFAULT_MASK_RATIO)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--cm", type=int, default=DEFAULT_Cm)
    parser.add_argument("--num_blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--patch_len", type=int, default=DEFAULT_PATCH_LEN)
    args = parser.parse_args()

    # paths
    dataset_folder = os.path.join(BASE_DIR, args.dataset_folder) if not os.path.isabs(args.dataset_folder) else args.dataset_folder
    ds_file = find_dataset_file(dataset_folder, args.dataset_name)
    df = load_ett_csv(ds_file)
    arr = df.values  # (T, C)
    T, C = arr.shape
    print(f"Loaded {ds_file} shape={arr.shape}")

    # split 70/10/20 by time
    n_train = int(0.7 * T)
    n_val = int(0.1 * T)
    n_test = T - n_train - n_val
    train_arr = arr[:n_train]
    val_arr = arr[n_train:n_train + n_val]
    test_arr = arr[n_train + n_val:]

    # datasets
    train_ds_full = ImputationWindowDataset(train_arr, window_len=args.window_len, mask_ratio=args.mask_ratio, mode="train")
    val_ds = ImputationWindowDataset(val_arr, window_len=args.window_len, mask_ratio=args.mask_ratio, mode="val")
    test_ds = ImputationWindowDataset(test_arr, window_len=args.window_len, mask_ratio=args.mask_ratio, mode="test")

    # dataloaders
    train_loader = DataLoader(train_ds_full, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # model
    model = TVNetImputer(in_channels=C, Cm=args.cm, num_blocks=args.num_blocks, patch_len=args.patch_len)
    model.to(DEVICE)

    # paths for outputs
    safe_name = args.dataset_name
    ckpt_path = os.path.join(CKPT_DIR, f"imputation_{safe_name}.pth")
    log_path = os.path.join(LOG_DIR, f"imputation_{safe_name}.log")
    plot_path = os.path.join(PLOT_DIR, f"imputation_{safe_name}.png")
    results_csv = os.path.join(RESULT_DIR, f"imputation_{safe_name}_test_results.csv")

    # train
    model, tr_hist, val_hist = train_imputer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        ckpt_path=ckpt_path,
        log_path=log_path,
        plot_path=plot_path,
    )

    # evaluate on test
    df_res = evaluate_on_test(model, test_loader, ckpt_path=ckpt_path, results_csv=results_csv, plot_path=plot_path)

    print("Saved checkpoint:", ckpt_path)
    print("Saved plots and results under experiments/")

if __name__ == "__main__":
    main()

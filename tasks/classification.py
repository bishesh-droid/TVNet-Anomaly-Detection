import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff
from models.tvnet import TVNet



def load_csv_file(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    if not np.issubdtype(df.iloc[:, -1].dtype, np.number):
        le = LabelEncoder()
        df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    return X, y


def load_arff_file(file_path):
    """
    Robust loader for ARFF files from UEA/UCR datasets.
    Handles nested structured arrays like EthanolConcentration, FaceDetection, etc.
    """
    data, meta = arff.loadarff(file_path)
    data = np.asarray(data)

    if data.dtype.names is not None:
        fields = data.dtype.names
        X, y = [], []

        for row in data:
            features = []
            for f in fields[:-1]:
                val = row[f]
                if isinstance(val, (np.void, np.ndarray)) and hasattr(val, "dtype") and val.dtype.names is not None:
                    # Structured sequence: [(t1, ...), (t2, ...)]
                    vals = [float(val[n]) for n in val.dtype.names]
                    features.extend(vals)
                elif isinstance(val, (np.ndarray, list, tuple)):
                    features.extend([float(v) for v in val])
                else:
                    try:
                        features.append(float(val))
                    except Exception:
                        pass
            X.append(features)
            y.append(row[fields[-1]])

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
    else:
        # Simple numeric array
        X = np.array(data[:, :-1], dtype=np.float32)
        y = data[:, -1]

    # Label encoding
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(np.int64)

    return X, y


def load_ts_file(file_path):
    """Loads .ts time series classification files."""
    X, y = [], []
    reading_data = False
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue
            if "@data" in line.lower():
                reading_data = True
                continue
            if reading_data:
                parts = line.split(":")
                if len(parts) == 2:
                    label, values = parts
                    vals = [float(v) for v in values.split(",") if v.strip()]
                    X.append(vals)
                    y.append(label.strip())
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = np.array(X, dtype=np.float32)
    return X, y


def load_data_auto(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return load_csv_file(file_path)
    elif ext == ".arff":
        return load_arff_file(file_path)
    elif ext == ".ts":
        return load_ts_file(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# Dataset

class UniversalClassificationDataset(Dataset):
    def __init__(self, file_path):
        X, y = load_data_auto(file_path)

        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        # (batch, L, C)
        if self.X.dim() == 2:
            self.X = self.X.unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Training

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# 🚀 Main

import matplotlib.pyplot as plt  # ⬅️ add this

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = UniversalClassificationDataset(args.train_file)
    test_dataset = UniversalClassificationDataset(args.test_file)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    in_dim = train_dataset.X.shape[-1]
    num_classes = len(torch.unique(train_dataset.y))

    model = TVNet(
        in_dim=in_dim,
        Cm=args.Cm,
        num_blocks=args.num_blocks,
        patch_len=args.patch_len,
        task="classify",
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[INFO] Training {args.dataset_name} | Classes={num_classes} | Device={device}")

    train_losses, test_accs = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        test_accs.append(acc)
        print(f"[Epoch {epoch:02d}] TrainLoss={train_loss:.4f} TestAcc={acc*100:.2f}%")

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(test_accs, label='Test Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f"Training Progress – {args.dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("experiments/plots/classification", exist_ok=True)
    plt.savefig(f"experiments/plots/classification/{args.dataset_name}_error_graph.png")
    plt.close()

    os.makedirs("experiments/checkpoints/classification", exist_ok=True)
    save_path = f"experiments/checkpoints/classification/tvnet_classify_{args.dataset_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved → {save_path}")



# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVNet Classification Task")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--Cm", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=8)
    args = parser.parse_args()

    main(args)

# utils/data_loader.py
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class TimeSeriesWindowDataset(Dataset):
    """
    Simple sliding window dataset for forecasting.
    Expects CSV files with time in rows and columns as features (no index handling here).
    """
    def __init__(self, csv_path, input_len=96, output_len=96, target_cols=None, normalize=True):
        df = pd.read_csv(csv_path)
        data = df.values.astype('float32')  # shape (T, C)
        if normalize:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-6
            self.data = (data - mean) / std
            self._mean = mean
            self._std = std
        else:
            self.data = data
            self._mean = None
            self._std = None
        self.input_len = input_len
        self.output_len = output_len
        self.T, self.C = self.data.shape
        self.target_cols = target_cols if target_cols is not None else list(range(self.C))

    def __len__(self):
        return max(0, self.T - self.input_len - self.output_len + 1)

    def __getitem__(self, idx):
        s = idx
        x = self.data[s:s + self.input_len]  # (L, C)
        y = self.data[s + self.input_len: s + self.input_len + self.output_len, self.target_cols]  # (out_len, out_dim)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloader(csv_path, batch_size=32, input_len=96, output_len=96, shuffle=True, num_workers=0):
    ds = TimeSeriesWindowDataset(csv_path, input_len=input_len, output_len=output_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

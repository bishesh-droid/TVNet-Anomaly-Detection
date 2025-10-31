# utils/preprocess.py
"""
Simple preprocessing utilities:
- normalize: fit/transform
- sliding window generator helpers
- mask generation for imputation experiments
"""
import numpy as np

class StandardScaler1D:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        # data: (T, C)
        self.mean_ = np.nanmean(data, axis=0, keepdims=True)
        self.std_ = np.nanstd(data, axis=0, keepdims=True) + 1e-6

    def transform(self, data):
        return (data - self.mean_) / self.std_

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.std_ + self.mean_

def sliding_windows(data, input_len, output_len, step=1):
    """
    Generator of windows for data (T, C)
    Yields (x, y) where x shape (input_len, C), y shape (output_len, C)
    """
    T = data.shape[0]
    end = T - input_len - output_len + 1
    for s in range(0, end, step):
        x = data[s:s+input_len]
        y = data[s+input_len:s+input_len+output_len]
        yield x, y

def random_mask(batch_shape, mask_ratio):
    """
    Returns boolean mask of shape batch_shape where True indicates observed, False indicates missing.
    batch_shape: (batch, timesteps, features)
    """
    b, t, c = batch_shape
    mask = np.random.rand(b, t, c) >= mask_ratio
    return mask.astype(bool)

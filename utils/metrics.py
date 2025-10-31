# utils/metrics.py
import torch

def mse(pred, true):
    """
    Compute Mean Squared Error between predictions and targets.
    Works with both torch tensors and numpy arrays.
    """
    if isinstance(pred, torch.Tensor):
        return torch.mean((pred - true) ** 2)
    else:
        import numpy as np
        return np.mean((pred - true) ** 2)

def mae(pred, true):
    """
    Compute Mean Absolute Error between predictions and targets.
    Works with both torch tensors and numpy arrays.
    """
    if isinstance(pred, torch.Tensor):
        return torch.mean(torch.abs(pred - true))
    else:
        import numpy as np
        return np.mean(np.abs(pred - true))

def smape(pred, true):
    return torch.mean(2 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))

def mase(pred, true, naive_forecast):
    return torch.mean(torch.abs(pred - true)) / torch.mean(torch.abs(true - naive_forecast))

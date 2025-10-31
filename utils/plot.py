# utils/plot.py
"""
Plot helpers (matplotlib). These are intentionally minimal—save figures to disk.
"""
import matplotlib.pyplot as plt
import os

def plot_series(true, pred, out_path=None, title="prediction vs truth", n_series=1):
    """
    true, pred: numpy arrays (T, C) or (T,)
    """
    if true.ndim == 2:
        C = true.shape[1]
    else:
        C = 1
    plt.figure(figsize=(10, 4))
    if C == 1:
        plt.plot(true, label="true")
        plt.plot(pred, label="pred")
    else:
        # plot first n_series columns
        for i in range(min(C, n_series)):
            plt.plot(true[:, i], label=f"true_{i}")
            plt.plot(pred[:, i], label=f"pred_{i}", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
    plt.close()

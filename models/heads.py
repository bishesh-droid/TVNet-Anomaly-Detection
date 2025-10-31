import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastHead(nn.Module):
    def __init__(self, Cm, out_len, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(Cm, out_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(out_len, out_len)

        self.out_len = out_len
        self.out_dim = out_dim

    def forward(self, x):
        if x.dim() > 3:
            x = x.mean(dim=-1)
        out = self.fc(x)
        out = F.adaptive_avg_pool1d(out, self.out_len)
        out = out.transpose(1, 2)  # (B, out_len, out_dim)
        return out


class ImputationHead(nn.Module):
    def __init__(self, Cm, out_len, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(Cm, out_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.out_len = out_len
        self.out_dim = out_dim

    def forward(self, x):
        if x.dim() > 3:
            x = x.mean(dim=-1)
        out = self.fc(x)  # (B, out_dim, N)
        out = out.transpose(1, 2)  # (B, N, out_dim)
        return out


class ClassificationHead(nn.Module):
    def __init__(self, Cm, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(Cm, num_classes)
        )

    def forward(self, x):
        if x.dim() > 3:
            x = x.mean(dim=-1)
        return self.fc(x)


class ReconstructionHead(nn.Module):
    def __init__(self, Cm, out_len, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(Cm, out_dim, kernel_size=1),
            nn.ReLU()
        )
        self.out_len = out_len
        self.out_dim = out_dim

    def forward(self, x):
        if x.dim() > 3:
            x = x.mean(dim=-1)
        out = self.fc(x)
        out = out.transpose(1, 2)  # (B, N, out_dim)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_conv import DynamicConv

class TV3DBlock(nn.Module):
    def __init__(self, Cm, kernel_size=3):
        super().__init__()
        self.alpha_gen = DynamicConv(Cm)
        self.conv = nn.Sequential(
            nn.Conv1d(Cm, Cm, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(Cm),
            nn.ReLU(),
            nn.Conv1d(Cm, Cm, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(Cm)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = x

        if out.dim() > 3:
            out = out.flatten(start_dim=2)
        if residual.dim() > 3:
            residual = residual.flatten(start_dim=2)

        alpha = self.alpha_gen(out)  # (B, C, L)
        out = self.conv(out)
        out = out * alpha

        if out.shape[-1] != residual.shape[-1]:
            residual = F.interpolate(residual, size=out.shape[-1], mode='nearest')

        out = out + residual
        out = self.relu(out)
        return out

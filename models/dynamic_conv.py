import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConv(nn.Module):

    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

        self.fintra = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
        )

        self.norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        while x.dim() > 3:
            x = x.mean(dim=-1)  # pool extra dims safely

        # Input must be (B, Cm, N)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Intra-feature processing
        fintra_out = self.fintra(x)  # (B, Cm, N)

        # Temporal feature extraction
        out = self.conv(fintra_out)
        out = self.norm(out)
        out = self.activation(out)
        return out

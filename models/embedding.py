import torch
import torch.nn as nn
import torch.nn.functional as F


class Patch3DEmbedding(nn.Module):

    def __init__(self, in_channels, embed_dim, patch_len=24):
        super().__init__()
        self.patch_len = patch_len
        self.embed_dim = embed_dim

        self.feature_proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(embed_dim)

        self.patch_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=patch_len,
            stride=patch_len,
        )

    def forward(self, x):

        L = x.shape[-1]
        if L < self.patch_len:
            pad_len = self.patch_len - L
            x = F.pad(x, (0, pad_len), mode="replicate")

        x = self.feature_proj(x)  # (B, embed_dim, L)
        x = self.norm(x)

        x_patches = self.patch_conv(x)  # (B, embed_dim, N)
        B, Cm, N = x_patches.shape

        if N == 1:
            x_patches = torch.cat([x_patches, x_patches.clone()], dim=2)
            N = 2

        if N % 2 != 0:
            pad = x_patches[:, :, -1:].clone()
            x_patches = torch.cat([x_patches, pad], dim=2)
            N = x_patches.shape[-1]

        half = N // 2
        x_even = x_patches[:, :, :half].unsqueeze(-2)  # (B, Cm, half, 1)
        x_odd = x_patches[:, :, half:].unsqueeze(-2)   # (B, Cm, half, 1)

        if x_even.shape[2] != x_odd.shape[2]:
            diff = abs(x_even.shape[2] - x_odd.shape[2])
            pad = (
                x_even[:, :, -1:, :].clone()
                if x_even.shape[2] < x_odd.shape[2]
                else x_odd[:, :, -1:, :].clone()
            )
            if x_even.shape[2] < x_odd.shape[2]:
                x_even = torch.cat([x_even, pad.repeat(1, 1, diff, 1)], dim=2)
            else:
                x_odd = torch.cat([x_odd, pad.repeat(1, 1, diff, 1)], dim=2)

        x3d = torch.cat([x_even, x_odd], dim=-2)  # (B, Cm, N/2, 2)
        return x3d

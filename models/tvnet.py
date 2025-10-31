import torch
import torch.nn as nn
from .embedding import Patch3DEmbedding
from .blocks import TV3DBlock
from .heads import ForecastHead, ClassificationHead, ImputationHead, ReconstructionHead


class TVNet(nn.Module):
    def __init__(self, in_dim, Cm=64, num_blocks=3, patch_len=24, kernel_size=3,
                 task='forecast', out_len=96, out_dim=None, num_classes=None):
        
        super().__init__()
        self.in_dim = in_dim
        self.Cm = Cm
        self.patch_len = patch_len
        self.task = task

        # Embedding layer
        self.embedding = Patch3DEmbedding(in_channels=in_dim, embed_dim=Cm, patch_len=patch_len)

        # Stacked TV3D blocks
        self.blocks = nn.ModuleList([TV3DBlock(Cm, kernel_size=kernel_size) for _ in range(num_blocks)])

        # Select head based on task
        if task in ['forecast', 'short_forecast']:
            assert out_dim is not None
            self.head = ForecastHead(Cm, out_len, out_dim)
        elif task in ['impute', 'imputation']:
            assert out_dim is not None
            self.head = ImputationHead(Cm, out_len, out_dim)
        elif task == 'classify':
            assert num_classes is not None
            self.head = ClassificationHead(Cm, num_classes)
        elif task == 'reconstruct':
            assert out_dim is not None and out_len is not None
            self.head = ReconstructionHead(Cm, out_len, out_dim)
        else:
            raise ValueError(f"Unknown task '{task}'. Valid options: "
                             f"['forecast', 'short_forecast', 'impute', 'imputation', 'classify', 'reconstruct']")

    def forward(self, x):
        """
        x: (batch, L, C) or (batch, C, L)
        """
        # Ensure input format (B, C, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[1] != self.in_dim:
            x = x.permute(0, 2, 1)

        xemb = self.embedding(x)  # (B, Cm, N, 2, P/2)
        x3d = xemb

        for blk in self.blocks:
            x3d = blk(x3d)

        out = self.head(x3d)
        return out

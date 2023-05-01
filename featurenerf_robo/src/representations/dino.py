import torch.nn as nn
from .utils_dino.dino import DINO


class DINOEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dino = DINO().cuda()

    def forward(self, x):
        feature = self.dino(x)
        return feature.flatten(1)

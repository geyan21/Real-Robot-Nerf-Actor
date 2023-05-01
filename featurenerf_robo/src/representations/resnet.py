import torch
from torchvision.models import resnet18, resnet34, resnet50
import torch.nn as nn


class ResNet18Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = resnet18(pretrained=True).cuda()
        self.freeze_encoder = cfg.freeze_encoder
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        return x


class ResNet34Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = resnet34(pretrained=True).cuda()
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        return self.encoder(x)


class ResNet50Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = resnet50(pretrained=True).cuda()
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        return self.encoder(x)
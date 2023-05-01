from .mvp import MVPEncoder
from .pixelnerf import PixelNeRF, PixelNeRFEncoder
from .featurenerf import FeatureNeRF
from .resnet import ResNet18Encoder, ResNet34Encoder, ResNet50Encoder
from .dino import DINOEncoder
from .pri3d import Pri3DEncoder
from .mocov2 import MoCov2Encoder
from .pointnet import PointNetEncoder
from .pointnet2 import PointNet2Encoder
from .pointnerf import PointNeRF

import torch
import torch.nn as nn
import os
from termcolor import colored

class ZeroEncoder(torch.nn.Module):
    """
    test the nonsense encoder
    """
    def __init__(self, cfg):
        super(ZeroEncoder, self).__init__()
        self.fc = torch.nn.Linear(1,1)
        self.output_dim = 32

    def forward(self, x):
        return torch.zeros((x.shape[0], self.output_dim)).cuda()

class SimpleEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super(SimpleEncoder, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ).cuda()

        
    def forward(self, x):
        return self.convnet(x).flatten(1)


encoder_dict = {
    # nerf class
    "pixelnerf": PixelNeRFEncoder,
    "pixelnerf+dino": PixelNeRFEncoder,
    "featurenerf": FeatureNeRF,

    # transformer class
    "dino": DINOEncoder,
    "mvp": MVPEncoder,

    # resnet class
    "resnet18": ResNet18Encoder,
    "resnet34": ResNet34Encoder,
    "resnet50": ResNet50Encoder,
    "imgnet": ResNet50Encoder,
    "pri3d": Pri3DEncoder,
    "mocov2": MoCov2Encoder,

    # random vector
    "zero": ZeroEncoder,

    # simple conv
    "simple": SimpleEncoder,

    # pointnet class
    "pointnet": PointNetEncoder,
    "pointnet2": PointNet2Encoder,
    "pointnerf": PointNeRF,
    
    
}

def make_embedding(model_name, cfg, use_3D=False):
    if model_name not in encoder_dict:
        raise NotImplementedError(f"Encoder {model_name} is not implemented")
    encoder = encoder_dict[model_name](cfg)
    encoder.cuda()

    if model_name=="pointnerf":
        dummy_input2d = torch.zeros((2, 3, cfg.image_size, cfg.image_size)).cuda()
        dummy_input3d = torch.zeros((2, 3, cfg.num_points)).cuda()
        pose = torch.zeros((2, 4, 4)).cuda()
        focal = torch.zeros((2)).cuda()
        dummy_output = encoder(obs2d=dummy_input2d, obs3d=dummy_input3d, pose=pose, focal=focal)
        encoder.output_dim = dummy_output.shape[-1]
    elif not use_3D:
        # compute output dim
        dummy_input = torch.zeros((1, 3, cfg.image_size, cfg.image_size)).cuda()
        dummy_output = encoder(dummy_input)
        encoder.output_dim = dummy_output.shape[-1]
    else:
        # compute output dim
        dummy_input = torch.zeros((2, 3, cfg.num_points)).cuda() # B, D, N
        dummy_output = encoder(dummy_input)
        encoder.output_dim = dummy_output.shape[-1]

    print(colored(f"[make_embedding] Embedding output dim: {encoder.output_dim}", "cyan"))
    return encoder



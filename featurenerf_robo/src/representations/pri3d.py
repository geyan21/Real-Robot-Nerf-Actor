import torch.nn as nn
import torch
from .utils_pri3d import build_backbone
from termcolor import colored

class Pri3DEncoder(nn.Module):
    """
    [ICCV'21] Pri3D: Can 3D Priors Help 2D Representation Learning?
    github: https://github.com/Sekunde/Pri3D
    """
    def __init__(self, cfg):
        super(Pri3DEncoder, self).__init__()
        backbone_output_channels = {
            'Res18UNet': 32,
            'Res50UNet': 128,
            'ResNet18': 32,
            'Res18UNetMultiRes': 32,
        }
        self.backbone = build_backbone("Res50UNet", 
                                       backbone_output_channels["Res50UNet"],
                                       True)
        # self.backbone3d = build_backbone('ResUNet3D', 96, False)
        ckpt_path = "ckpts/pri3d.pth"
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["model"], strict=False) # just load backbone
        print(colored("Pri3D backbone loaded from {}".format(ckpt_path), "green"))

    
    def forward(self, x):
        feature = self.backbone(x)
        # x = self.backbone3d(x)
        feature_avg = feature.mean(1).reshape(x.shape[0], -1) # tmp usage of 2d feature
        return feature_avg

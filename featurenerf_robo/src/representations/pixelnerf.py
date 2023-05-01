import torch.nn as nn
import os
import torch
from termcolor import colored
import numpy as np

from .utils_pixelnerf.render import NeRFEmbedRenderer
from .utils_pixelnerf.model import make_model
from .utils_pixelnerf.util import gen_rays

from analysis.image import save_feature_map, save_rgb_image

class PixelNeRFEncoder(nn.Module):
    """
    pixelnerf with/without dino 
    """
    def __init__(self, cfg):
        super().__init__()
        net = make_model(cfg.pixelnerf_model).cuda()

        # two forward mode: 1. spatial 2. global
        self.pixelnerf_mode = cfg.pixelnerf_mode
        
        # choose model
        if cfg.embedding_name == "pixelnerf+dino":
            ckpt_path = "ckpts/pixelnerf_dino_r50"
        else:
            ckpt_path = "ckpts/pixelnerf_r50"

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            info = net.load_state_dict(ckpt, strict=False)
            print(info)
            print(colored(f"Loaded encoder from {ckpt_path}", 'cyan'))
        else:
            print(colored(f"The encoder is not loaded from your ckpt path.", 'red'))

        # only use the encoder part of pixelnerf
        self.encoder = net.encoder.cuda() 

        if self.pixelnerf_mode == "global":
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).cuda()
        elif self.pixelnerf_mode == "spatial":
            self.conv = nn.Conv2d(3904, 3, 1, 1, 0).cuda()
        elif self.pixelnerf_mode == "shallow":
            self.conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False).cuda()
        

        # resnet50 avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_spatial(self, x):
        with torch.no_grad():
            feature = self.encoder(x) # [B, 3, 128, 128] -> [B, 3904, 64, 64]
        feature = self.conv(feature) # [B, 3904, 64, 64] -> [B, 3, 64, 64]
        feature = torch.flatten(feature, start_dim=1) # [B, 3, 64, 64] -> [B, 3*64*64]
        return feature
    

    def forward_global(self, x):
        """
        x: [1,3,128,128]
        """
        with torch.no_grad():
            # feature = self.encoder.model(x)
            feature = self.encoder.model.conv1(x) # [1, 64, 64, 64]
            feature = self.encoder.model.bn1(feature)
            feature = self.encoder.model.relu(feature)
            feature = self.encoder.model.maxpool(feature) # [1, 64, 32, 32]
            feature = self.encoder.model.layer1(feature) # [1, 256, 32, 32]
            feature = self.encoder.model.layer2(feature) # [1, 512, 16, 16]
            feature = self.encoder.model.layer3(feature) # [1, 1024, 8, 8]
            feature = self.encoder.model.layer4(feature) # [1, 2048, 4, 4]
            feature = self.avgpool(feature) # [1, 2048, 1, 1]
            feature = torch.flatten(feature, start_dim=1)
        return feature
        
    def forward_shallow(self, x):
        with torch.no_grad():
            feature = self.encoder.model.conv1(x) # [1, 64, 64, 64]
        feature = self.conv(feature) # [B, 64, 64, 64] -> [B, 1, 64, 64]
        feature = torch.flatten(feature, start_dim=1) # [B, 1, 64, 64] -> [B, 1*64*64]
        return feature
    
    def extract_feature_map(self, x):
        with torch.no_grad():
            feature = self.encoder(x) # [B, 3, 128, 128] -> [B, 3904, 64, 64]
            feature = self.conv(feature) # [B, 3904, 64, 64] -> [B, 3, 64, 64]
        return feature


    def forward(self, x):
        if self.pixelnerf_mode == "spatial":
            return self.forward_spatial(x)
        elif self.pixelnerf_mode == "global":
            return self.forward_global(x)
        elif self.pixelnerf_mode == "shallow":
            return self.forward_shallow(x)
        else:
            raise NotImplementedError


class PixelNeRF(nn.Module):
    """
    pixelnerf with/without dino 
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = make_model(cfg.pixelnerf_model).cuda()
        renderer = NeRFEmbedRenderer.from_conf(
            cfg.pixelnerf_renderer, eval_batch_size=cfg.ray_batch_size
        ).cuda()

        # choose model
        if cfg.embedding_name == "pixelnerf+dino":
            ckpt_path = "ckpts/pixelnerf_dino_r50"
        else:
            ckpt_path = "ckpts/pixelnerf_r50"

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            info = self.net.load_state_dict(ckpt, strict=False)
            print(info)
            print(colored(f"Loaded encoder from {ckpt_path}", 'cyan'))
        else:
            print(colored(f"The encoder is not loaded from your ckpt path.", 'red'))

        coarse = True
        if renderer.n_coarse < 64:
                # Ensure decent sampling resolution
            renderer.n_coarse = 64
        if coarse:
            renderer.n_coarse = 64
            renderer.n_fine = 128
            renderer.using_fine = True

        self.renderer = renderer.bind_parallel(self.net, '1', simple_output=True).eval()

        # some nerf param
        self.W = self.H = 128
        self.z_near = 0.1
        self.z_far = 5.0

        # hard code the param for nerf
        self.pose_static_camera = torch.from_numpy(np.array([[-0.79793298,  0.21090717, -0.56464247,  0.9559    ],
                                            [-0.60241838, -0.30994417,  0.73554517,  1.        ],
                                            [-0.01987589,  0.92706676,  0.37436903,  1.1       ],
                                            [ 0.        ,  0.        ,  0.        ,  1.        ]])).float()
        self.focal = torch.as_tensor(137.24844291261175).float()

        # downscale the feature map
        self.downscale = nn.Sequential(
            nn.AvgPool2d(5, stride=4))


    def forward(self, x):
        """
        x: torch.tensor, [SB, 3, H, W]

        return: latent  
        """
        SB = x.shape[0]
        pri_images = x.unsqueeze(1) # [SB, 1, 3, H, W]

        pri_poses = self.pose_static_camera.unsqueeze(0).unsqueeze(0).repeat(SB, 1, 1, 1) # [4,4] -> [SB, 1, 4, 4]

        focal = self.focal.repeat(SB) # [1] -> [SB]

        feature_map = self.encode(pri_images, pri_poses, focal)

        embed = self.downscale(feature_map).reshape(SB, -1)
        return embed

    @torch.no_grad()
    def encode(self, pri_images, pri_poses, focal):
        """
        pri_images: torch.tensor, [SB, NV, 3, H, W]
        pri_poses: torch.tensor, [SB, NV, 4, 4]
        focal: torch.tensor, [SB]
        """
        self.net.encode(
                pri_images.cuda(),
                pri_poses.cuda(),
                focal.cuda(),
            )
        
        return self.net.encoder.latent
    

    @torch.no_grad()
    def decode(self, tgt_poses):
        """
        tgt_poses: torch.tensor, [SB, 4, 4]
        """
        SB = tgt_poses.shape[0]
        # gen rays from tgt poses
        all_rays = gen_rays(
                tgt_poses.reshape(-1, 4, 4), self.W, self.H, self.focal, self.z_near, self.z_far
            ).reshape(SB, -1, 8)
        
        rgb_fine, _depth, _embed = self.renderer(all_rays.cuda())

        # reshape into img
        rgb_fine = rgb_fine.reshape(SB, self.H, self.W, 3).permute(0, 3, 1, 2)
        return rgb_fine


    

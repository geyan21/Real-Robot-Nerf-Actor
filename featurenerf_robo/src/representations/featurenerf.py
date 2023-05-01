import torch.nn as nn
import os
import torch
from termcolor import colored
import numpy as np

from .utils_pixelnerf.render import NeRFEmbedRenderer
from .utils_pixelnerf.model import make_model
from .utils_pixelnerf.util import gen_rays

from analysis.image import save_feature_map, save_rgb_image


    

class FeatureNeRF(nn.Module):
    """
    pixelnerf with dino
    """
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()
        net = make_model(cfg.pixelnerf_model)
        self.net = net
        renderer = NeRFEmbedRenderer.from_conf(
            cfg.pixelnerf_renderer, eval_batch_size=cfg.ray_batch_size
        ).cuda()

        # choose model
        self.pixelnerf_mode = cfg.pixelnerf_mode
        print(colored(f"[FeatureNeRF] pixelnerf mode: {self.pixelnerf_mode}", "red"))
        ckpt_path = cfg.featurenerf_ckpt

        self.encoder = net.encoder.cuda() 

        self.freeze_encoder = cfg.freeze_encoder

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            info = net.load_state_dict(ckpt, strict=False)
            print(info)
            print(colored(f"[FeatureNeRF] Loaded encoder from {ckpt_path}", 'cyan'))
        else:
            print(colored(f"[FeatureNeRF] The encoder is not loaded from your ckpt path.", 'red'))

        if self.pixelnerf_mode == "global":
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).cuda()
        elif self.pixelnerf_mode == "spatial":
            if cfg.pixelnerf_model.encoder.backbone == "resnet50":
                self.conv = nn.Conv2d(3904, 1, 1, 1, 0).cuda()
            elif cfg.pixelnerf_model.encoder.backbone == "resnet18":
                self.conv = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False).cuda()
        else:
            raise NotImplementedError
        
        self.renderer = renderer.bind_parallel(self.net, '1', simple_output=True).eval()

        # some nerf param
        self.W = self.H = self.cfg.image_size
        self.z_near = 0.1
        self.z_far = 5.0

        # # hard code the param for nerf
        # self.pose_static_camera = torch.from_numpy(np.array([[-0.79793298,  0.21090717, -0.56464247,  0.9559    ],
        #                                     [-0.60241838, -0.30994417,  0.73554517,  1.        ],
        #                                     [-0.01987589,  0.92706676,  0.37436903,  1.1       ],
        #                                     [ 0.        ,  0.        ,  0.        ,  1.        ]])).float()
        # self.focal = torch.as_tensor(137.24844291261175).float()


    # def forward(self, x):
    #     """
    #     x: torch.tensor, [SB, 3, H, W]

    #     return: latent  
    #     """
    #     SB, _, H, W = x.shape
    #     pri_images = x.unsqueeze(1) # [SB, 1, 3, H, W]


    def forward(self, x):
        if self.pixelnerf_mode == "spatial":
            return self.forward_spatial(x)
        elif self.pixelnerf_mode == "global":
            return self.forward_global(x)
        else:
            raise NotImplementedError

    def forward_spatial(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                feature = self.encoder(x) # [B, 3, 128, 128] -> [B, 3904, 64, 64]
        else:
            feature = self.encoder(x)
        feature = self.conv(feature) # [B, 3904, 64, 64] -> [B, 3, 64, 64]
        feature = torch.flatten(feature, start_dim=1) # [B, 3, 64, 64] -> [B, 3*64*64]
        return feature

    def forward_global(self, x):
        """
        x: [1,3,128,128]
        """
        if self.freeze_encoder:
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
        else:
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
    def decode(self, tgt_poses, focal):
        """
        tgt_poses: torch.tensor, [SB, 4, 4]
        focal: [SB]
        """
        SB = tgt_poses.shape[0]

        # focal from [SB] to [2]
        focal = torch.stack([focal[0], focal[0]], dim=0)
        # gen rays from tgt poses
        all_rays = gen_rays(
                tgt_poses.reshape(-1, 4, 4), self.W, self.H, focal, self.z_near, self.z_far
            ).reshape(SB, -1, 8)
        
        rgb_fine, _depth, _embed = self.renderer(all_rays.cuda())

        # reshape into img
        rgb_fine = rgb_fine.reshape(SB, self.H, self.W, 3).permute(0, 3, 1, 2)
        return rgb_fine, _depth, _embed

    def extract_feature_map(self, x):
        with torch.no_grad():
            feature = self.encoder(x) # [B, 3, 128, 128] -> [B, 3904, 64, 64]
            feature = self.conv(feature) # [B, 3904, 64, 64] -> [B, 3, 64, 64]
        return feature
    
    

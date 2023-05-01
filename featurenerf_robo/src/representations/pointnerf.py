import torch
import torch.nn as nn
import torch.nn.functional as F
from .featurenerf import FeatureNeRF
from .pointnet import PointNetEncoder


class PointNeRF(nn.Module):
    """
    pointnet with nerf feature
    """
    def __init__(self, cfg):
        super().__init__()
        self.pointnet = PointNetEncoder(cfg, channel=3).cuda()
        self.featurenerf = FeatureNeRF(cfg).cuda()
        self.nerf_mlp = self.featurenerf.net

        self.pointnet.cuda()
        self.nerf_mlp.cuda()
        # freeze nerf
        for param in self.nerf_mlp.parameters():
            param.requires_grad = False
        print("[PointNeRF] freeze nerf.")
        # unfreeze pointnet
        for param in self.pointnet.parameters():
            param.requires_grad = True
        print("[PointNeRF] unfreeze pointnet.")

        # aggregate
        self.conv1 = torch.nn.Conv1d(384+64, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)


    def encode_prior(self, pri_images, pri_poses, focal):
        """
        pri_images: torch.tensor, [SB, NV, 3, H, W]
        pri_poses: torch.tensor, [SB, NV, 4, 4]
        focal: torch.tensor, [SB]
        """
        self.nerf_mlp.encode(
                pri_images.cuda(),
                pri_poses.cuda(),
                focal.cuda(),
            )
        
        return self.nerf_mlp.encoder.latent


    def forward(self, obs2d, obs3d, pose, focal):
        """
        obs2d: [B, 3, H, W]
        obs3d: [B, 3, N]
        pose: [B, 4, 4]
        focal: [B]
        """
        self.encode_prior(pri_images=obs2d, pri_poses=pose, focal=focal)

        B, C_3d, N = obs3d.shape
        B, C_2d, H, W = obs2d.shape

        input_nerf = obs3d.transpose(1, 2)
        view_dir = torch.zeros_like(input_nerf).to(input_nerf.device)
        point_feature = self.nerf_mlp(input_nerf, viewdirs=view_dir) # [B, N, D], D=4+384+3=391

        dino_feature = point_feature[..., 4:4+384] # [B, N, 384]
        pointnet_feature = self.pointnet(obs3d, global_feat=False) # [B, 64, N]
        
        # concat
        dino_feature = dino_feature.transpose(1, 2) # [B, 384, N]
        feature_concat = torch.cat((dino_feature, pointnet_feature), dim=1) # [B, 384+64, N]


        # aggregate
        x = F.relu(self.bn1(self.conv1(feature_concat))) # [B, 512, N]
        x = self.bn2(self.conv2(x))  # [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        x = x.view(-1, 1024)

        return x
    

    def reconstruct(self, pri_images, pri_poses, focal):
        """
        recontruct the prior images with prior poses
        """
        self.encode_prior(pri_images, pri_poses, focal)

        rgb_fine, _depth, _embed = self.featurenerf.decode(pri_poses, focal)


        return rgb_fine
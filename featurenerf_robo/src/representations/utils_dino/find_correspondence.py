import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vlab

from featurenerf.correspondence.dino import DINO


class PretrainedCorrespondence(nn.Module):
    def __init__(self, img_size, pretrain_k=1):
        super().__init__()
        self.net = DINO().eval()
        self.img_size = img_size
        self.feat_size = img_size // 8
        self.k = pretrain_k

        grid = (
            torch.Tensor(np.array(np.meshgrid(range(self.feat_size), range(self.feat_size)))).cuda().reshape(2, -1)
        )  # 2,h*w
        grid = grid / (self.feat_size / 2) - 1
        self.grid = grid  # 2,h*w  # range in [-1,1]*[-1,1], 2d location of every pixel

        for param in self.net.parameters():
            param.requires_grad = False

    def match(self, src_img, tgt_img, src_mask, tgt_mask):
        # return the match
        bsz = src_img.shape[0]
        src_mask = src_mask[:, None]
        tgt_mask = tgt_mask[:, None]

        src_feat = self.net(src_img)
        tgt_feat = self.net(tgt_img)

        # all_img = torch.cat([src_img, tgt_img], dim=0)
        # MAX_BATCH_SIZE = 64
        # with torch.no_grad():
        #     torch.cuda.empty_cache()
        #     if all_img.shape[0] <= MAX_BATCH_SIZE:
        #         all_feat = self.net(all_img)
        #
        #     # Process in chunks to avoid CUDA out-of-memory
        #     else:
        #         num_chunks = np.ceil(all_img.shape[0] / MAX_BATCH_SIZE).astype('int')
        #         data_chunks = []
        #         for i, ims_ in enumerate(all_img.chunk(num_chunks)):
        #             data_chunks.append(self.net(ims_))
        #         all_feat = torch.cat(data_chunks, dim=0)
        # src_feat = all_feat[:bsz]
        # tgt_feat = all_feat[bsz:]

        src_feat = src_feat.reshape(*src_feat.shape[:2], -1)
        tgt_feat = tgt_feat.reshape(*tgt_feat.shape[:2], -1)
        hw = src_feat.shape[-1]

        src_mask_down = (
            F.interpolate(src_mask, (self.feat_size, self.feat_size), mode="nearest").reshape(bsz, -1) * 1.0
        )  # b,h*w
        tgt_mask_down = (
            F.interpolate(tgt_mask, (self.feat_size, self.feat_size), mode="nearest").reshape(bsz, -1) * 1.0
        )  # b,h*w

        mask_down = src_mask_down[:, :, None] * tgt_mask_down[:, None, :]

        pointcorr = src_feat.permute(0, 2, 1).bmm(tgt_feat)  # b,h*w,h*w(tgt)  # TODO chunked cosine similarity
        pointcorr = pointcorr * (mask_down > 0) - 1e5 * (mask_down == 0)  # b,h*w,h*w

        pointcorr_max_bw = pointcorr.max(1).indices  # b,h*w(tgt)
        pointcorr_max_fw = pointcorr.max(2).indices  # b,h*w(src)
        pointcorr_max_cy = torch.gather(pointcorr_max_fw, -1, pointcorr_max_bw)

        grid = self.grid.reshape(bsz, 2, -1)
        match = torch.gather(grid, -1, pointcorr_max_bw[:, None].repeat(1, 2, 1))  # bsz,2,h*w
        cycle = torch.gather(grid, -1, pointcorr_max_cy[:, None].repeat(1, 2, 1))  # bsz,2,h*w

        distance = (cycle - grid).norm(2, 1)  # bsz,h*w
        distance = distance * (tgt_mask_down > 0) + 1e5 * (tgt_mask_down == 0)  # b,h*w
        _, indices = torch.topk(-distance, k=self.k, dim=1)  # b,k

        match = torch.gather(match, -1, indices[:, None].repeat(1, 2, 1))  # b,2,k
        grid = torch.gather(grid, -1, indices[:, None].repeat(1, 2, 1))  # b,2,k
        match_mask = torch.gather(tgt_mask_down, -1, indices)  # b,k
        indices_match = torch.gather(pointcorr_max_bw, -1, indices)  # b,k

        # match: 2d point locations on src image;
        # grid: 2d point locations on tgt image;
        # indices_match: grid indices on src image;
        # indices: grid indices on tgt image;
        # match_mask: whether selected points on tgt image is within tgt mask;
        return match, grid, indices_match, indices, match_mask  # b,2,k; b,2,k; b,k; b,k


def draw_correspondence(src_img_vis, tgt_img_vis, match, grid):
    src_img_vis = src_img_vis.copy().astype(np.uint8)
    tgt_img_vis = tgt_img_vis.copy().astype(np.uint8)

    colors = vlab.flow_to_image(match.cpu().numpy().transpose(0, 2, 1))

    for pi, (point_src, point_tgt) in enumerate(zip(match[0].permute(1, 0), grid[0].permute(1, 0))):
        # color = (int((point_src[0] + 1) / 2 * 255), int((point_src[1] + 1) / 2 * 255), int(255))
        color = colors[0, pi].tolist()
        cv2.circle(
            tgt_img_vis, (int((point_tgt[0] + 1) / 2 * img_size), int((point_tgt[1] + 1) / 2 * img_size)), 4, color, -1
        )
        cv2.circle(
            src_img_vis, (int((point_src[0] + 1) / 2 * img_size), int((point_src[1] + 1) / 2 * img_size)), 4, color, -1
        )
    return np.concatenate([src_img_vis, tgt_img_vis], axis=1)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # data_dir = "/data/jianglong/data/raw/sapien/articulated_nerf/dnerf/9968"
    # img1_path = os.path.join(data_dir, "train/20_0.png")
    # img2_path = os.path.join(data_dir, "train/20_1.png")
    data_dir = "/data/jianglong/data/raw/nerf/pixel_nerf_data/srn_cars/cars_val/"
    img1_path = os.path.join(data_dir, "98b30f0a29fe2a1ba7fd25564c2e888e/rgb/000000.png")
    # img2_path = os.path.join(data_dir, "98b30f0a29fe2a1ba7fd25564c2e888e/rgb/000001.png")
    img2_path = os.path.join(data_dir, "ffbf897d9867fadff9a62a8acc9e8cfe/rgb/000000.png")

    img1 = vlab.imread(img1_path)
    img2 = vlab.imread(img2_path)

    # img_size = 128
    # pretrain_k = 16
    img_size = 256
    pretrain_k = 64
    if img1.shape[0] != img_size or img1.shape[1] != img_size:
        img1 = vlab.imresize(img1, (img_size, img_size))
        img2 = vlab.imresize(img2, (img_size, img_size))

    img1, mask1 = img1[..., :3], img1[..., 3]
    img2, mask2 = img2[..., :3], img2[..., 3]

    if "srn" in data_dir:
        mask1 = np.all(img1 == 255, -1).astype(np.float32) * 255
        mask2 = np.all(img2 == 255, -1).astype(np.float32) * 255

    # print(mask1[mask1 != 0])
    # vlab.imshow(mask1)
    # vlab.imshow(img1)

    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mask1 = 1 - torch.from_numpy(mask1).unsqueeze(0).float() / 255.0
    mask2 = 1 - torch.from_numpy(mask2).unsqueeze(0).float() / 255.0

    img1, mask1 = img1.to(device), mask1.to(device)
    img2, mask2 = img2.to(device), mask2.to(device)

    corr = PretrainedCorrespondence(img_size=img_size, pretrain_k=pretrain_k)
    corr = corr.to(device)
    match, grid, indices_match, indices, match_mask = corr.match(img1, img2, mask1, mask2)

    src_img_vis = img1[0].clone().detach().permute(1, 2, 0).cpu().numpy() * 255.0
    tgt_img_vis = img2[0].clone().detach().permute(1, 2, 0).cpu().numpy() * 255.0

    img_vis = draw_correspondence(src_img_vis, tgt_img_vis, match, grid)
    vlab.imshow(img_vis)

import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor
from .color_jitter import random_color_jitter

import matplotlib.pyplot as plt
import torchvision
from natsort import natsorted

from termcolor import colored

import clip

def parse_camera_file(file_path):
    """
    Parse our camera format.

    The format is (*.txt):
    
    4x4 matrix (camera extrinsic)
    space
    3x3 matrix (camera intrinsic)

    focal is extracted from the intrinsc matrix
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    camera_extrinsic = []
    for x in lines[0:4]:
        camera_extrinsic += [float(y) for y in x.split()]
    camera_extrinsic = np.array(camera_extrinsic).reshape(4, 4)

    camera_intrinsic = []
    for x in lines[5:8]:
        camera_intrinsic += [float(y) for y in x.split()]
    camera_intrinsic = np.array(camera_intrinsic).reshape(3, 3)

    focal = camera_intrinsic[0, 0]

    return camera_extrinsic, camera_intrinsic, focal


def parse_img_file(file_path):
    """
    Read as tensor with range [0,1]
    """
    # img = torchvision.io.read_image(file_path).float().div(255.)
    img = imageio.imread(file_path).astype(np.float32) / 255.
    img = torch.from_numpy(img).float()
    # [128,128,3] -> [3,128,128]
    img = img.permute(2, 0, 1)
    return img


def save_single_channel_img(img, fname):
    """
    visualize a single channel image and save
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # threshold
    plt.figure()
    plt.imshow(img, cmap="rainbow")
    plt.tight_layout()
    plt.savefig(fname)


class PerActDataset(torch.utils.data.Dataset):
    """
    PerAct dataset
    """

    def __init__(
        self,
        path,
        stage="train",
        image_size=128,
        scale_focal=True,
        max_imgs=100000,
        z_near=1.2,
        z_far=4.0,
        skip_step=None,
        task_list=None,
        teacher_model="none",
        use_color_jitter=False,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | test
        :param image_size result image size (resizes if different); None to keep original size
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)
        if task_list is None:
            task_list = ['close_jar', 'open_drawer']
        else:
            task_list = task_list
        
        print("task_list: ", task_list)

        self.all_objs = []
        for task in task_list:
            data_path = os.path.join(self.base_path, task, 'all_variations', 'episodes')
            if os.path.exists(data_path):
                episode_dirs = [os.path.join(data_path, x) for x in os.listdir(data_path)]
                for episode_dir in episode_dirs:
                    self.all_objs += [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]


        # dino_model_path= "../ckpt/dino_deitsmall8_pretrain_full_checkpoint.pth"
        # model_name = 'dino'
        # self.dino = get_model(model_name, dino_model_path, 'cuda:0')
        self.stage = stage

        print(colored("Loading Robo dataset: %s, stage: %s, num scenes: %d" % (self.base_path, stage, len(self.all_objs)), 'red'))

        self.use_color_jitter = use_color_jitter
        print(colored("peract dataset use_color_jitter: %s" % (self.use_color_jitter), 'cyan'))



        self.teacher_model = teacher_model
        print(colored("teacher_model: %s" % (self.teacher_model), 'red'))
        if self.teacher_model == 'clipdino':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("RN50", device=self.device)
            self.language_model = model

        self.image_size = image_size
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs
        self._coord_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32)) # for aligning coordinate system

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False


    def __len__(self):
        return len(self.all_objs)


    def __getitem__(self, index):

        root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "images", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]

        pose_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "poses", "*"))
            if (x.endswith(".txt"))
        ]

        rgb_paths = natsorted(rgb_paths)
        pose_paths = natsorted(pose_paths)

        all_imgs = []
        all_poses = []

        focal = None

        for idx, (rgb_path, pose_path) in enumerate(zip(rgb_paths, pose_paths)):
            
            camera_extrinsic, camera_intrinsic, focal = parse_camera_file(pose_path)
            img_tensor = parse_img_file(rgb_path) # [0,1]

            all_imgs.append(img_tensor)
            camera_extrinsic = torch.from_numpy(camera_extrinsic).float()
            camera_extrinsic = camera_extrinsic @ self._coord_trans
            all_poses.append(camera_extrinsic)

        focal =  torch.tensor((focal, focal), dtype=torch.float32)
        
        all_imgs = torch.stack(all_imgs)

        if self.use_color_jitter:
            # we use same color jitter for all images in a scene
            B, C, H, W = all_imgs.shape
            all_imgs = all_imgs.view(1, B*C, H, W)
            all_imgs = random_color_jitter(all_imgs)
            all_imgs = all_imgs.view(B, C, H, W)

        
        # convert [1,1] to [-1,1]
        all_imgs = all_imgs * 2 - 1  # [0,1] -> [-1,1]
        all_poses = torch.stack(all_poses)

        if self.teacher_model == 'dino':
            feature_path = os.path.join(root_dir, "features.npz")
            feat_dim = 384
            sentence_emb = torch.zeros(1)
            token_emb = torch.zeros(1)

        elif self.teacher_model == 'clip':
            feature_path = os.path.join(root_dir, "features_clip_2048.npz")
            feat_dim = 2048
            sentence_emb = torch.zeros(1)
            token_emb = torch.zeros(1)

        elif self.teacher_model == 'clipdino':
            # get visual dino feature
            feature_path = os.path.join(root_dir, "features.npz")
            feat_dim = 384

            # get language description
            description_feature_path = os.path.join(root_dir, "description_feature.npz")
            text_features = np.load(description_feature_path, allow_pickle=True)['arr_0']
            text_features = text_features.item()
            sentence_emb = torch.from_numpy(text_features['sentence_emb']).float().squeeze(0)
            token_emb = torch.from_numpy(text_features['token_emb']).float().squeeze(0)


        if os.path.exists(feature_path):
            all_feats = np.load(feature_path)['arr_0'] # [N, embed_dim, H, W] np.ndarray

            all_feats = torch.from_numpy(all_feats).float()
            N, C, H, W = all_imgs.shape
            all_feats = torch.nn.functional.interpolate(all_feats, size=(H, W), mode='bilinear', align_corners=False) # (N, C, H, W)
        else:
            all_feats = torch.zeros([all_imgs.shape[0], feat_dim, all_imgs.shape[2], all_imgs.shape[3]]) # [N, 64, H, W] torch.Tensor
            print(colored("No feature file found for %s, using zero." % root_dir, 'red'))

        # torchvision.utils.save_image(all_imgs[0], "test.png")
        # save_single_channel_img(all_feats[1].mean(0), "test_feat.png")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "feats": all_feats,
            
            # language features
            "sentence_emb": sentence_emb,
            "token_emb": token_emb,
        }
        
        return result

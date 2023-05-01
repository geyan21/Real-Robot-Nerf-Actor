import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor

import matplotlib.pyplot as plt
import torchvision
import pdb

def parse_camera_file(file_path):
    """
    Parse our camera format.

    The format is (*.txt):
    
    extrinsc
    4x4 matrix
    intrinsc
    3x3 matrix
    focal
    float
    image_size
    float, float
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    camera_extrinsic = []
    for x in lines[0:4]:
        camera_extrinsic += [float(y) for y in x.split()]
    camera_extrinsic = np.array(camera_extrinsic).reshape(4, 4)

    # camera_intrinsic = []
    # for x in lines[6:9]:
    #     camera_intrinsic += [float(y) for y in x.split()]
    # camera_intrinsic = np.array(camera_intrinsic).reshape(3, 3)

    # focal = float(lines[10])
    # return camera_extrinsic, camera_intrinsic, focal
    return camera_extrinsic


def parse_img_file(file_path):
    """
    Read as tensor with range [0,1]
    """
    img = torchvision.io.read_image(file_path).float().div(255.)
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


class realRoboDataset(torch.utils.data.Dataset):
    """
    xArm dataset
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
        self.image_to_tensor = get_image_to_tensor_balanced()
        assert os.path.exists(self.base_path)
        # if task_list is None:
        #     task_list = ['lift', 'peginsert', 'reachwall', 'shelfplacing', 'stacking']
        # else:
        #     task_list = task_list
        
        # print("task_list: ", task_list)

        # self.all_objs = []
        # for task in task_list:
        #     data_path = os.path.join(self.base_path, 'robot_'+task+'_128')
        #     if os.path.exists(data_path):
        #         self.all_objs += [os.path.join(data_path, x) for x in os.listdir(data_path)]




        # dino_model_path= "../ckpt/dino_deitsmall8_pretrain_full_checkpoint.pth"
        # model_name = 'dino'
        # self.dino = get_model(model_name, dino_model_path, 'cuda:0')
        self.stage = stage

        print(
            "Loading Real-Robo dataset",
            self.base_path,
            "stage:",
            stage,
            # "num scenes:",
            # len(self.all_objs),
        )

        self.image_size = image_size
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False
        
        self.realdata_names = []
        self.data_path = os.path.join(self.base_path, 'img')
        self.feature_path = os.path.join(self.base_path, "features_real.npz")
        filenames = os.listdir(self.data_path)
        for i, name in enumerate(filenames):
            self.realdata_names.append(
                os.path.join(self.data_path,name)
            )
        self.dataset_name = self.base_path.split('/')[-1]
        


        


    def __len__(self):
        #return len(self.all_objs)
        return len(self.realdata_names)
        #return self.images.shape[0]

    def __getitem__(self, index):

        npz = True

        if npz:
            root_dir = self.realdata_names[index]
            realdata = np.load(self.realdata_names[index])
            imgs = torch.from_numpy(np.transpose(realdata['images'],(0, 3, 1, 2)))#float32
            all_poses = torch.from_numpy(realdata['poses']) #float32
            focal = torch.from_numpy(realdata['focal']) #float64
            all_imgs = []
            for im in imgs:
                im = im * 2 - 1
                all_imgs.append(im)

            all_imgs = torch.stack(all_imgs)
            #torchvision.utils.save_image(all_imgs[0], "test_Clean_1024_npz.png")
        else:
            root_dir = self.realdata_names[index]
            rgb_paths = [
                x
                for x in glob.glob(os.path.join(root_dir, "images", "*"))
                if (x.endswith(".jpg") or x.endswith(".png"))
            ]
            rgb_paths = sorted(rgb_paths)

            pose_paths = [
                x
                for x in glob.glob(os.path.join(root_dir, "poses", "*"))
                if (x.endswith(".txt"))
            ]
            pose_paths = sorted(pose_paths)

            all_imgs = []
            all_poses = []
            
            # focal = 74.61835
            focal = torch.tensor((focal, focal), dtype=torch.float32)

            for idx, (rgb_path, pose_path) in enumerate(zip(rgb_paths, pose_paths)):
                
                camera_extrinsic  = parse_camera_file(pose_path)
                
                img_tensor = parse_img_file(rgb_path)
                # conver [1,1] to [-1,1]
                img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]

                # img = imageio.imread(rgb_path)[..., :3]
                # img_tensor = self.image_to_tensor(img)

                all_imgs.append(img_tensor)
                camera_extrinsic = torch.from_numpy(camera_extrinsic).float()
                all_poses.append(camera_extrinsic)

            all_imgs = torch.stack(all_imgs)
            all_poses = torch.stack(all_poses)
            #torchvision.utils.save_image(all_imgs[0], "test.png")

        
        if os.path.exists(self.feature_path):
            all_feats = np.load(self.feature_path)['features'] # [N, embed_dim, H, W] np.ndarray
            cls_attn = np.load(self.feature_path)['cls_attn'] # [N, nh=6, H, W] np.ndarray
            all_feats = torch.from_numpy(all_feats).float()
            cls_attn = torch.from_numpy(cls_attn).float()
            N, C, H, W = all_imgs.shape
            all_feats = torch.nn.functional.interpolate(all_feats, size=(H, W), mode='bilinear', align_corners=False) # (N, C, H, W)
            cls_attn = torch.nn.functional.interpolate(cls_attn, size=(H, W), mode='bilinear', align_corners=False) # (N, 6, H, W)
        else:
            all_feats = torch.zeros([all_imgs.shape[0], 64, all_imgs.shape[2], all_imgs.shape[3]]) # [N, 64, H, W] torch.Tensor
        
        id_ = 0
        torchvision.utils.save_image(all_imgs[id_], f"test_{self.dataset_name}_{id_}.png")
        save_single_channel_img(all_feats[id_].mean(0), f"test_feat_{self.dataset_name}_{id_}.png")
        save_single_channel_img(cls_attn[id_].mean(0), f"test_cls_attn_{self.dataset_name}_{id_}.png")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "feats": all_feats,
            "cls_attn": cls_attn
        }
        
        return result
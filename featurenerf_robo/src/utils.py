import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import os
import json
import random
import subprocess
import platform
from datetime import datetime
from copy import deepcopy
from multiprocessing import Process
from termcolor import colored
import time
import cv2
from PIL import Image
from natsort import natsorted
from random import shuffle
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
import math


EMBED_DIM = {"resnet18":512, "resnet50":2048, "resnet101":2048, "resnet152":2048, \
			"dino":2048, 'mocov2': 2048, 'pri3d': 64*64, \
			'pixelnerf+dino': 12288, 'pixelnerf':12288}
            
def print_red(content):
    print(colored(content, color="red"))
    
def print_cyan(content):
    print(colored(content, color="cyan"))

def color_print(content, color="red"):
    print(colored(content, color=color))


def make_video(images, fname, fps=30):
    """
    Transform a list of images into a video and save.
    """
    for i in range(len(images)):
        if isinstance(images[i], torch.Tensor):
            images[i] = images[i].detach().cpu()
        if images[i].ndim == 4:
            images[i] = images[i].squeeze(0)
        if images[i].shape[0] == 3:
            images[i] = images[i].permute(1, 2, 0)
        if images[i].max() < 1:
            images[i] = images[i] * 255.
    images = torch.stack(images)
    torchvision.io.write_video(fname, images, fps=fps) # TxHxWxC

def save_single_channel_img(img, fname, threhold=0):
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
    

def save_single_channel_video(imgs:list, fname:str, buffer:str):
    """
    save a list of single channel images as a video
    """
    if not os.path.exists(".buffer/{}/".format(buffer)):
        os.makedirs(".buffer/{}/".format(buffer))

    # make img with matplotlib
    img_list = []
    for i in range(len(imgs)):
        if isinstance(imgs[i], torch.Tensor):
            imgs[i] = imgs[i].detach().cpu().numpy()
        plt.imshow(imgs[i], cmap="rainbow")
        plt.savefig(".buffer/{}/{}.png".format(buffer, i))
        img_list.append(".buffer/{}/{}.png".format(buffer, i))

    # make video with torchvision
    images = []
    for img_path in img_list:
        img = torchvision.io.read_image(img_path)
        # clean the buffer
        os.remove(img_path)
        images.append(img[:-1])
    images = torch.stack(images).permute(0, 2, 3, 1)
    torchvision.io.write_video(fname, images, fps=30) # TxHxWxC

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    




def load_config(key=None):
    path = os.path.join('setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def prefill_memory(capacity, obs_shape):
    obses = []
    if len(obs_shape) > 1:
        c, h, w = obs_shape
        for _ in range(capacity):
            frame = np.ones((c, h, w), dtype=np.uint8)
            obses.append(frame)
    else:
        for _ in range(capacity):
            obses.append(np.ones(obs_shape[0], dtype=np.float32))

    return obses



class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3:(i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns number of params in network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'


def save_obs(obs, fname='obs', resize_factor=None):
    assert obs.ndim == 3, 'expected observation of shape (C, H, W)'
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu()
    else:
        obs = torch.FloatTensor(obs)
    c, h, w = obs.shape
    if resize_factor is not None:
        obs = torchvision.transforms.functional.resize(obs, size=(h * resize_factor, w * resize_factor))
    if c == 3:
        torchvision.utils.save_image(obs / 255., fname + '.png')
    elif c == 9:
        grid = torch.stack([obs[i * 3:(i + 1) * 3] for i in range(3)], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=3)
        torchvision.utils.save_image(grid / 255., fname + '.png')
    else:
        raise NotImplementedError('save_obs does not support other number of channels than 3 or 9')


def parallel(fn, args, wait=10):
	"""Executes function multiple times in parallel, using individual seeds (given in args.seed)"""
	assert not isinstance(fn, (list, tuple)), 'fn must be a function, not a list or tuple'
	assert args.seed is not None, 'No seed(s) given'
	seeds = args.seed
	if not isinstance(seeds, (list, tuple)):
		return fn(args)
	proc = []
	for seed in seeds:
		_args = deepcopy(args)
		_args.seed = seed
		p = Process(target=fn, args=(_args,))
		p.start()
		proc.append(p)
		print(colored(f'Started process {p.pid} with seed {seed}', 'green', attrs=['bold']))
		time.sleep(wait) # ensure that seed has been set in child process
	for p in proc:
		p.join()
	while len(proc) > 0:
		time.sleep(wait)
		for p in proc:
			if not p.is_alive():
				p.terminate()
				proc.remove(p)
				print(f'One of the child processes have terminated.')
	exit(0)

def save_image(obs, fname):
    torchvision.utils.save_image(obs / 255., fname + '.png')




def psnr(pred, target):
    """
    Compute PSNR of two tensors in decibels.
    pred/target should be of same size or broadcastable
    """
    mse = ((pred - target) ** 2).mean()
    psnr = -10 * math.log10(mse)
    return psnr


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]

    link: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py"""

    def __init__(self):
        self.name = "SSIM"

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
            else:
                raise ValueError("Input images must have 1 or 3 channels.")
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # img1 = img1.astype(np.float64)
        # img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class BehaviorCloneDataset(Dataset):
    """BehaviorCloneDataset"""

    def __init__(self, cfg, root_dir, episode_length, image_size, num_trajs, num_cameras, use_3d=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.num_cameras = num_cameras
        self.traj_dirs = os.listdir(self.root_dir)
        self.use_3d = use_3d
        # sample num_trajs trajectories
        if num_trajs >= len(self.traj_dirs):
            self.traj_dirs = self.traj_dirs
        else:
            self.traj_dirs = random.sample(self.traj_dirs, num_trajs)
        
        self.traj_dirs = natsorted(self.traj_dirs)

        print(colored(f'[BehaviorCloneDataset] training trajs in total: {len(self.traj_dirs)}', "cyan"))
        if self.use_3d:
            print(colored(f'[BehaviorCloneDataset] point cloud num points: {cfg.num_points}', "cyan"))

        self.episode_length = episode_length
        self.nominate_function_img = lambda x, idx: "{0:03}".format(x)+"_%u.png"%idx
        self.nominate_function_depth = lambda x, idx: "{0:03}".format(x)+"_%u_depth.png"%idx
        self.nominate_function_pc = lambda x: "{0:03}".format(x)+"_pc.npy"

        from torchvision import transforms

        self.transform = [
                transforms.Resize(image_size),
                transforms.ToTensor()
            ]

        self.transform = transforms.Compose(self.transform)


    def __len__(self):
        return len(self.traj_dirs)


    def load_img_obs(self, traj_idx):
        """
        load all imgs of one trajectory
        """

        img_path_root = os.path.join(self.root_dir, self.traj_dirs[traj_idx], 'rgb')
        img_trajs = []

        for step in range( self.episode_length+1 ): # plus one for the last obs
            one_obs = []
            for cam_idx in range(self.num_cameras):
                img_path = os.path.join(img_path_root, self.nominate_function_img(step, cam_idx))
                img_i = Image.open(img_path).convert('RGB')
                img_i = self.transform(img_i)
                one_obs.append(img_i)
            one_obs = torch.cat(one_obs, dim=0)
            img_trajs.append( one_obs )

        return torch.stack(img_trajs)


    def load_depth_obs(self, traj_idx):
        """
        load all imgs of one trajectory
        """

        depth_path_root = os.path.join(self.root_dir, self.traj_dirs[traj_idx], 'depth')
        depth_trajs = []

        for step in range( self.episode_length+1 ):
            one_obs = []
            for cam_idx in range(self.num_cameras):
                depth_path = os.path.join(depth_path_root, self.nominate_function_depth(step, cam_idx))
                depth_i = Image.open(depth_path).convert('L')
                depth_i = self.transform(depth_i)
                one_obs.append(depth_i)
            one_obs = torch.cat(one_obs, dim=0)
            depth_trajs.append( one_obs )
        
        return torch.stack(depth_trajs)

    
    def load_point_cloud(self, traj_idx):
        """
        load all point clouds of one trajectory
        """
        point_cloud_path_root = os.path.join(self.root_dir, self.traj_dirs[traj_idx], 'point_cloud')
        point_cloud_trajs = []

        for step in range( self.episode_length+1 ):
            point_cloud_path = os.path.join(point_cloud_path_root, self.nominate_function_pc(step))
            point_cloud_i = np.load(point_cloud_path)
            # downsample to num_points with random sampling
            point_cloud_i = point_cloud_i[np.random.choice(point_cloud_i.shape[0], self.cfg.num_points, replace=False), :]
            point_cloud_trajs.append( point_cloud_i )

        return self.preprocess(point_cloud_trajs)


    def load_transitions(self, traj_idx):
        """
        load all transitions of one trajectory
        """
        traj_path = os.path.join(self.root_dir, self.traj_dirs[traj_idx], 'transition', "transitions.npy")
        transitions = np.load(traj_path, allow_pickle=True)
        return transitions

    def load_camera_extrinsics(self, infos):
        camera_extrinsics = []
        for i in range(len(infos)):
            camera_extrinsics.append(infos[i]['camera_extrinsics'])

    def preprocess(self, data):
        return np.stack(data[:])

    def __getitem__(self, idx):
        
        # load batch of trajs
        img_trajs = self.load_img_obs(idx)
        transitions = self.load_transitions(idx)
        if self.use_3d:
            depth_trajs = self.load_depth_obs(idx)
            point_cloud_trajs = self.load_point_cloud(idx)

        # keys: 'full_state', 'state', 'info', 'action', 'reward', 'done', 'next_full_state', 'next_state', 'next_info'
        # split these keys into different lists
        full_state, state, info, action, reward, done, next_full_state, next_state, next_info = [], [], [], [], [], [], [], [], []
        for i in range(len(transitions)):
            transition = transitions[i]
            full_state.append(transition['full_state'])
            state.append(transition['state'])
            info.append(transition['info'])
            action.append(transition['action'])
            reward.append(transition['reward'])
            done.append(transition['done'])
            next_full_state.append(transition['next_full_state'])
            next_state.append(transition['next_state'])
            next_info.append(transition['next_info'])

        # preprocess these lists
        full_state = self.preprocess(full_state)
        state = self.preprocess(state)
        info = self.preprocess(info)
        action = self.preprocess(action)
        reward = self.preprocess(reward)
        done = self.preprocess(done)
        next_full_state = self.preprocess(next_full_state)
        next_state = self.preprocess(next_state)
        next_info = self.preprocess(next_info)


        # get img obs
        obs = img_trajs[:-1]
        next_obs = img_trajs[1:]

        # get depth obs
        if self.use_3d:
            depth_obs = depth_trajs[:-1]
            next_depth_obs = depth_trajs[1:]
            point_cloud = point_cloud_trajs[:-1]
            next_point_cloud = point_cloud_trajs[1:]
        else:
            depth_obs = None
            next_depth_obs = None
            point_cloud = None
            next_point_cloud = None

        
        # return dict
        return_dict = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done,
            'next_obs': next_obs,
            'full_state': full_state,
            'state': state,
            'info': info,
            'next_full_state': next_full_state,
            'next_state': next_state,
            'next_info': next_info,
            # 3D info
            'depth': depth_obs,
            'next_depth': next_depth_obs,
            'point_cloud': point_cloud,
            'next_point_cloud': next_point_cloud,
        }   


        return return_dict
        
    
    def collect_fn(self, batch):
        """
        'obs': obs,
        'depth': depth_obs,
        'action': action,
        'reward': reward,
        'done': done,
        'next_obs': next_obs,
        'next_depth': next_depth_obs,
        'full_state': full_state,
        'state': state,
        'info': info,
        'next_full_state': next_full_state,
        'next_state': next_state,
        'next_info': next_info,
        """
        # collect batch of data
        obs, action, reward, done, next_obs, full_state, state, info, next_full_state, next_state, next_info = \
            [], [], [], [], [], [], [], [], [], [], []

        if self.use_3d:
            depth, next_depth = [], []
            point_cloud, next_point_cloud = [], []
        for i in range(len(batch)):
            obs.append(batch[i]['obs'])
            action.append(batch[i]['action'])
            reward.append(batch[i]['reward'])
            done.append(batch[i]['done'])
            next_obs.append(batch[i]['next_obs'])
            full_state.append(batch[i]['full_state'])
            state.append(batch[i]['state'])
            info.append(batch[i]['info'])
            next_full_state.append(batch[i]['next_full_state'])
            next_state.append(batch[i]['next_state'])
            next_info.append(batch[i]['next_info'])

            if self.use_3d:
                depth.append(batch[i]['depth'])
                next_depth.append(batch[i]['next_depth'])
                point_cloud.append(batch[i]['point_cloud'])
                next_point_cloud.append(batch[i]['next_point_cloud'])


        obs = np.stack(obs, axis=0)
        
        action = np.stack(action, axis=0)
        reward = np.stack(reward, axis=0)
        done = np.stack(done, axis=0)
        next_obs = np.stack(next_obs, axis=0)
        
        full_state = np.stack(full_state, axis=0)
        state = np.stack(state, axis=0)
        info = np.stack(info, axis=0)
        next_full_state = np.stack(next_full_state, axis=0)
        next_state = np.stack(next_state, axis=0)
        next_info = np.stack(next_info, axis=0)

        if self.use_3d:
            depth = np.stack(depth, axis=0)
            next_depth = np.stack(next_depth, axis=0)
            point_cloud = np.stack(point_cloud, axis=0)
            next_point_cloud = np.stack(next_point_cloud, axis=0)
        else:
            depth = None
            next_depth = None
            point_cloud = None
            next_point_cloud = None

        return dict(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            next_obs=next_obs,
            full_state=full_state,
            state=state,
            info=info,
            next_full_state=next_full_state,
            next_state=next_state,
            next_info=next_info,
            
            # 3D info
            depth=depth,
            next_depth=next_depth,
            point_cloud=point_cloud,
            next_point_cloud=next_point_cloud,
        )





def calc_pca(emb:torch.Tensor, target_dim=3):
    """
    emb: [Dim, H, W]
    return: torch.Tensor, [H, W, C]
    """
    X = emb.cpu().reshape(emb.shape[0], -1).T
    np.random.seed(6)
    pca = PCA(n_components=3)
    pca.fit(X)
    X_rgb = pca.transform(X).reshape(emb.shape[1], emb.shape[2], target_dim)
    X_rgb = torch.as_tensor(X_rgb)
    return X_rgb

class RobotMultiViewDataset(Dataset):
    """RobotMultiViewDataset dataset.
    
    This dataset class also provide collect function.
    """

    def __init__(self, root_dir:str, episode_length:int, image_size:int, near_far:list, 
                 mode:str="train", num_scenes:int=10, cfg=None):
        
        self.root_dir = root_dir 
        self.mode = mode
        self.num_scenes = num_scenes
        self.episode_length = episode_length
        self.distill_active = cfg.distill_active
        self.scene_batch_size = cfg.scene_batch_size
        self.ray_batch_size = cfg.ray_batch_size
        self.scene_dirs = []

        # all scene dirs
        root_dir = os.path.join(root_dir, self.mode)
        self.scene_dirs = [ os.path.join(root_dir, d) for d in os.listdir(root_dir) ]
        if len(self.scene_dirs) > self.num_scenes: # sample num_scenes 
            self.scene_dirs = np.random.choice(self.scene_dirs, self.num_scenes, replace=False)

        # all image and camera dirs corresponding to each scene
        self.scene_image_dirs = [ [ os.path.join(scene_dir, "images", d) for d in os.listdir(os.path.join(scene_dir, "images")) ] for scene_dir in self.scene_dirs ]
        self.scene_camera_dirs = [ [ os.path.join(scene_dir, "cameras", d) for d in os.listdir(os.path.join(scene_dir, "cameras")) ] for scene_dir in self.scene_dirs ]

        from torchvision import transforms
        self.transform = [
            transforms.Resize(image_size),
        ]
        self.transform = transforms.Compose(self.transform)

        self.image_size = image_size
        self.near_far = near_far

        self.read_meta(0, scene_batch_size=1)


    def get_one_view_data(self, scene_idx:int, view_idx:int):
        """
        get one view data
        """
        # get file path
        scene_idx = scene_idx % len(self.scene_dirs)
        view_idx = view_idx % len(self.scene_image_dirs[scene_idx])
        image_path =self.scene_image_dirs[scene_idx][view_idx] 
        camera_path = self.scene_camera_dirs[scene_idx][view_idx]
        
        # read file
        img = self.parse_img_file(image_path)
        camera_extrinsic, camera_intrinsic, focal_origin = self.parse_camera_file(camera_path)
        c2w = np.linalg.inv(camera_extrinsic)

        # to tensor
        img = torch.as_tensor(img).float()
        c2w = torch.as_tensor(c2w).float()
        camera_extrinsic = torch.as_tensor(camera_extrinsic).float()
        camera_intrinsic = torch.as_tensor(camera_intrinsic).float()
        focal_origin = torch.as_tensor(focal_origin).float()

        # get ray
        center = [camera_intrinsic[0,2], camera_intrinsic[1,2]]
        focal = [camera_intrinsic[0,0], camera_intrinsic[1,1]]
        C, H, W = img.shape
        directions = nerf_utils.get_ray_directions(H, W, focal, center)  # (h, w, 3)
        rays_o, rays_d = nerf_utils.get_rays(directions, c2w)  # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, 
                        self.near_far[0] * torch.ones_like(rays_o[:, :1]),
                        self.near_far[1] * torch.ones_like(rays_o[:, :1])], 1) 

        # return a dict
        return {"img":img, "ray":rays,  "extrinsic":camera_extrinsic,
                "c2w":c2w, "intrinsic":camera_intrinsic, "focal":focal_origin}


    @property
    def number_of_scenes(self):
        return len(self.scene_dirs)
    

    def parse_camera_file(self, file_path):
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
        for x in lines[1:5]:
            camera_extrinsic += [float(y) for y in x.split()]
        camera_extrinsic = np.array(camera_extrinsic).reshape(4, 4)

        camera_intrinsic = []
        for x in lines[6:9]:
            camera_intrinsic += [float(y) for y in x.split()]
        camera_intrinsic = np.array(camera_intrinsic).reshape(3, 3)

        focal = float(lines[10])
        return camera_extrinsic, camera_intrinsic, focal


    def parse_img_file(self, file_path):
        """
        Read as tensor with range [0,1]
        """
        img = torchvision.io.read_image(file_path).float().div(255.)
        img = self.transform(img)
        return img


    def get_one_scene_data(self, scene_idx:int):
        """
        Get one scene data
        """
        # check scene_idx
        scene_idx = scene_idx % len(self.scene_dirs)
        num_view = len(self.scene_image_dirs[scene_idx])

        # load data using function `get_one_view_data`
        imgs = []
        extrinsics = []
        intrinsics = []
        focals = []
        c2ws = []
        rays = []
        for i in range(num_view):
            data = self.get_one_view_data(scene_idx, i)
            imgs.append(data["img"])
            rays.append(data["ray"])
            extrinsics.append(data["extrinsic"])
            intrinsics.append(data["intrinsic"])
            focals.append(data["focal"])
            c2ws.append(data["c2w"])

        rays = torch.stack(rays).float()
        imgs = torch.stack(imgs).float()
        extrinsics = torch.stack(extrinsics).float()
        intrinsics = torch.stack(intrinsics).float()
        c2ws = torch.stack(c2ws).float()
        focals = torch.stack(focals).float()

        # return a dict
        return {"img":imgs, "ray":rays, "extrinsic":extrinsics,
                "c2w":c2ws, "intrinsic":intrinsics, "focal":focals}


    def read_source_views(self, scene_idx, scene_batch_size=1, view_idx:list=None, random_sampling=False, num_views=3):
        """
        read source views for constructing the volume
        """
        if view_idx is None:
            if random_sampling:
                # separate length into num_views parts and then uniform sampling from each part
                # parts = np.linspace(0, self.episode_length-1, num_views+1).astype(int)
                # view_idx = [ np.random.randint(parts[i], parts[i+1]) for i in range(num_views) ]

                # uniform sampling from all views
                view_idx = np.random.randint(0, self.episode_length, num_views)
            else:
                view_idx = [0, 25, 45] # 3 views, from different views of the same scene
        else:
            view_idx = view_idx
        
        imgs_scene = []
        w2cs_scene = []
        c2ws_scene = []
        intrinsics_scene = []
        focals_scene = []
        proj_mats_scene = []

        for batch_idx in range(scene_batch_size):
            imgs = []
            w2cs = []
            c2ws = []
            intrinsics = []
            focals = []
            proj_mats = []

            for i in range(len(view_idx)): # collect data provided by the dataset
                view_id = view_idx[i]
                data = self.get_one_view_data(scene_idx+batch_idx, view_id)
                imgs.append(data["img"])
                w2cs.append(data["extrinsic"])
                c2ws.append(data["c2w"])
                intrinsics.append(data["intrinsic"])
                focals.append(data["focal"])

                intrinsic = data["intrinsic"]
                w2c = data["extrinsic"]

                # compute projection matrix
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
                if i == 0:  # reference view
                    ref_proj_inv = np.linalg.inv(proj_mat_l)
                    proj_mats += [np.eye(4)]
                else:
                    proj_mats += [proj_mat_l @ ref_proj_inv]

            imgs = torch.stack(imgs).float().cuda()
            w2cs = torch.stack(w2cs).float().cuda()
            c2ws = torch.stack(c2ws).float().cuda()
            intrinsics = torch.stack(intrinsics).float().cuda()
            focals = torch.stack(focals).float().cuda()
            proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().cuda()

            imgs_scene.append(imgs)
            w2cs_scene.append(w2cs)
            c2ws_scene.append(c2ws)
            intrinsics_scene.append(intrinsics)
            focals_scene.append(focals)
            proj_mats_scene.append(proj_mats)

        imgs_scene = torch.stack(imgs_scene)
        w2cs_scene = torch.stack(w2cs_scene)
        c2ws_scene = torch.stack(c2ws_scene)
        intrinsics_scene = torch.stack(intrinsics_scene)
        focals_scene = torch.stack(focals_scene)
        proj_mats_scene = torch.stack(proj_mats_scene)
        pose_source = {"w2cs": w2cs_scene, "c2ws": c2ws_scene, "intrinsics": intrinsics_scene, "focals": focals_scene}

        return imgs_scene, proj_mats_scene, pose_source


    def read_meta(self, scene_id, 
                scene_batch_size=1,
                img_batch_size=1,
                display_num_views=False,
                distill_active=False, 
                distill_model=None):
        """
        generate rays and ground truth rgb for nerf, i.e., training data
        """


        interval = 1
        self.img_idx = [i for i in range(0, 50, interval)]
        img_batch_size = min(img_batch_size, len(self.img_idx)) # limit the batch size to the number of images in the scene
        # random sample img_idx
        self.img_idx = np.random.choice(self.img_idx, img_batch_size, replace=False) # sample without replacement

        if display_num_views:
            print(colored("Use {} views for current scene.".format(len(self.img_idx)), "red"))

        w, h = self.image_size, self.image_size

        poses_scene = []
        all_rays_scene = []
        all_rgbs_scene = []
        all_imgs_scene = []
        all_distill_features_scene = []

        if scene_batch_size > 1:
            scene_idx_list = np.random.choice(len(self), scene_batch_size, replace=False) # sample without replacement
        else:
            scene_idx_list = [scene_id]

        for sceneid in scene_idx_list:
            poses = []
            all_rays = []
            all_rgbs = []
            all_images = []
            

            for idx in self.img_idx:
                data = self.get_one_view_data(sceneid, idx)
                img, w2c, c2w, intrinsic, focal = data["img"], data["extrinsic"], data["c2w"], data["intrinsic"], data["focal"]
                all_images.append(img)

                # flatten it
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
                near_far = self.near_far
                c2w = np.linalg.inv(w2c)
                poses += [c2w]
                c2w = torch.FloatTensor(c2w)            
                
                all_rgbs += [img]

                # ray directions for all pixels, same for all images (same H, W, focal)
                # intrinsic[:2] *= 4
                center = [intrinsic[0,2], intrinsic[1,2]]
                self.focal = [intrinsic[0,0], intrinsic[1,1]]

                directions = nerf_utils.get_ray_directions(h, w, self.focal, center)  # (h, w, 3)
                rays_o, rays_d = nerf_utils.get_rays(directions, c2w)  # both (h*w, 3)

                all_rays += [torch.cat([rays_o, rays_d,
                                            near_far[0] * torch.ones_like(rays_o[:, :1]),
                                            near_far[1] * torch.ones_like(rays_o[:, :1])],
                                            1)]  # (h*w, 8)

            if distill_active: # diffusion distillation
                # concat images
                all_images = torch.stack(all_images, dim=0).cuda().float()  # (N, 3, H, W)
                all_distill_feature = distill_model(all_images)
                all_images = all_images.cpu()
                # reshape feature shape as rays
                all_distill_feature = all_distill_feature.view(-1, all_distill_feature.shape[1]).contiguous().cpu()

            poses = np.stack(poses)
            all_rays = torch.cat(all_rays, 0)
            all_rgbs = torch.cat(all_rgbs, 0)

            poses_scene.append(poses)
            all_rays_scene.append(all_rays)
            all_rgbs_scene.append(all_rgbs)
            if distill_active:
                all_distill_features_scene.append(all_distill_feature)
        
        poses_scene = np.stack(poses_scene)
        all_rays_scene = torch.stack(all_rays_scene)
        all_rgbs_scene = torch.stack(all_rgbs_scene)
        
        self.poses = poses_scene
        self.all_rays = all_rays_scene
        self.all_rgbs = all_rgbs_scene
        if distill_active:
            self.all_distill_features = torch.stack(all_distill_features_scene)

        return {"poses": poses_scene, "rays": all_rays_scene, "rgbs": all_rgbs_scene, "distill_features": all_distill_features_scene}


    

    def __getitem__(self, idx):
        img = self.all_rgbs[:, idx]
        rays = self.all_rays[:, idx]
        if self.distill_active:
            distill_feature = self.all_distill_features[:, idx]
        else:
            distill_feature = None

        return {"rgbs": img, "rays": rays, 
                "distill_feature": distill_feature, "idx": idx}
    
    @staticmethod
    def collect_function(batch):
        rays = []
        imgs = []
        distill_features = []
        for i in range(len(batch)):
            rays.append(batch[i]["rays"])
            imgs.append(batch[i]["rgbs"])
            if batch[i]["distill_feature"] is not None:
                distill_features.append(batch[i]["distill_feature"])
        rays = torch.stack(rays, dim=1)
        imgs = torch.stack(imgs, dim=1)
        if len(distill_features) > 0:
            distill_features = torch.stack(distill_features, dim=1)
        else:
            distill_features = None

        return {"rays": rays, "rgbs": imgs, "distill_features": distill_features}


    def __len__(self):
        return self.all_rays.shape[1]
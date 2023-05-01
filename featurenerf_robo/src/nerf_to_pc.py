from representations import make_embedding
import torch
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
import gym
import utils
import warnings

warnings.filterwarnings("ignore")
from copy import deepcopy
from tqdm import tqdm
from config import parse_cfg
from env.wrappers import make_env
import augmentations
import wandb
from algorithms.modules import ContinuousPolicy, RewardPredictor
from termcolor import colored
from logger import Logger
import time
import itertools
import matplotlib.pyplot as plt
from analysis.image import save_feature_map, save_rgb_image

from representations.utils_pixelnerf.model import make_model
from representations.utils_pixelnerf.render import NeRFEmbedRenderer

def point_cloud(depth, cx, cy, fx, fy):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))


def main(cfg):

    # prepare dataset and dataloder
    demonstration_data_dir = os.path.join(cfg.dataset_root, "{}_{}".format(cfg.domain_name, cfg.task_name))
    image_size = cfg.image_size
    num_cameras = ("dynamic" in cfg.camera_mode) + ("static" in cfg.camera_mode) + cfg.num_static_cameras + cfg.num_dynamic_cameras - 2

    # prepare agent dimensions
    action_dim_dict = {"xy":(2,), "xyz":(3,), "xyzw":(4,)}
    state_dim_dict = {"xy":(3,), "xyz":(4,), "xyzw":(4,)}
    action_dim = action_dim_dict[cfg.action_space]
    observation_dim = (num_cameras*3, image_size, image_size) # default hard-coded value
    state_dim = state_dim_dict[cfg.action_space]

    # create env for evaluation
    env = make_env(
        domain_name=cfg.domain_name,
        task_name=cfg.task_name,
        seed=cfg.seed+42,
        episode_length=cfg.episode_length,
        n_substeps=cfg.n_substeps,
        frame_stack=cfg.frame_stack,
        image_size=image_size,
        cameras=cfg.camera_mode,
        render=cfg.render, # Only render if observation type is state
        observation_type="state+image", # state, image, state+image
        action_space=cfg.action_space,
        camera_move_range=cfg.camera_move_range,
        domain_randomization=cfg.domain_randomization,
        num_static_cameras=cfg.num_static_cameras,
        num_dynamic_cameras=cfg.num_dynamic_cameras,
    )

    ##########
    from mj_pc.mj_point_clouds import PointCloudGenerator
    import open3d as o3d


    # Bounds can optionally be provided to crop points outside the workspace
    pc_gen = PointCloudGenerator(env.sim, min_bound=(-1., -1., -1.), max_bound=(1., 1., 1.))
    points_np = pc_gen.generateCroppedPointCloud()
    
    # world_origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([cloud_with_normals, world_origin_axes])


    # sparse random sampling
    ratio = 0.01
    print(f"sample ratio: {ratio} | total number of points: {points_np.shape[0]} | used number of points: {int(points_np.shape[0]*ratio)}")
    points_np = points_np[np.random.choice(points_np.shape[0], int(points_np.shape[0]*ratio), replace=False), :]
    ##########



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create nerf
    net = make_model(cfg.pixelnerf_model)
    net = net.to(device)
    renderer = NeRFEmbedRenderer.from_conf(
        cfg.pixelnerf_renderer, eval_batch_size=cfg.ray_batch_size
    ).cuda()

    # render img from env
    env.reset()
    img_rgb = env.render_obs(width=128, height=128)[0:1]
    img_rgb = torch.from_numpy(img_rgb).float().div(255).cuda().permute(0, 3, 1, 2)
    img_depth = env.render_depth(width=128, height=128)[0:1]
    img_depth = torch.from_numpy(img_depth).float().cuda()
    pose = env.get_camera_extrinsic()[0:1]
    intrinsic = env.get_camera_intrinsic()[0:1]
    pose = torch.from_numpy(pose).float().cuda()
    focal = env.get_focal_length()
    focal = torch.tensor(focal).float().cuda()

    # encode prior
    net.encode(
            img_rgb.to(device),
            pose.to(device),
            focal,
            c=None,
    )
    

    # # get pointcloud from depth
    # focal = focal.cpu().numpy()
    # pc = point_cloud(img_depth[0].cpu().numpy(), cx=128/2, cy=128/2, fx=focal, fy=focal)
    # # filter far points
    # pc = pc[pc[:,:,2] < (pc[:,:,2].max()-0.02 )]
    # pc_copy = deepcopy(pc)
    
    # # transform pointcloud to camera pos
    # pc = pc.reshape(-1, 3)
    # inv_pose = np.linalg.inv(pose[0].cpu().numpy())
    # pc = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1)
    # pc = np.matmul(inv_pose, pc.T).T    
    # pc = pc[:, :3]


    pc = points_np
    pc_copy = deepcopy(pc)


    # fetch nerf rgb and embed using pc
    points = torch.from_numpy(pc).float().cuda().reshape(-1, 3).unsqueeze(0)
    with torch.no_grad():
        
        outputs = net(points, coarse=False, viewdirs=torch.zeros_like(points).cuda())
    rgb = outputs[..., :3]
    sigmas = outputs[..., 3]
    embed = outputs[..., 4:388]

    vis3d = True
    # use visdom to visualize pointcloud in remote server
    if vis3d:
        import visdom
        viz = visdom.Visdom()
        viz.scatter(pc_copy.reshape(-1, 3), win="pc_origin", opts=dict(title="pc_origin", markersize=2))
        # viz pc position
        viz.scatter(points[0].cpu().numpy(), win="pc_transformed", opts=dict(title="pc_transformed", markersize=2))

        # viz pc color
        # convert rgb to int label
        rgb = rgb[0].cpu().numpy()
        rgb = rgb * 255
        rgb = rgb.astype(np.uint8)
        rgb = rgb.sum(-1)
        viz.scatter(points[0].cpu().numpy(), rgb, win="pc_color", opts=dict(title="pc_color", markersize=2))

        # viz embed
        # convert embed to int label
        embed = embed[0].cpu().numpy()
        embed = embed.mean(-1)
        embed = embed * 255
        embed = embed.astype(np.uint8)
        # make min value 1
        embed[embed==0] = 1
        viz.scatter(points[0].cpu().numpy(), embed, win="embed", opts=dict(title="embed", markersize=2))
        


    vis2d = False
    if vis2d:
        # vis rgb, pc, pc rgb
        # vis point rgb
        points = points.cpu().numpy().reshape(-1, 3)
        rgb = rgb.cpu().numpy().reshape(-1, 3)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], c=rgb)
        plt.savefig('data/point_rgb.png')

        # vis point pos
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])
        plt.savefig('data/point_pos.png')

        # vis input
        plt.figure()
        plt.imshow(img_rgb[0].cpu().numpy().transpose(1,2,0))
        plt.savefig('data/input.png')

        # vis depth
        plt.figure()
        plt.imshow(img_depth[0].cpu().numpy())
        plt.savefig('data/depth.png')

    print("Done!")
		


if __name__ == '__main__':
	cfg = parse_cfg(cfg_path="configs", mode="bc")
	main(cfg)


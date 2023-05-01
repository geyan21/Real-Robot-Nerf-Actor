import os
import numpy as np
import cv2
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import transforms3d
import clip
import copy
from voxel_grid_real import VoxelGrid
import torch.nn.functional as F
from math import pi, log
from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce
from network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, ConvTranspose3DBlock, Conv3DUpsampleBlock
from model import *
from agent_function import euler_to_quaternion, perturb_se3, apply_se3_augmentation, get_action, _get_one_hot_expert_actions, _argmax_3d, choose_highest_action

def point_to_voxel_index(
        point: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy

def rand_dist(size, min=-1.0, max=1.0):
    return (max-min) * torch.rand(size) + min


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    
# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

def get_rgb_pcd(pcd_path, cam2base, device):
    cloud = o3d.io.read_point_cloud(pcd_path)
    rgb = np.asarray(cloud.colors)
    pointcloud = np.asarray(cloud.points)

    valid_bool = np.linalg.norm(pointcloud, axis=1) < 3.0
    pointcloud = pointcloud[valid_bool]
    rgb = rgb[valid_bool]

    pointcloud_robot = pointcloud @ cam2base[:3, :3].T + cam2base[:3, 3]
    pointcloud_robot = torch.Tensor(pointcloud_robot).unsqueeze(0)
    rgb = (rgb - 0.5) / 0.5
    rgb = torch.Tensor(rgb).unsqueeze(0)

    return pointcloud_robot, rgb



import wandb
USE_WANDB = True
PROJECT_NAME = "real-robot-peract"
model_name = "peract for kitchen 1" # peract for kitchen 1 , peract for 2 kitchens
if USE_WANDB:
    wandb.init(
            project=PROJECT_NAME, name=model_name, config={"model_for_real": "two_kitchens"},
            # config_exclude_keys=['model_name', 'save_every', 'log_every'],
        )
    wandb.run.log_code(".")  # Need to first enable it on wandb web UI.

pose_all = []
base_dir = '/data/geyan21/projects/real-robot-nerf-actor/data/4_8_2_kitchens/Demos'
model_dir = '/data/geyan21/projects/real-robot-nerf-actor/models'
kitchens = ['kitchen1', 'kitchen2']
tasks = ['faucet', 'oven','teapot']
n_kitchen = 1
n_task = 3
n_demo =  5
n_key = 4 # don't count first key



for kitchen_id in range(n_kitchen):
    for task_id in range(n_task):
        for demo in range(n_demo):
            position_path = os.path.join(base_dir, kitchens[kitchen_id], tasks[task_id], str(demo)+'_xarm_position.txt')
            print(position_path)
            f = open(position_path)
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('[','').replace(']','')
                line = line.split(',')
                for value in line:
                    try:
                        pose_all.append(float(value))
                    except:
                        if 'True' in value:
                            pose_all.append(1.0)
                        else:
                            pose_all.append(0.0)
pose_all = np.array(pose_all).reshape(n_kitchen, n_task, n_demo, n_key+1, -1)
xyz_all = pose_all[:, :, :, :, :3] * 0.001
rotation_all = pose_all[:, :, :, :, 3:6]
gripper_open_all = pose_all[:, :, :, :, -1]
print(xyz_all.shape, rotation_all.shape, gripper_open_all.shape)


#bounds = torch.Tensor([-0.1, -0.1, -0.2, 0.8, 0.8, 0.8])
bounds = torch.Tensor([-0.1, -0.3, -0.2, 0.8, 0.7, 0.7])
vox_size = 100
rotation_resolution = 5
max_num_coords=220000
bs = 1
_num_rotation_classes = 72

desk2camera = [[0.9992296353045714, 0.03351173805892007, -0.020423010095561807, -0.30459545551541667], [0.016111903829177272, 0.12421545808372213, 0.9921244511290155, 0.22325071524867773], [0.03578466828255411, -0.9916892070529564, 0.12357982897943404, 0.7236517078874926], [0.0, 0.0, 0.0, 1.0]]
# RECALIBRATE
adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
adjust_pos_mat = np.array([[1, 0, 0, -0.06], [0, 1, 0, 0.12], [0, 0, 1, -0.005], [0, 0, 0, 1]]) # manually adjust

base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
cam2base = np.linalg.inv(base2camera).reshape(4, 4)

gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
gl2cv_homo = np.eye(4)
gl2cv_homo[:3, :3] = gl2cv
cam2base = cam2base @ gl2cv_homo

device = "cuda:1"
#description = "open the cabinet door"
description = ["Turn the faucet", "Open the top oven door", "Place the Tea Pot on the stove"]
tokens = clip.tokenize(description).numpy()
token_tensor = torch.from_numpy(tokens).to(device)
clip_model, preprocess = clip.load("RN50", device=device)
lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
lang_goal_embs_all = lang_embs.float().detach()
print("lang_goal_embs_all shape:", lang_goal_embs_all.shape)
#lang_goal_embs = lang_embs[0].float().detach().unsqueeze(0)
lang_goal = np.array([description], dtype=object)

voxelizer = VoxelGrid(
    coord_bounds=bounds,
    voxel_size=vox_size,
    device=device,
    batch_size=1,
    feature_size=3,
    max_num_coords=max_num_coords,
)


# initialize PerceiverIO Transformer
perceiver_encoder = PerceiverIO(
    depth=6,
    iterations=1,
    voxel_size=vox_size,
    initial_dim=3 + 3 + 1 + 3,
    low_dim_size=4,
    layer=0,
    num_rotation_classes=72,
    num_grip_classes=2,
    num_collision_classes=2,
    num_latents=2048,
    latent_dim=512,
    cross_heads=1,
    latent_heads=8,
    cross_dim_head=64,
    latent_dim_head=64,
    weight_tie_layers=False,
    activation='lrelu',
    input_dropout=0.1,
    attn_dropout=0.1,
    decoder_dropout=0.0,
    voxel_patch_size=5,
    voxel_patch_stride=5,
    final_dim=64,
)
qnet = copy.deepcopy(perceiver_encoder).to(device)

# load pretrained model
checkpoint = torch.load('/data/geyan21/projects/real-robot-nerf-actor/models/kitchen_1_3_tasks/ckpt_5demo_multi_aug_2048_4_8_4key_150000.pth')
qnet.load_state_dict(checkpoint)

optimizer = torch.optim.Adam(qnet.parameters(), lr=0.0001, weight_decay=0.000001)
_cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
total_loss = 0.
backprop = True
_transform_augmentation = True
#_transform_augmentation_xyz = torch.Tensor([0.125, 0.125, 0.125])
#_transform_augmentation_xyz = torch.Tensor([0.1, 0.05, 0.05])
#_transform_augmentation_xyz = torch.Tensor([0.1, 0.1, 0.1])
# _transform_augmentation_xyz = torch.Tensor([0.2, 0.05, 0.05])
_transform_augmentation_xyz = torch.Tensor([0.1, 0.05, 0.1])
#_transform_augmentation_xyz = torch.Tensor([0.125, 0, 0])

for iter in range(150010):
    kitchen_id = random.randint(0, n_kitchen-1)
    task_id = random.randint(0, n_task-1)
    demo = random.randint(0, n_demo-1)
    i = random.randint(0, n_key-1)
    j = 1

    trans_indicies, rot_and_grip_indicies, ignore_collisions = get_action(xyz_all[kitchen_id][task_id][demo][i+j], rotation_all[kitchen_id][task_id][demo][i+j], gripper_open_all[kitchen_id][task_id][demo][i+j], 1, bounds, vox_size, rotation_resolution)
    trans_indicies = torch.Tensor(trans_indicies).to(device).unsqueeze(0)
    rot_and_grip_indicies  = torch.Tensor(rot_and_grip_indicies).to(device).unsqueeze(0)
    ignore_collisions = torch.Tensor(ignore_collisions).to(device).unsqueeze(0)
    #print(trans_indicies.shape, rot_and_grip_indicies.shape, ignore_collisions.shape)

    action_trans = trans_indicies.int()
    action_rot_grip = rot_and_grip_indicies.int()
    action_ignore_collisions = ignore_collisions.int()

    pcd_path = os.path.join(base_dir,kitchens[kitchen_id], tasks[task_id], 'real' + str(demo), 'pcd' + str(i) + '.ply')
    lang_goal_embs = lang_goal_embs_all[task_id].unsqueeze(0)

    pointcloud_robot, rgb = get_rgb_pcd(pcd_path, cam2base, device)



    trans_indicies_prev, rot_and_grip_indicies_prev, _ = get_action(xyz_all[kitchen_id][task_id][demo][i], rotation_all[kitchen_id][task_id][demo][i],
                                                                            gripper_open_all[kitchen_id][task_id][demo][i], 1, bounds,
                                                                            vox_size, rotation_resolution)
    xyz_prev = torch.Tensor(xyz_all[kitchen_id][task_id][demo][i]).unsqueeze(0)

    
    trans_indicies_prev = torch.Tensor(trans_indicies_prev).to(device).unsqueeze(0)
    #rot_and_grip_indicies_prev = torch.Tensor(rot_and_grip_indicies_prev).to(device).unsqueeze(0)
    rot_and_grip_indicies_prev = torch.Tensor(rot_and_grip_indicies_prev).unsqueeze(0)
    action_trans_prev = trans_indicies_prev.int()
    action_rot_grip_prev =  rot_and_grip_indicies_prev.int()


    if _transform_augmentation:
        action_trans_cat = torch.cat([action_trans_prev, action_trans], 0)
        #print(demo, i, 'before', action_trans_prev, action_trans )
        xyz_cat = torch.cat([xyz_prev, torch.Tensor(xyz_all[kitchen_id][task_id][demo][i+j]).unsqueeze(0)], 0)
        pointcloud_robot_cat = torch.cat([pointcloud_robot, pointcloud_robot], 0)
        bounds_cat = torch.cat([bounds.unsqueeze(0), bounds.unsqueeze(0)], 0)

        # print(action_trans_cat.shape, xyz_cat.shape, pointcloud_robot_cat.shape, bounds_cat.shape)

        action_trans_cat, \
        pointcloud_robot_cat = apply_se3_augmentation([pointcloud_robot_cat],
                                         xyz_cat,
                                         action_trans_cat.cpu(),
                                         bounds_cat,
                                         1,
                                         _transform_augmentation_xyz,
                                         vox_size,
                                         "cpu")
        pointcloud_robot = pointcloud_robot_cat[0][:1]
        action_trans = action_trans_cat[1:]
        action_trans_prev = action_trans_cat[:1]
        #print(demo, i, 'after', action_trans_prev, action_trans)

    proprio = torch.cat([action_trans_prev, action_rot_grip_prev], 1).to(device).float()
    #print(proprio.shape, proprio)

    voxel_grid = voxelizer.coords_to_bounding_voxel_grid(
        pointcloud_robot, coord_features=rgb, coord_bounds=bounds)
    #print("voxel_grid:", voxel_grid.shape)
    voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().to(device)

    #print(demo, i, xyz_all[demo][i], rotation_all[demo][i], gripper_open_all[demo][i], pcd_path, proprio)
    q_trans, rot_grip_q, collision_q = qnet(voxel_grid, proprio, lang_goal_embs)
    #print(q_trans.shape, rot_and_grip_q.shape, collision_q.shape)


    action_trans_one_hot, action_rot_x_one_hot, \
        action_rot_y_one_hot, action_rot_z_one_hot, \
        action_grip_one_hot, action_collision_one_hot = _get_one_hot_expert_actions(bs,
                                                                                     action_trans,
                                                                                     action_rot_grip,
                                                                                     action_ignore_collisions,
                                                                                     vox_size,
                                                                                     _num_rotation_classes,
                                                                                     device=device)

    if backprop:
        # cross-entropy loss
        trans_loss = _cross_entropy_loss(q_trans.view(bs, -1),
                                              action_trans_one_hot.argmax(-1))

        rot_grip_loss = 0.
        rot_grip_loss += _cross_entropy_loss(
            rot_grip_q[:, 0 * _num_rotation_classes:1 * _num_rotation_classes],
            action_rot_x_one_hot.argmax(-1))
        rot_grip_loss += _cross_entropy_loss(
            rot_grip_q[:, 1 * _num_rotation_classes:2 * _num_rotation_classes],
            action_rot_y_one_hot.argmax(-1))
        rot_grip_loss += _cross_entropy_loss(
            rot_grip_q[:, 2 * _num_rotation_classes:3 * _num_rotation_classes],
            action_rot_z_one_hot.argmax(-1))
        rot_grip_loss += _cross_entropy_loss(rot_grip_q[:, 3 * _num_rotation_classes:],
                                                  action_grip_one_hot.argmax(-1))

        collision_loss = _cross_entropy_loss(collision_q,
                                                  action_collision_one_hot.argmax(-1))

        total_loss = trans_loss + rot_grip_loss + collision_loss
        total_loss = total_loss.mean()

        # backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss = total_loss.item()
        if (iter + 1) % 50 == 0:
            log_dict = {
                        'n_iter': iter,
                        'loss_pred': total_loss,
                    }
            wandb.log(log_dict)
            print(iter, kitchen_id, task_id, demo, i, total_loss)

    # choose best action through argmax
    # coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = choose_highest_action(q_trans,
    #                                                                                              rot_grip_q,
    #                                                                                              collision_q,
    #                                                                                              rotation_resolution)
    # # discrete to continuous translation action
    # bounds_new = bounds.unsqueeze(0).to(device)
    # res = (bounds_new[:, 3:] - bounds_new[:, :3]) / vox_size
    # continuous_trans = bounds_new[:, :3] + res * coords_indicies.int() + res / 2

    if (iter+1) % 10000 == 0:
       save_checkpoint(qnet, model_dir + '/kitchen_1_3_tasks/ckpt_5demo_multi_aug_2048_4_8_4key_' + str(iter+1+150000) + '.pth')

    #print(i, continuous_trans)

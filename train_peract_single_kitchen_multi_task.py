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
import pdb
import time

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


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module): # is all you need. Living up to its name.
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


def euler_to_quaternion(r):
    roll = r[0]
    pitch = r[1]
    yaw = r[2]
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def perturb_se3(pcd,
                trans_shift_4x4,
                rot_shift_4x4,
                action_gripper_4x4,
                bounds):
    """ Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # baatch bounds if necessary
    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p = p.permute(0, 2, 1)
        p_shape = p.shape
        num_points = p_shape[-1]
        #num_points = p_shape[-1] * p_shape[-2]

        action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(p_flat_4x1_action_origin.transpose(2, 1),
                                                       rot_shift_4x4).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(action_then_trans_3x1[:, 0],
                                              min=bounds_x_min, max=bounds_x_max)
        action_then_trans_3x1_y = torch.clamp(action_then_trans_3x1[:, 1],
                                              min=bounds_y_min, max=bounds_y_max)
        action_then_trans_3x1_z = torch.clamp(action_then_trans_3x1[:, 2],
                                              min=bounds_z_min, max=bounds_z_max)
        action_then_trans_3x1 = torch.stack([action_then_trans_3x1_x,
                                             action_then_trans_3x1_y,
                                             action_then_trans_3x1_z], dim=1)

        # shift back the origin
        perturbed_p_flat_3x1 = perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1

        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p.permute(0, 2, 1))
    return perturbed_pcd


def apply_se3_augmentation(pcd,
                           action_gripper_pose_xyz,
                           action_trans,
                           bounds,
                           layer,
                           trans_aug_range,
                           voxel_size,
                           device):
    """ Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose_xyz
    # action_gripper_quat_wxyz = torch.cat((action_gripper_pose[:, 6].unsqueeze(1),
    #                                       action_gripper_pose[:, 3:6]), dim=1)
    # action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    # action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.)
    # perturbed_rot_grip = torch.full_like(action_rot_grip, -1.)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            raise Exception('Failing to perturb action and keep it within bounds.')

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)

        trans_shift = trans_range * rand_dist((2, 3)).to(device=device)
        #print(trans_shift)
        trans_shift[1] = trans_shift[0]
        #trans_shift = torch.tile(trans_shift, (bs,1))
        #print(trans_shift.shape, trans_shift)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        #print('trans_shift:', trans_shift)

        # sample rotation perturbation at specified resolution and range
        # roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        # pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        # yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)
        #
        # roll = utils.rand_discrete((bs, 1),
        #                            min=-roll_aug_steps,
        #                            max=roll_aug_steps) * np.deg2rad(rot_aug_resolution)
        # pitch = utils.rand_discrete((bs, 1),
        #                             min=-pitch_aug_steps,
        #                             max=pitch_aug_steps) * np.deg2rad(rot_aug_resolution)
        # yaw = utils.rand_discrete((bs, 1),
        #                           min=-yaw_aug_steps,
        #                           max=yaw_aug_steps) * np.deg2rad(rot_aug_resolution)
        # rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(torch.cat((roll, pitch, yaw), dim=1), "XYZ")
        rot_shift_4x4 = identity_4x4.detach().clone()
        #rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        # perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(perturbed_action_gripper_4x4[:, :3, :3])
        # perturbed_action_quat_xyzw = torch.cat([perturbed_action_quat_wxyz[:, 1:],
        #                                         perturbed_action_quat_wxyz[:, 0].unsqueeze(1)],
        #                                        dim=1).cpu().numpy()

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = point_to_voxel_index(perturbed_action_trans[b], voxel_size, bounds_np)
            trans_indicies.append(trans_idx.tolist())

            # quat = perturbed_action_quat_xyzw[b]
            # quat = utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            # if quat[-1] < 0:
            #     quat = -quat
            # disc_rot = utils.quaternion_to_discrete_euler(quat, rot_resolution)
            # rot_grip_indicies.append(disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())])

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        #perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(device=device)

    #print(action_trans, perturbed_trans)
    action_trans = perturbed_trans
    #action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    #pcd_ori = pcd
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    #print(pcd_ori[0][0][:5], pcd[0][0][:5], trans_shift)

    return action_trans, pcd


def get_action(xyz, rotation, gripper_open, ignore_collisions, bounds, vox_size, rotation_resolution):
    # rotation = euler_to_quaternion(rotation)
    # quat = utils.normalize_quaternion(rotation)
    # if quat[-1] < 0:
    #     quat = -quat
    # disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    disc_rot = ((np.array(rotation) + 180) / rotation_resolution).astype(int) -1
    #print(rotation, disc_rot)
    trans_indicies = []
    ignore_collisions = [int(ignore_collisions)]
    index = point_to_voxel_index(
        xyz, vox_size, bounds)
    trans_indicies.extend(index.tolist())
    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(gripper_open)
    rot_and_grip_indicies.extend([int(gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions


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


class PerceiverIO(nn.Module):
    def __init__(
            self,
            depth,  # number of self-attention layers
            iterations,  # number cross-attention iterations (PerceiverIO uses just 1)
            voxel_size,  # N voxels per side (size: N*N*N)
            initial_dim,  # 10 dimensions - dimension of the input sequence to be encoded
            low_dim_size,
            # 4 dimensions - proprioception: {gripper_open, left_finger_joint, right_finger_joint, timestep}
            layer=0,
            num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
            num_grip_classes=2,  # open or not open
            num_collision_classes=2,  # collisions allowed or not allowed
            input_axis=3,  # 3D tensors have 3 axes
            num_latents=2048,  # number of latent vectors
            im_channels=64,  # intermediate channel size
            latent_dim=512,  # dimensions of latent vectors
            cross_heads=1,  # number of cross-attention heads
            latent_heads=8,  # number of latent heads
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
            weight_tie_layers=False,
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            voxel_patch_size=5,  # intial patch size
            voxel_patch_stride=5,  # initial stride to patchify voxel input
            final_dim=64,  # final dimensions of features
    ):
        super().__init__()
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout

        # patchified input dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # 64 voxel features + 64 proprio features
        self.input_dim_before_seq = self.im_channels * 2
        #self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        lang_emb_dim, lang_max_seq_len = 512, 77
        self.pos_encoding = nn.Parameter(torch.randn(1,
                                                     lang_max_seq_len + spatial_size ** 3,
                                                     self.input_dim_before_seq))

        # voxel input preprocessing encoder
        self.input_preprocess = Conv3DBlock(
            self.init_dim, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )

        # proprio preprocessing encoder
        self.proprio_preprocess = DenseBlock(
            7, self.im_channels, norm=None, activation=activation,
        )

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels, self.im_channels,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation)

        # lang preprocess
        self.lang_preprocess = nn.Linear(lang_emb_dim, self.im_channels * 2)
        #self.lang_preprocess = nn.Linear(lang_emb_dim, self.im_channels * 1)

        # pooling functions
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size,
            self.im_channels)
        flat_size = self.im_channels * 4

        # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, self.input_dim_before_seq, heads=cross_heads,
                                          dim_head=cross_dim_head, dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads,
                                                    dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(self.input_dim_before_seq,
                                          Attention(self.input_dim_before_seq, latent_dim, heads=cross_heads,
                                                    dim_head=cross_dim_head,
                                                    dropout=decoder_dropout),
                                          context_dim=latent_dim)

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, self.final_dim,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation,
        )

        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self.input_dim_before_seq)

        flat_size += self.input_dim_before_seq * 4

        # final layers
        self.final = Conv3DBlock(
            self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1, norm=None, activation=activation)

        # 100x100x100x64 -> 100x100x100x1 decoder for translation Q-values
        self.trans_decoder = Conv3DBlock(
            self.final_dim, 1, kernel_sizes=3, strides=1,
            norm=None, activation=None,
        )

        # final 3D softmax
        self.ss_final = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size,
            self.im_channels)

        flat_size += self.im_channels * 4

        # MLP layers
        self.dense0 = DenseBlock(
            flat_size, 256, None, activation)
        self.dense1 = DenseBlock(
            256, self.final_dim, None, activation)

        # 1x64 -> 1x(72+72+72+2+2) decoders for rotation, gripper open, and collision Q-values
        self.rot_grip_collision_ff = DenseBlock(self.final_dim,
                                                self.num_rotation_classes * 3 + \
                                                self.num_grip_classes + \
                                                self.num_collision_classes,
                                                None, None)

    def forward(
            self,
            ins,
            proprio,
            lang_goal_embs,
            mask=None,
    ):
        # preprocess
        d0 = self.input_preprocess(ins)  # [B,10,100,100,100] -> [B,64,100,100,100]

        # aggregated features from 1st softmax and maxpool for MLP decoders
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        # patchify input (5x5x5 patches)
        ins = self.patchify(d0)  # [B,64,100,100,100] -> [B,64,20,20,20]

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert len(axis) == self.input_axis, 'input must have the same number of axis as input_axis'

        # concat proprio
        p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
        p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
        ins = torch.cat([ins, p], dim=1)  # [B,128,20,20,20]

        # channel last
        ins = rearrange(ins, 'b d ... -> b ... d')  # [B,20,20,20,128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten voxel grid into sequence
        ins = rearrange(ins, 'b ... d -> b (...) d')  # [B,8000,128]

        # append language features as sequence
        l = self.lang_preprocess(lang_goal_embs)  # [B,77,1024] -> [B,77,128]

        ins = torch.cat((l, ins), dim=1)  # [B,8077,128]

        # add learable pos encoding
        ins = ins + self.pos_encoding

        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=ins, mask=mask) + x
            x = cross_ff(x) + x

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(ins, context=x)
        latents = latents[:, l.shape[1]:]

        # reshape back to voxel grid
        latents = latents.view(b, *ins_orig_shape[1:-1], latents.shape[-1])  # [B,20,20,20,64]
        latents = rearrange(latents, 'b ... d -> b d ...')  # [B,64,20,20,20]

        # aggregated features from 2nd softmax and maxpool for MLP decoders
        feats.extend([self.ss1(latents.contiguous()), self.global_maxp(latents).view(b, -1)])

        # upsample layer
        u0 = self.up0(latents)  # [B,64,100,100,100]

        # skip connection like in UNets
        u = self.final(torch.cat([d0, u0], dim=1))  # [B,64+64,100,100,100] -> [B,64,100,100,100]

        # translation decoder
        trans = self.trans_decoder(u)  # [B,64,100,100,100] -> [B,1,100,100,100]

        # aggregated features from final softmax and maxpool for MLP decoders
        feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(b, -1)])

        # decoder MLP layers for rotation, gripper open, and collision
        dense0 = self.dense0(torch.cat(feats, dim=1))
        dense1 = self.dense1(dense0)  # [B,72*3+2+2]

        # format output
        rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
        rot_and_grip_out = rot_and_grip_collision_out[:, :-self.num_collision_classes]
        collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes:]

        return trans, rot_and_grip_out, collision_out


def _get_one_hot_expert_actions(batch_size,
                                action_trans,
                                action_rot_grip,
                                action_ignore_collisions,
                                voxel_size,
                                _num_rotation_classes,
                                device):
    bs = batch_size

    # initialize with zero tensors
    action_trans_one_hot = torch.zeros((bs, voxel_size, voxel_size, voxel_size), dtype=int,
                                       device=device)
    action_rot_x_one_hot = torch.zeros((bs, _num_rotation_classes), dtype=int, device=device)
    action_rot_y_one_hot = torch.zeros((bs, _num_rotation_classes), dtype=int, device=device)
    action_rot_z_one_hot = torch.zeros((bs, _num_rotation_classes), dtype=int, device=device)
    action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
    action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

    # fill one-hots
    for b in range(bs):
        # translation
        gt_coord = action_trans[b, :]
        action_trans_one_hot[b, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # rotation
        gt_rot_grip = action_rot_grip[b, :]
        action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
        action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
        action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
        action_grip_one_hot[b, gt_rot_grip[3]] = 1

        # ignore collision
        gt_ignore_collisions = action_ignore_collisions[b, :]
        action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

    # flatten trans
    action_trans_one_hot = action_trans_one_hot.view(bs, -1)

    return action_trans_one_hot, \
        action_rot_x_one_hot, \
        action_rot_y_one_hot, \
        action_rot_z_one_hot, \
        action_grip_one_hot, \
        action_collision_one_hot


def _argmax_3d(tensor_orig):
    b, c, d, h, w = tensor_orig.shape  # c will be one
    idxs = tensor_orig.view(b, c, -1).argmax(-1)
    indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
    return indices


def choose_highest_action(q_trans, q_rot_grip, q_collision, rotation_resolution):
    coords = _argmax_3d(q_trans)
    rot_and_grip_indicies = None
    if q_rot_grip is not None:
        q_rot = torch.stack(torch.split(
            q_rot_grip[:, :-2],
            int(360 // rotation_resolution),
            dim=1), dim=1)
        rot_and_grip_indicies = torch.cat(
            [q_rot[:, 0:1].argmax(-1),
             q_rot[:, 1:2].argmax(-1),
             q_rot[:, 2:3].argmax(-1),
             q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
        ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
    return coords, rot_and_grip_indicies, ignore_collision

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

import wandb
USE_WANDB = False
PROJECT_NAME = "real-robot-peract"
model_name = "peract for kitchen 1" # peract for kitchen 1 , peract for 2 kitchens
if USE_WANDB:
    wandb.init(
            project=PROJECT_NAME, name=model_name, config={"model_for_real": "two_kitchens"},
            # config_exclude_keys=['model_name', 'save_every', 'log_every'],
        )
    wandb.run.log_code(".")  # Need to first enable it on wandb web UI.

pose_all = []
base_dir = '/data/geyan21/projects/real-robot-nerf-actor/data/kitchen1_box_generalize'
model_dir = '/data/geyan21/projects/real-robot-nerf-actor/models_kitchen1_box_generalize'
kitchens = ['Peract_kitchen_generalize']
tasks = ['grasp_green', 'grasp_red','grasp_white']
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

desk2camera = [[0.9992322829292472, -0.03876636955809762, 0.005658033517907016, -0.19047521587073285], [-0.0028732620580650126, 0.07151729043294505, 0.9974352217233334, 0.25503978911275005], [-0.03907158964196928, -0.9966857306896767, 0.07135099944944548, 0.7605795247007009], [0.0, 0.0, 0.0, 1.0]]
# RECALIBRATE
adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
adjust_pos_mat = np.array([[1, 0, 0, -0.10], [0, 1, 0, 0.14], [0, 0, 1, 0.01], [0, 0, 0, 1]])

base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
cam2base = np.linalg.inv(base2camera).reshape(4, 4)

gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
gl2cv_homo = np.eye(4)
gl2cv_homo[:3, :3] = gl2cv
cam2base = cam2base @ gl2cv_homo

device = "cuda:2"
#description = "open the cabinet door"
# description = ["Turn the faucet", "Open the top oven door", "Place the Tea Pot on the stove"]
description = ["Place the green box into the right tank", "Place the red box into the right tank", "Place the white box into the right tank"]
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
# checkpoint = torch.load('models_kitchen1_box_generalize/ckpt_5demo_multi_aug_2048_5_7_4key_49999.pth')
# qnet.load_state_dict(checkpoint)

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

start_time = time.time()
for iter in range(200010):
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
        if (iter + 1) % 20 == 0:
            elapsed_time = time.time() - start_time
            log_dict = {
                        'n_iter': iter,
                        'loss_pred': total_loss,
                    }
            if USE_WANDB:
                wandb.log(log_dict)
            print(iter, kitchen_id, task_id, demo, i, total_loss)
            print(f"Iteration:{iter}, demo:{demo}, step:{i}, Loss:{total_loss}, Time:{elapsed_time}")
            start_time = time.time()

    # choose best action through argmax
    # coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = choose_highest_action(q_trans,
    #                                                                                              rot_grip_q,
    #                                                                                              collision_q,
    #                                                                                              rotation_resolution)
    # # discrete to continuous translation action
    # bounds_new = bounds.unsqueeze(0).to(device)
    # res = (bounds_new[:, 3:] - bounds_new[:, :3]) / vox_size
    # continuous_trans = bounds_new[:, :3] + res * coords_indicies.int() + res / 2

    if (iter+1) % 3000 == 0:
       save_checkpoint(qnet, model_dir + '/ckpt_5demo_multi_aug_2048_5_7_4key_' + str(iter+1+50000) + '.pth')

    #print(i, continuous_trans)

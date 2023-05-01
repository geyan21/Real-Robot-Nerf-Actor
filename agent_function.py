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

def rand_dist(size, min=-1.0, max=1.0):
    return (max-min) * torch.rand(size) + min


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

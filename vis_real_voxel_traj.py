import os
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import transforms3d
import utils
from utils import visualise_voxel
from voxel_grid_real import VoxelGrid


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
        #print(p.shape)
        p_shape = p.shape
        #num_points = p_shape[-1] * p_shape[-2]
        num_points = p_shape[-1]
        action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        print(p_flat.shape, action_trans_3x1.shape)
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
        perturbed_pcd.append(perturbed_p)
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
    #print(bs)

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
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

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

            trans_idx = utils.point_to_voxel_index(perturbed_action_trans[b], voxel_size, bounds_np)
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

    action_trans = perturbed_trans
    #action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return action_trans, pcd


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo


def pointcloud_from_depth_and_camera_params(
        depth: np.ndarray, extrinsics: np.ndarray,
        intrinsics: np.ndarray) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
    :return: A numpy array of size (width, height, 3)
    """
    # make sure intrinsic is non-negative
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(
        pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords

#
# def _get_action(
#         obs_tp1: Observation,
#         obs_tm1: Observation,
#         rlbench_scene_bounds: List[float], # metric 3D bounds of the scene
#         voxel_sizes: List[int],
#         bounds_offset: List[float],
#         rotation_resolution: int,
#         crop_augmentation: bool):
#     quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
#     if quat[-1] < 0:
#         quat = -quat
#     disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
#     disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)
#
#     attention_coordinate = obs_tp1.gripper_pose[:3]
#     trans_indicies, attention_coordinates = [], []
#     bounds = np.array(rlbench_scene_bounds)
#     ignore_collisions = int(obs_tm1.ignore_collisions)
#     for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
#         if depth > 0:
#             if crop_augmentation:
#                 shift = bounds_offset[depth - 1] * 0.75
#                 attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
#             bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
#                                      attention_coordinate + bounds_offset[depth - 1]])
#         index = utils.point_to_voxel_index(
#             obs_tp1.gripper_pose[:3], vox_size, bounds)
#         trans_indicies.extend(index.tolist())
#         res = (bounds[3:] - bounds[:3]) / vox_size
#         attention_coordinate = bounds[:3] + res * index
#         attention_coordinates.append(attention_coordinate)
#
#     rot_and_grip_indicies = disc_rot.tolist()
#     grip = float(obs_tp1.gripper_open)
#     rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
#     return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
#         [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates

# depth_path = '/data2/yuyingge/real_kitchen/color3.npz.npy'
# rgb_path = '/data2/yuyingge/real_kitchen/color3.jpg'
#
# depthmap = np.load(depth_path)[:, 80:560]
# print(depthmap.shape, depthmap.max(), depthmap.min())
#
# rgb = cv2.imread(rgb_path) / 255
# # rgb = rgb[:, 80:560]
# rgb = rgb[:, 80:560]
# rgb = (rgb - 0.5) / 0.5
# rgb = rgb.reshape(-1, 3)
# print(rgb.max(), rgb.min())
# rgb = torch.Tensor(rgb).unsqueeze(0)
#
# intrinsics = np.array([612.76031494, 0, 322.287323, 0, 613.31005859, 234.6758728, 0, 0, 1]).reshape(3,3)
# pointcloud = get_pointcloud(depthmap, intrinsics).reshape(-1, 3)

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def extract_keyframe(rotation_all, gripper_open_all):
    last_gripper_open = 0.0
    n_frames = rotation_all.shape[0]
    keys = []
    for i in range(n_frames):
        gripper_open = gripper_open_all[i]
        if gripper_open != last_gripper_open:
            keys.append(i)
            last_gripper_open = gripper_open
            #print(i, gripper_open)
    final_roll = rotation_all[-1][0]
    frame_idx = np.where(rotation_all[:, 0]==final_roll)[0][0]
    keys.append(frame_idx)
    keys.append(n_frames-1)
    keys.sort()
    return keys


position_dir = '/data2/yuyingge/real_kitchen/kitchen_2_26_real_time'
n_demo =  8
key_all = []
xyz_all = []
rotation_all = []
gripper_open_all = []

for demo in range(n_demo):
    poses = []
    position_path = os.path.join(position_dir, str(demo)+'_xarm_position.txt')
    f = open(position_path)
    lines = f.readlines()
    for line in lines:
        line = line.strip().replace('[','').replace(']','')
        line = line.split(',')
        for value in line:
            try:
                poses.append(float(value))
            except:
                if 'True' in value:
                    poses.append(1.0)
                else:
                    poses.append(0.0)

    poses = np.array(poses).reshape(-1, 7)
    xyzs = poses[:, :3] * 0.001
    rotations = poses[:, 3:6]
    gripper_opens = poses[:, -1]

    keys = extract_keyframe(rotations, gripper_opens)
    key_all.append(keys)
    xyz_key = xyzs[keys]
    rotation_key = rotations[keys]
    gripper_open_key = gripper_opens[keys]

    xyz_all.append(xyz_key)
    rotation_all.append(rotation_key)
    gripper_open_all.append(gripper_open_key)
print(key_all)


bounds = torch.Tensor([-0.1, -0.3, -0.2, 0.8, 0.7, 0.7])
#_transform_augmentation_xyz = torch.Tensor([0.125, 0.125, 0.125])
_transform_augmentation_xyz = torch.Tensor([0.1, 0.05, 0.05])
vox_size = 100
_transform_augmentation = False

desk2camera = [[0.9995792239014981, 0.02157846929671545, -0.01938413803317837, -0.17597187766266176], [0.018393687027616863, 0.04518885264459464, 0.9988091108285883, 0.29348066683993745], [0.022428718688518135, -0.9987453815960682, 0.04477293064470426, 0.8964792271178371], [0.0, 0.0, 0.0, 1.0]]
# RECALIBRATE
adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
adjust_pos_mat = np.array([[1, 0, 0, -0.21], [0, 1, 0, -0.04], [0, 0, 1, 0], [0, 0, 0, 1]]) # manually adjust
base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
cam2base = np.linalg.inv(base2camera).reshape(4, 4)

gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
gl2cv_homo = np.eye(4)
gl2cv_homo[:3, :3] = gl2cv
cam2base = cam2base @ gl2cv_homo

for demo in range(n_demo):
  n_key = len(key_all[demo]) - 1
  for i in range(n_key):
    pcd_indx = key_all[demo][i]
    cloud = o3d.io.read_point_cloud("/data2/yuyingge/real_kitchen/kitchen_2_26_real_time/real" + str(demo) + "/pcd" + str(pcd_indx) + ".ply")
    rgb = np.asarray(cloud.colors)
    pointcloud = np.asarray(cloud.points)
    print(rgb.shape, pointcloud.shape)
    # rgb = (rgb - 0.5) / 0.5
    # rgb = torch.Tensor(rgb).unsqueeze(0)


    valid_bool = np.linalg.norm(pointcloud, axis=1) < 3.0
    pointcloud = pointcloud[valid_bool]
    rgb = rgb[valid_bool]

    pointcloud_robot = pointcloud @ cam2base[:3, :3].T + cam2base[:3, 3]
    pointcloud_robot = torch.Tensor(pointcloud_robot).unsqueeze(0)
    rgb = (rgb - 0.5) / 0.5
    rgb = torch.Tensor(rgb).unsqueeze(0)


    print(pointcloud_robot[:, :, 0].max(), pointcloud_robot[:,:, 0].min())
    print(pointcloud_robot[:, :, 1].max(), pointcloud_robot[:,:, 1].min())
    print(pointcloud_robot[:, :, 2].max(), pointcloud_robot[:,:, 2].min())

    vis_gt_coord = utils.point_to_voxel_index(
        xyz_all[demo][i+1], 100, bounds)

    if _transform_augmentation:
        action_trans = torch.Tensor(vis_gt_coord).unsqueeze(0)
        action_trans, \
        pointcloud_robot = apply_se3_augmentation([pointcloud_robot],
                                         torch.Tensor(pose_all[i]).unsqueeze(0),
                                         action_trans,
                                         bounds.unsqueeze(0),
                                         1,
                                         _transform_augmentation_xyz,
                                         vox_size,
                                         device="cpu")
        pointcloud_robot = pointcloud_robot[0].permute(0, 2, 1)
        action_trans = action_trans[0]
        #print(action_trans[0].shape, pointcloud_robot[0].shape)
        vis_gt_coord = action_trans.detach().numpy()
        #print(vis_gt_coord)

    voxelizer = VoxelGrid(
        coord_bounds=bounds,
        voxel_size=100,
        device="cpu",
        batch_size=1,
        feature_size=3,
        max_num_coords=220000,
    )

    voxel_grid = voxelizer.coords_to_bounding_voxel_grid(
                pointcloud_robot, coord_features=rgb, coord_bounds=bounds)
    #print(voxel_grid.shape)

    vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()



    vis_gt_coord = np.array(vis_gt_coord).astype(np.int32).reshape(1, -1)
    rotation_amount = -90
    rendered_img = visualise_voxel(vis_voxel_grid[0],
                                   None,
                                   None,
                                   vis_gt_coord[0],
                                   voxel_size=0.045,
                                   rotation_amount=np.deg2rad(rotation_amount))

    fig = plt.figure(figsize=(15, 15))
    plt.imshow(rendered_img)
    plt.axis('off')
    plt.savefig('/data2/yuyingge/peract/real_kitchen/2_26_ar_real_real_time_gt_new_bound_new_' + str(demo) + '_' + str(i) + '.jpg')

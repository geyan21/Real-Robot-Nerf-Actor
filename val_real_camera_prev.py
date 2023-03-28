import os
import numpy as np
import cv2
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import transforms3d
import clip
import copy
from math import pi
import open3d as o3d
import numpy as np
import transforms3d
import math
from xarm.wrapper import XArmAPI
import time
from voxel_grid_real import VoxelGrid
import torch.nn.functional as F
from math import pi, log
from functools import wraps
from torch import nn, einsum
import pyrealsense2 as rs
from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce
from network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, ConvTranspose3DBlock, Conv3DUpsampleBlock
from scipy.spatial.transform import Rotation as R


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


def get_from_camera(seq):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()


    depth_sensor.set_option(rs.option.exposure, 4000) #12000
    depth_sensor.set_option(rs.option.depth_units, 0.0001)
    print("Depth Scale is: " , depth_scale)

    # # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    # # depth_sensor.set_option(rs.option.exposure, 4000) #12000
    # depth_sensor.set_option(rs.option.depth_units, 0.0001)
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
    spatial.set_option(rs.option.filter_smooth_delta, 1)
    spatial.set_option(rs.option.holes_fill, 1)
    temporal = rs.temporal_filter()
    # temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
    # temporal.set_option(rs.option.filter_smooth_delta, 1)
    # print("Depth Scale is: " , depth_scale)

    # Adjust exposure of rgb/color sensor
    color_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    # color_sensor.set_option(rs.option.exposure, 200) # for block
    color_sensor.set_option(rs.option.exposure, 70) # 200
    #color_sensor.set_option(rs.option.exposure, 150)
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #2 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    for i in range(30):
        pipeline.wait_for_frames() # wait for autoexposure to settle

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Get the point cloud
    cloud_path = 'pcd_data/pcd_'+str(seq)+'.ply'
    for i in range(1):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame:
            continue
        depth_frame = depth_to_disparity.process(depth_frame)
        filtered_depth = spatial.process(depth_frame)
        filtered_depth = temporal.process(filtered_depth)
        filtered_depth = disparity_to_depth.process(filtered_depth)
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        cloud = pc.calculate(filtered_depth)
        if i == 0 and os.path.exists(cloud_path):
            os.remove(cloud_path)
        cloud.export_to_ply(cloud_path, color_frame)
        print(cloud)


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


def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [pitch, roll, yaw]


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
            num_latents=512,  # number of latent vectors
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

def _softmax_q(q):
    q_shape = q.shape
    return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)


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

def visualize_pcd(path, cam2base):
    cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

    # desk2camera = [[0.9995041445583243, 0.015233611744522804, 0.027557251023093778, -0.06908457188626575], [-0.02793638244421772, 0.025262065250222506, 0.9992904415610229, 0.2559835936616738], [0.014526649533291223, -0.9995647878614526, 0.025675110921401928, 0.8168659375022858], [0.0, 0.0, 0.0, 1.0]]
    # # RECALIBRATE
    # adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # adjust_pos_mat = np.array([[1, 0, 0, -0.2], [0, 1, 0, -0.06], [0, 0, 1, 0], [0, 0, 0, 1]]) # manually adjust 

    # base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
    # cam2base = np.linalg.inv(base2camera).reshape(4,4)
    # #print(cam2base.reshape(-1).tolist())

    # gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
    # #print(gl2cv)
    # gl2cv_homo = np.eye(4)
    # gl2cv_homo[:3, :3] = gl2cv
    # cam2base = cam2base @ gl2cv_homo

    valid_bool = np.linalg.norm(points, axis=1) < 3.0
    points = points[valid_bool]
    colors = colors[valid_bool]

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_points = points @ cam2base[:3, :3].T + cam2base[:3, 3]
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pcd.colors = o3d.utility.Vector3dVector(colors)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    
    o3d.visualization.draw_geometries(geometry_list=[transformed_pcd, coordinate])

def init_robot():
    arm = XArmAPI("192.168.1.209")
    # arm = XArmAPI(ip)
    arm.motion_enable(enable=True) # Enables motion. Set True so the robot motors don't turn on and off all the time
    print("Arm's current position: ", arm.get_position()) # Curent position of the robot
    # print(arm.get_inverse_kinematics(pose)) # Get the joint angles from teach pendant pose
    print("joint speed limit: ", arm.joint_speed_limit) # minimum is 0.057 deg and max is 180 deg. Start with 10 deg per sec
    arm.set_mode(0)
    arm.set_state(state=0)

    # Test gripper
    arm.set_gripper_position(300)
    time.sleep(1)

    speed = 30  # mm/s
    acc = 60

    # initial position
    init_pos = [206, 0, 110]  # for video
    #arm.set_position(x=init_pos[0], y=init_pos[1], z=init_pos[2], roll=179.9, pitch=0, yaw=0, speed=speed, mvacc=acc, wait=True)
    return arm


# bounds = torch.Tensor([-0.1, -0.1, -0.2, 0.8, 0.8, 0.8])
bounds = torch.Tensor([-0.1, -0.3, -0.2, 0.8, 0.7, 0.7]) # the same as that in the training
vox_size = 100
rotation_resolution = 5
max_num_coords=220000
bs = 1
_num_rotation_classes = 72


# change the following desk2camera, adjust_ori_mat, and adjust_pos_mat each time we do the calibration
desk2camera = [[0.9999181075257777, 0.012558288952198796, 0.002463254079527643, -0.17409898059817605], [-0.0025892709454548683, 0.01002819281479538, 0.9999463640740138, 0.30000481310346677], [0.012532913389880709, -0.9998708540243897, 0.010059888393982353, 0.8866193118943917], [0.0, 0.0, 0.0, 1.0]]
# RECALIBRATE
adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
adjust_pos_mat = np.array([[1, 0, 0, -0.20], [0, 1, 0, 0.01], [0, 0, 1, 0], [0, 0, 0, 1]]) # manually adjust 
base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat

base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
cam2base = np.linalg.inv(base2camera).reshape(4, 4)
print('cam2base:', cam2base)

gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
gl2cv_homo = np.eye(4)
gl2cv_homo[:3, :3] = gl2cv
cam2base = cam2base @ gl2cv_homo

device = "cuda:1"
#description = "open the top oven door"
description = "open the bottom cabinet door" # the same as the one used in training
tokens = clip.tokenize([description]).numpy()
token_tensor = torch.from_numpy(tokens).to(device)
clip_model, preprocess = clip.load("RN50", device=device)
lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
lang_goal_embs = lang_embs[0].float().detach().unsqueeze(0)
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
    num_latents=512,
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
# open the cabinet door
#checkpoint = torch.load('real_ckpt/ckpt_10demo_ar_prev_2_23_new_aug_20000.pth',map_location='cuda:1')
# open the cabinet door traj
#checkpoint = torch.load('real_ckpt/ckpt_4demo_ar_prev_2_26_aug_2408_cabinet_traj_10016000.pth',map_location='cuda:1')
# open the oven door
#checkpoint = torch.load('real_ckpt/ckpt_10demo_ar_prev_aug_2_25_oven_door_3keys_30000.pth',map_location='cuda:1')
# multi
checkpoint = torch.load('real_ckpt/ckpt_4demo_ar_prev_2_26_aug_2408_cabinet_traj_all_12000.pth',map_location='cuda:1')
qnet.load_state_dict(checkpoint)

arm = init_robot()
speed = 30  # mm/s
acc = 60


grip_curr = 1 # the initial gripper state 
for seq in range(20):
    get_from_camera(seq) # get the current pointcloud
    pcd_path = 'pcd_data/pcd_'+str(seq)+'.ply'

    pointcloud_robot, rgb = get_rgb_pcd(pcd_path, cam2base, device)
    voxel_grid = voxelizer.coords_to_bounding_voxel_grid(
        pointcloud_robot, coord_features=rgb, coord_bounds=bounds)
    #print("voxel_grid:", voxel_grid.shape)
    voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().to(device)

    # get the current state of the robot
    cur_pos_xyz = np.array(arm.get_position()[1][:3]) * 0.001
    cur_pos_rot = arm.get_position()[1][3:]
    trans_indices_prev = point_to_voxel_index(cur_pos_xyz, vox_size, bounds)
    rot_indices_prev = ((np.array(cur_pos_rot) + 180) / rotation_resolution).astype(int) -1
    print('cur_indicies:', cur_pos_rot, trans_indices_prev, rot_indices_prev)
    trans_indices_prev = torch.Tensor(trans_indices_prev).unsqueeze(0)
    rot_indices_prev = torch.Tensor(rot_indices_prev).unsqueeze(0)
    proprio = torch.Tensor(np.array([grip_curr])).unsqueeze(0)
    proprio = torch.cat((trans_indices_prev, rot_indices_prev, proprio), 1).to(device)

    with torch.no_grad():
        q_trans, rot_grip_q, collision_q = qnet(voxel_grid, proprio, lang_goal_embs)
    #print(q_trans.shape, rot_and_grip_q.shape, collision_q.shape)

    # choose best action through argmax
    coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = choose_highest_action(q_trans,
                                                                                            rot_grip_q,
                                                                                            collision_q,
                                                                                            rotation_resolution)
    # discrete to continuous translation action
    bounds_new = bounds.unsqueeze(0).to(device)
    res = (bounds_new[:, 3:] - bounds_new[:, :3]) / vox_size
    continuous_trans = bounds_new[:, :3] + res * coords_indicies.int() + res / 2

    q_trans = _softmax_q(q_trans)
    trans = coords_indicies
    continuous_trans = continuous_trans
    rot_and_grip = rot_and_grip_indicies
    collision = ignore_collision_indicies

    # discrete to continuous
    continuous_trans = continuous_trans[0].detach().cpu().numpy()
    rotation_pred = ((rot_and_grip[0][:3] + 1) * rotation_resolution - 180).detach().cpu().numpy()
    gripper_open = bool(rot_and_grip[0][-1].detach().cpu().numpy())
    ignore_collision = bool(collision[0][0].detach().cpu().numpy())

    pos = continuous_trans*1000
    rot = rotation_pred
    #print('pred_indicies:', coords_indicies, rot_and_grip_indicies)
    print(pos, rotation_pred, gripper_open, ignore_collision)

    # make use the prediction is safe for the robot to move, then close the image; otherwise, control + c to stop the code, then close the image
    visualize_pcd(pcd_path, cam2base)

    # if seq == 2:
    #     x_cur = arm.get_position()[1][0]
    #     z_cur = arm.get_position()[1][2]
    #     arm.set_position(x=x_cur, y=pos[1], z=pos[2], roll=rot[0], pitch=rot[1], yaw=rot[2], speed=speed, mvacc=acc, wait=True)
    #     arm.set_position(x=pos[0], y=pos[1], z=pos[2], roll=rot[0], pitch=rot[1], yaw=rot[2], speed=speed, mvacc=acc, wait=True)
    # else:
    arm.set_position(x=pos[0], y=pos[1], z=pos[2], roll=rot[0], pitch=rot[1], yaw=rot[2], speed=speed, mvacc=acc, wait=True)
    if gripper_open is True:
        arm.set_gripper_position(300)
    else:
        arm.set_gripper_position(160)
    grip_curr = int(gripper_open) # save the current gripper state


import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
import warnings
warnings.filterwarnings("ignore") # ignore warnings

import torch
import numpy as np
import imageio
import skimage.metrics
from skimage import measure
import torchvision
import util
from data import get_split_dataset
from render import NeRFEmbedRenderer
from model import make_model
import tqdm
from termcolor import colored
import matplotlib.pyplot as plt
from dotmap import DotMap
from PIL import Image, ImageFilter
# import mesh_utils
import trimesh
#import pytorch3d
import peract_utils
import time
import visdom
import pdb
import transforms3d
import open3d as o3d

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



def extra_args(parser):
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split of data to use train | val | test",
    )

    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="0",
        help="Source view(s) in image, in increasing order. -1 to use random 1 view.",
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for selecting target views of each object",
    )
    parser.add_argument("--coarse", action="store_true", help="Coarse network as fine")

    # parser = mesh_utils.mesh_parser(parser)
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

nviews = list(map(int, args.nviews.split()))

device = util.get_cuda(args.gpu_id[0])
net = make_model(conf["model"]).to(device=device)
net.load_weights(args)

if args.coarse:
    net.mlp_fine = None

dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False,
    teacher_model=conf['model']['teacher_model'],
    #task_list=conf['data']['task_list']
)
data_loader = torch.utils.data.DataLoader(
    dset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False
)



renderer = NeRFEmbedRenderer.from_conf(
    conf["renderer"], eval_batch_size=args.ray_batch_size,
).to(device=device)


render_par = renderer.bind_parallel(net, args.gpu_id).eval()

z_near = dset.z_near
z_far = dset.z_far
print("z_near", z_near, "z_far", z_far)
torch.random.manual_seed(args.seed)

total_psnr = 0.0
total_ssim = 0.0
cnt = 0


source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1

video_num=3
renderer.eval()

desk2camera = [[0.999925694259164, 0.012145615372237635, -0.0010440246926720622, -0.1881796343190582], [0.000716273741748547, 0.0269582443046212, 0.9996363038705324, 0.29186754659095254], [0.012169343131661861, -0.9995627729618796, 0.02694754156693799, 0.8745701514323453], [0.0, 0.0, 0.0, 1.0]]
adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
adjust_pos_mat = np.array([[1, 0, 0, -0.20], [0, 1, 0, -0.03], [0, 0, 1, 0], [0, 0, 0, 1]]) 
adjust_pos_mat_ = np.array([[1, 0, 0, -0.40], [0, 1, 0, 0.96], [0, 0, 1, -0.22], [0, 0, 0, 1]]) 

base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
cam2base = np.linalg.inv(base2camera).reshape(4, 4)
print('cam2base:', cam2base)

base2camera_ = desk2camera@adjust_ori_mat@adjust_pos_mat_
cam2base_ = np.linalg.inv(base2camera_).reshape(4, 4)
print('cam2base_:', cam2base_)

gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
gl2cv_homo = np.eye(4)
gl2cv_homo[:3, :3] = gl2cv
cam2base = cam2base @ gl2cv_homo  # gl2cv_homo means points are in gl!
cam2base_ = cam2base_ @ gl2cv_homo


#cam2base = np.concatenate([cam2base[:,0:1], -cam2base[:,2:3], cam2base[:,1:2], cam2base[:,3:]], 1)

# camtoworld = poses[0].cpu().numpy()
# camtoworld_inv = np.linalg.inv(camtoworld)
# worldtobase = cam2base @ camtoworld_inv
# gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
# gl2cv_homo = np.eye(4)
# gl2cv_homo[:3, :3] = gl2cv
# worldtobase = worldtobase @ gl2cv_homo

with torch.no_grad():
    for data_id, data in enumerate(data_loader):
        all_embeds = []
        all_imgs = []
        all_imgs_pred = []
        all_raw_embeds = []

        batch_idx = 0
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape

        cam_rays = util.gen_rays(
            poses, W, H, focal, z_near, z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        vis = visdom.Visdom()
        for view_dest in tqdm.tqdm(range(NV)):
            gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
            with torch.no_grad():
                test_pose = poses[view_dest]
                test_rays = cam_rays[view_dest]  # (H, W, 8)
                test_images = images[views_src]  # (NS, 3, H, W)
                net.encode(
                    test_images.unsqueeze(0),
                    poses[views_src].unsqueeze(0),
                    focal.to(device=device),
                    c=c.to(device=device) if c is not None else None,
                )
                test_rays = test_rays.reshape(1, H * W, -1)
                start_time = time.time()
                pnts, rgbs, sigmas, embeds = render_par(test_rays, want_weights=True, extract_radience=True)
                print("rgbs generated with time: ", time.time()-start_time)
                #pdb.set_trace()
                
                scale = 2.3
                _coord_trans = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
                ).numpy()
                _coord_trans_1 = torch.tensor(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
                ).numpy()
                camtoworld = scale * poses[0].cpu().numpy()
                camtoworld_inv = np.linalg.inv(camtoworld)
                # pdb.set_trace()
                # camtoworld_inv = np.concatenate([camtoworld_inv[:,0:1], -camtoworld_inv[:,2:3], camtoworld_inv[:,1:2], camtoworld_inv[:,3:]], 1)
                # #camtoworld_inv = camtoworld_inv @ _coord_trans
                # worldtobase = (cam2base @ camtoworld_inv) @ _coord_trans 
                # pdb.set_trace()
                worldtobase = cam2base_ @ camtoworld_inv
                # gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
                # gl2cv_homo = np.eye(4)
                # gl2cv_homo[:3, :3] = gl2cv
                # worldtobase = worldtobase @ gl2cv_homo
                worldtobase = torch.from_numpy(worldtobase.astype(np.float32)).cuda()
                #worldtobase = torch.concatenate([worldtobase[:,0:1], -worldtobase[:,2:3], worldtobase[:,1:2], worldtobase[:,3:]], 1)
                pnts = pnts @ worldtobase[:3, :3].T + worldtobase[:3, 3]

                cloud = o3d.io.read_point_cloud("/data/geyan21/projects/nerf-ssl-master/pcd0.ply")
                points = np.asarray(cloud.points)
                colors = np.asarray(cloud.colors) * 255
                valid_bool = np.linalg.norm(points, axis=1) < 2.5
                points = points[valid_bool]
                colors = colors[valid_bool].mean(-1)
                transformed_points = points @ cam2base[:3, :3].T + cam2base[:3, 3]
                bounds = [0, -0.5, 0.6, 0.8, 0.5, 1.6]
                bound_x = [-0.3, 0.7]
                bound_y = [-0.5, 0.5]
                bound_z = [0.6, 1.6]
                mask = (pnts[...,0] > bound_x[0]) & (pnts[...,0] < bound_x[1]) & \
                        (pnts[...,1] > bound_y[0]) & (pnts[...,1] < bound_y[1]) & \
                        (pnts[...,2] > bound_z[0]) & (pnts[...,2] < bound_z[1])
                #transformed_points = points @ (adjust_ori_mat.T@np.array(desk2camera).T@gl2cv_homo)[:3, :3].T
                vis.scatter(X=transformed_points, Y=colors.astype(np.uint8)+1, win='points', \
                            opts=dict(markersize=2, title=f'nerf_pnts_1'))
                #pdb.set_trace()

                # cam2base = torch.from_numpy(cam2base.astype(np.float32)).cuda()
                # pnts = pnts @ cam2base[:3, :3].T + cam2base[:3, 3]
                


                #vis = visdom.Visdom()
                # convert rgb to int for visdom
                # get mask

                # get top 10000 points with sigma

                
                # mask2 = rgbs.sum(-1) > rgbs.sum(-1).mean() # remove black background
                # mask = mask1 & mask2

                # mask by bounds

                bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
                bound_x = [-0.3, 0.7]
                bound_y = [-0.5, 0.5]
                bound_z = [0.6, 1.6]
                mask = (pnts[...,0] > bound_x[0]) & (pnts[...,0] < bound_x[1]) & \
                        (pnts[...,1] > bound_y[0]) & (pnts[...,1] < bound_y[1]) & \
                        (pnts[...,2] > bound_z[0]) & (pnts[...,2] < bound_z[1])

                # pnts = pnts[mask]
                # rgbs = rgbs[mask]
                # sigmas = sigmas[mask]

                
                            
                # masking
                threhold = 200
                mask = sigmas > (sigmas.max()/8)
                pnts = pnts[mask]
                rgbs = rgbs[mask]


                # convert to label
                rgbs = rgbs.cpu().numpy()
                rgbs = (rgbs * 255).astype(np.uint8)
                rgbs = rgbs.mean(axis=-1)
                
                # use points and rgbs to get volume
                vis.scatter(X=pnts.cpu().numpy(), Y=rgbs.astype(np.uint8)+1, win='pnts', \
                            opts=dict(markersize=2, title=f'nerf_pnts_{threhold}'))
                
                pdb.set_trace()
                
                
                # Create sample tiles

                # create a grid of points
                # x = torch.linspace(-0.3103, 3.0554, N, device=device)
                # y = torch.linspace(-2.8981, 0.3607, N, device=device)
                # z = torch.linspace(-1.8048, 1.3243, N, device=device)
                # x = torch.linspace(-2.4, 2.4, N, device=device)
                # y = torch.linspace(-2.4, 2.4, N, device=device)
                # z = torch.linspace(-2.4, 2.4, N, device=device)


                # tiles = [x, y, z]

                # # tiles = [torch.linspace(-args.limit, args.limit, num) for num in nums]

                # # Generate 3D samples
                # samples = torch.stack(torch.meshgrid(*tiles), -1).float()
                # occupancy = torch.ones_like(samples[..., 0]).unsqueeze(-1).to(device=device)
                # samples = torch.concat([samples, occupancy], -1)
                # samples = samples.permute(3, 0, 1, 2)
                # samples = samples.cpu().numpy()

                
                # peract_utils.visualise_voxel(samples, show=False)

                # radiance = mesh_utils.extract_radiance_voxel(nerf_model, samples, args, device=device, nums=N)

                # # visualize voxel
                # sigma = np.maximum(radiance[...,3], 0)

                # plt.hist(np.maximum(0,sigma.ravel()), log=True)
                # plt.savefig('../data/sigma.png')
                # threshold = 50 # 50
                # print('fraction occupied', np.mean(sigma > threshold))
                # vertices, triangles = mcubes.marching_cubes(sigma, threshold)
                # print('done', vertices.shape, triangles.shape)

                # mesh = trimesh.Trimesh(vertices / N - .5, triangles)
                # # trimesh.exchange.obj.export_obj(mesh, '../data/mesh.obj')
                # mesh.export('../data/mesh.obj')
        exit()




        if not os.path.exists(f"eval_videos/{args.name}/"):
            os.makedirs(f"eval_videos/{args.name}/")
        # save img array as video
        all_embeds[0].save(f'eval_videos/{args.name}/{data_id}_recon_embeds.gif', save_all=True, append_images=all_embeds[1:], optimize=False, duration=100, loop=0)
        all_imgs_pred[0].save(f'eval_videos/{args.name}/{data_id}_recon_imgs_pred.gif', save_all=True, append_images=all_imgs_pred[1:], optimize=False, duration=100, loop=0)
        save_single_channel_video(all_raw_embeds, f'eval_videos/{args.name}/{data_id}_recon_raw_embeds.mp4', buffer=f'{args.name}')
        
        # save video
        if data_id == video_num:
            break

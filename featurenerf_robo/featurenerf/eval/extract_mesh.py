
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
import mesh_utils

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

    parser = mesh_utils.mesh_parser(parser)
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
    task_list=conf['data']['task_list']
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

        for view_dest in tqdm.tqdm(range(NV)):
            gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
            with torch.no_grad():
                test_rays = cam_rays[view_dest]  # (H, W, 8)
                test_images = images[views_src]  # (NS, 3, H, W)
                net.encode(
                    test_images.unsqueeze(0),
                    poses[views_src].unsqueeze(0),
                    focal.to(device=device),
                    c=c.to(device=device) if c is not None else None,
                )
                test_rays = test_rays.reshape(1, H * W, -1)

                # pnts, rgbs, sigmas, embeds = render_par(test_rays, want_weights=True, extract_radience=True)

                ### get mesh ###
                """
                args.limit = 1.2
                args.res = 240
                nerf_model = render_par.net
                radiance = mesh_utils.extract_radiance(nerf_model, limit=args.limit, batch_size=4096, nums=args.res)
                density = radiance[..., 3]

                iso_value = mesh_utils.extract_iso_level(density)
                results = measure.marching_cubes(density, iso_value)
                # Use contiguous tensors
                vertices, triangles, normals, _ = [torch.from_numpy(np.ascontiguousarray(result)) for result in results]

                # Use contiguous tensors
                normals = torch.from_numpy(np.ascontiguousarray(normals))
                vertices = torch.from_numpy(np.ascontiguousarray(vertices))
                triangles = torch.from_numpy(np.ascontiguousarray(triangles))

                # Normalize vertices, to the (-limit, limit)
                vertices = args.limit * (vertices / (args.res / 2.) - 1.)
                ### finish generating mesh ###

                print('normal shape', normals.shape)
                print('vertices shape', vertices.shape)
                print('triangles shape', triangles.shape)
                print('density shape', density.shape)
                print('finish visualization')
                """
                nerf_model = render_par.net
                mesh_utils.export_marching_cubes(nerf_model, render_par, args)
                ### do some visualization ###
                
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

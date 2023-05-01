
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
                render_dict = DotMap(render_par(test_rays, want_weights=True))
                coarse = render_dict.coarse
                fine = render_dict.fine

                using_fine = len(fine) > 0


                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)
                embed_fine_np = fine.embed[0].cpu().numpy().reshape(H, W, -1).mean(-1)
                
                depth_fine_cmap = util.cmap(depth_fine_np) 
                alpha_fine_cmap = util.cmap(alpha_fine_np) 
                embed_fine_cmap = util.cmap(embed_fine_np)
                rgb_fine_np = (rgb_fine_np * 255).astype(np.uint8)

                rgb_fine_np = Image.fromarray(rgb_fine_np)
                embed_fine_cmap = Image.fromarray(embed_fine_cmap)

                # # apply Gaussian blur
                # rgb_fine_np = rgb_fine_np.filter(ImageFilter.GaussianBlur(radius=1))
                # embed_fine_cmap = embed_fine_cmap.filter(ImageFilter.GaussianBlur(radius=1))

                all_embeds.append(embed_fine_cmap)
                all_imgs_pred.append(rgb_fine_np)
                all_raw_embeds.append(fine.embed[0].cpu().reshape(H, W, -1).mean(-1))



        if not os.path.exists(f"eval_videos/{args.name}/"):
            os.makedirs(f"eval_videos/{args.name}/")
        # save img array as video
        all_embeds[0].save(f'eval_videos/{args.name}/{data_id}_recon_embeds.gif', save_all=True, append_images=all_embeds[1:], optimize=False, duration=100, loop=0)
        all_imgs_pred[0].save(f'eval_videos/{args.name}/{data_id}_recon_imgs_pred.gif', save_all=True, append_images=all_imgs_pred[1:], optimize=False, duration=100, loop=0)
        # save_single_channel_video(all_raw_embeds, f'eval_videos/{args.name}/{data_id}_recon_raw_embeds.mp4', buffer=f'{args.name}')
        
        # save video
        if data_id == video_num:
            break

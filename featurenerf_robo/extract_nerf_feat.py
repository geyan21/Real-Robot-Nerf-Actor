
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "./featurenerf/src"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "./featurenerf/"))
)
import warnings
warnings.filterwarnings("ignore") # ignore warnings
import pdb
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
# import peract_utils
import time
import visdom
import pdb
import transforms3d
from pyhocon import ConfigFactory
import open3d as o3d

def PSNR_np(img1, img2, max_val=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

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


def save_single_channel_img(img,name):
    """
    save a list of single channel images as a video
    """
    if not os.path.exists(".visualize_recon/"):
        os.makedirs(".visualize_recon/")

    # make img with matplotlib
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    plt.imshow(img, cmap="rainbow")
    plt.savefig(".visualize_recon/{}.png".format(name))
    

conf_path = "/data/geyan21/projects/real-robot-nerf-actor/featurenerf_robo/featurenerf/conf/exp/robo_dino_real.conf"
conf = ConfigFactory.parse_file(conf_path)

device = "cuda:3"
id = 4
# test_image = "/data/geyan21/projects/real-robot-nerf-actor/data/Nerfact_data/kitchen1/teapot/real0/rgb0.png"   
test_image = f"/data/geyan21/projects/real-robot-nerf-actor/data/Nerfact_kitchen/oven/real0/rgb{id}.png"
# model_path = "/data/geyan21/projects/featurenerf-robo/featurenerf/checkpoints/robo_dino_real_Nerf_ContrastMV_14imgs_1024/pixel_nerf_latest_00274000"
# model_path = "/data/geyan21/projects/featurenerf-robo/featurenerf/checkpoints/robo_dino_real_Nerf_ContrastMV_14imgs_MV_1024/pixel_nerf_latest_00036000"
# model_path = "/data/geyan21/projects/featurenerf-robo/featurenerf/checkpoints/robo_dino_real_Nerf_ContrastMV_14imgs_MV_512/pixel_nerf_latest_00160000"
# model_path = "/data/geyan21/projects/featurenerf-robo/featurenerf/checkpoints/robo_dino_real_Nerf_ContrastMV_14imgs_MV_512/pixel_nerf_latest_00396000"
model_path = "/data/geyan21/projects/featurenerf-robo/featurenerf/checkpoints/robo_dino_real_Nerf_ContrastMV_three_Kitchens_three_tasks_512/backup6/pixel_nerf_latest_00084000"
nerf_npz = '/data/geyan21/projects/featurenerf-robo/Data/NerfContrastMV/kitchen1/oven/step0/Nerfreal_8.npz'
# model_path = "/data/geyan21/projects/featurenerf-robo/featurenerf/checkpoints/robo_dino_real_Nerf_ContrastMV_sep_14imgs/pixel_nerf_latest_00008000"
# nerf_npz = f'/data/geyan21/projects/featurenerf-robo/Data/Nerf_ContrastMV/Multi_step_img/Nerfreal_8_{id}.npz'
net = make_model(conf["model"]).to(device=device)
# pdb.set_trace()
net.load_weights(model_path=model_path,device=device)

renderer = NeRFEmbedRenderer.from_conf(
    conf["renderer"], eval_batch_size=50000,
).to(device=device)
render_par = renderer.bind_parallel(net).eval()
renderer.eval()
# model_path = '/data/geyan21/projects/real-robot-nerf-actor/nerfmodel/rl03/pixel_nerf_latest'
# nerf_npz = '/data/geyan21/projects/real-robot-nerf-actor/nerfmodel/rl03/Nerfreal_5_0.npz'
def extract_nerf_feat(conf_path, device, model_path, test_image, nerf_npz):

    conf = ConfigFactory.parse_file(conf_path)

    # pose_path = "/data/geyan21/projects/featurenerf-robo/featurenerf/data/robo_dino_real/test/1/Oven/000000_pose.txt"
    # focal = 80.5
    # pdb.set_trace()

    # load one image and resize to 128x128 if needed
    img = Image.open(test_image)
    # img = img.resize((128, 128))
    img = img.resize((80, 60))
    img = np.array(img) / 255.0
    # pdb.set_trace()
    # normalize to -1, 1
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device=device)
    # pdb.set_trace()
    # load pose
    # pose = np.loadtxt(pose_path)
    nerf_data = np.load(nerf_npz)
    pose = nerf_data['poses'][0]
    pose = torch.from_numpy(pose).unsqueeze(0).float().to(device=device)

    # load focal
    focal = torch.from_numpy(nerf_data['focal']).float().to(device=device)


    
    use_visdom = True
    if use_visdom:
        vis = visdom.Visdom()

    z_near = 1.2
    z_far = 4.0
    print("z_near", z_near, "z_far", z_far)
    torch.random.manual_seed(0)

    scale = 2.5

    desk2camera = [[0.9990809680298979, -0.04226655945655472, 0.007124413810835721, -0.21317943800854028], [-0.004920145095573605, 0.052028245371015504, 0.9986334932575875, 0.24697715742900836], [-0.04257947266795357, -0.997750770300539, 0.05177247214495428, 0.7521121095357662], [0.0, 0.0, 0.0, 1.0]]
    adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    adjust_pos_mat = np.array([[1, 0, 0, -0.08], [0, 1, 0, 0.16], [0, 0, 1, 0.01], [0, 0, 0, 1]])
    adjust_pos_mat_ = np.array([[1, 0, 0, -0.65], [0, 1, 0, 1.85], [0, 0, 1, -0.7], [0, 0, 0, 1]]) 

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

    camtoworld = scale * pose[0].cpu().numpy()
    camtoworld_inv = np.linalg.inv(camtoworld)
    worldtobase = cam2base_ @ camtoworld_inv
    worldtobase = torch.from_numpy(worldtobase.astype(np.float32)).to(device=device)

    with torch.no_grad():           # this mattters! otherwise the memory will explode
        NV, _, H, W = img.shape

        cam_rays = util.gen_rays(
            pose, W, H, focal, z_near, z_far, c=None
        )  # (NV, H, W, 8)
        images_0to1 = img * 0.5 + 0.5  # (NV, 3, H, W)


        gt = images_0to1[0].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        test_pose = pose
        test_rays = cam_rays[0]  # (H, W, 8)
        test_image = img  # (NS, 3, H, W)
        net.encode(
            test_image.unsqueeze(0),
            test_pose.unsqueeze(0),
            focal.to(device=device),
            c=None,
        )
        test_rays = test_rays.reshape(1, H * W, -1)
        # pdb.set_trace()
        chunk_size = 4096  #4096
        pnts = []
        rgbs = []
        sigmas = []
        embeds = []
        start_time = time.time()

        Visualize_reconstruction = True
        if Visualize_reconstruction:
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0
            alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
            depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
            rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)
            embed_fine_np = fine.embed[0].cpu().numpy().reshape(H, W, -1).mean(-1) #H, W
            # pdb.set_trace()

            # free memory
            del render_dict
            del coarse
            del fine

            psnr = PSNR_np(rgb_fine_np, gt)
            print("psnr: ", psnr)
            
            depth_fine_cmap = util.cmap(depth_fine_np) 
            alpha_fine_cmap = util.cmap(alpha_fine_np) 
            embed_fine_cmap = util.cmap(embed_fine_np)
            rgb_fine_np = (rgb_fine_np * 255).astype(np.uint8)

            rgb_fine_np = Image.fromarray(rgb_fine_np)
            embed_fine_cmap = Image.fromarray(embed_fine_cmap)


            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 4)
            axs[0].imshow(images_0to1.cpu().numpy().transpose(0, 2, 3, 1)[0])
            axs[0].title.set_text('tgt')
            axs[1].imshow(rgb_fine_np)
            axs[1].title.set_text('psnr={:.2f}'.format(psnr))
            axs[2].imshow(alpha_fine_cmap)
            axs[2].title.set_text('alpha')
            axs[3].imshow(depth_fine_cmap)
            axs[3].title.set_text('depth')
            plt.tight_layout()
            plt.savefig(f".visualize_recon/all_step{id}.png")

            # save as image using Image
            if not os.path.exists(".visualize_recon/"):
                os.makedirs(".visualize_recon/")
            Image.fromarray(depth_fine_cmap).save(f".visualize_recon/depth_fine_cmap_step{id}.png")
            Image.fromarray(alpha_fine_cmap).save(f".visualize_recon/alpha_fine_cmap_step{id}.png")
            rgb_fine_np.save(f".visualize_recon/rgb_fine_np_step{id}.png")
            embed_fine_cmap.save(f".visualize_recon/embed_fine_cmap_step{id}.png")

            # pdb.set_trace()

        with torch.no_grad():
            for i in range(0, test_rays.shape[1], chunk_size):
                # pdb.set_trace()
                ret_last_feat = True
                chunk = test_rays[:, i : i + chunk_size]
                pnts_, rgbs_, sigmas_, embeds_ = render_par(chunk, want_weights=True, extract_radience=True, ret_last_feat=ret_last_feat)
                # reshape embeds
                if ret_last_feat:
                    embeds_ = embeds_.reshape(1, -1, embeds_.shape[-1])  # 512 if ret_last_feat else 384+3
                pnts.append(pnts_)
                rgbs.append(rgbs_)
                sigmas.append(sigmas_)
                embeds.append(embeds_)
                # free memory that is not used
                del pnts_, rgbs_, sigmas_, embeds_

        pnts = torch.cat(pnts, dim=1)
        rgbs = torch.cat(rgbs, dim=1)
        sigmas = torch.cat(sigmas, dim=1)
        embeds = torch.cat(embeds, dim=1)
        # pdb.set_trace()
        # pnts, rgbs, sigmas, embeds = render_par(test_rays, want_weights=True, extract_radience=True)
        print("rgbs generated with time: ", time.time()-start_time)

        # # masking
        # mask_init = sigmas > (sigmas.max()/8)
        # # number of inintial points after masking using 1/8 of the max
        # num_init = mask_init.sum()
        # print("num_init: ", num_init)
        # if num_init > 180000:
        #     mask1 = sigmas > (sigmas.max()/7)   # 1/7 of the max
        #     # print use 1/7 of the max
        #     print("use 1/7 of the max")
        # elif num_init > 120000:
        #     mask1 = sigmas > (sigmas.max()/8)  # 1/8 of the max
        #     # print use 1/8 of the max
        #     print("use 1/8 of the max")
        # else:
        #     mask1 = sigmas > (sigmas.max()/9)
        
        mask2 = rgbs.sum(-1) > rgbs.sum(-1).mean() 
        # write a loop to make sure the points after masked with mask1 are between 90000 and 150000
        step = 0.1
        num_current = 0
        lower_bound = 50000
        upper_bound = 70000
        while num_current < lower_bound or num_current > upper_bound:
            mask1 = sigmas > (sigmas.max()*step)
            num_current = (mask1&mask2).sum()
            # print step
            print("step: ", step)
            print("num_current: ", num_current)
            if num_current < lower_bound:
                step -= 0.01
            elif num_current > upper_bound:
                step += 0.02
            else:
                break


        # get furthest points coordinates
        pnts_wo_black = pnts[mask2]
        # z_furthest = pnts_wo_black[...,2].min()  # get the furthest z in opengl coordinate system, the z is out of the screen
        # get top 1000 min points in z and get the average z
        z_furthest = pnts_wo_black[...,2].topk(1000, largest=False).values.mean()
        mask2_ = pnts[...,2] > z_furthest
        # remove black ground far in the back
        # mask2 = (pnts[...,2] > 0.6) & (pnts[...,2] < 1.6)
        mask = mask1 & mask2
       
        # mask = (sigmas > (sigmas.max()/9)) & mask2
        # mask = (sigmas > (sigmas.max()/8))
        # mask = mask1

        bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]   # mask by bounds
        bound_x = [-0.3, 0.7]
        bound_y = [-0.5, 0.5]
        bound_z = [0.6, 1.6]
        mask3 = (pnts[...,0] > bound_x[0]) & (pnts[...,0] < bound_x[1]) & \
                (pnts[...,1] > bound_y[0]) & (pnts[...,1] < bound_y[1]) & \
                (pnts[...,2] > bound_z[0]) & (pnts[...,2] < bound_z[1])
        # print how many points are selected and how many points are in total
        print("selected points: ", mask.sum().item(), "total points: ", mask.shape[1])
        pnts = pnts[mask]
        rgbs = rgbs[mask]
        embeds = embeds[mask]
        # convert to label
        rgbs = rgbs.cpu().numpy()  # 0-1
        # pdb.set_trace()
        rgbs_vis = (rgbs * 255).astype(np.uint8)
        # rgbs = rgbs.mean(axis=-1)
        # pdb.set_trace()
        # pnts = pnts[:200000,:]
        # rgbs = rgbs[:200000,:]
        # use points and rgbs to get volume
        pnts = pnts @ worldtobase[:3, :3].T + worldtobase[:3, 3]

        threhold = None
        if use_visdom:
            print("visualizing...")
            vis.scatter(X=pnts.cpu().numpy(), win='pnts', \
                                        opts=dict(markersize=2, markercolor=rgbs_vis, title=f'nerf_pnts_step{id}'))

        # save as ply
        save_ply = True
        if save_ply:
            # pdb.set_trace()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pnts.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(rgbs)
            o3d.io.write_point_cloud("nerf.ply", pcd)
        
        # free memory that is not used
        
        return pnts, rgbs, embeds


# test the nerf feature extraction in main
if __name__ == "__main__":
    nerfdata = extract_nerf_feat(conf_path, device, model_path, test_image, nerf_npz)
    # pdb.set_trace()
    print("done")
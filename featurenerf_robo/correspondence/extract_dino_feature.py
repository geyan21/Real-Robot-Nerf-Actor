import torchvision
from sklearn.decomposition import PCA
import os
import imageio
import torch
import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np
from dino import DINO
from natsort import natsorted

warnings.filterwarnings("ignore") # ignore warnings

def save_feature_map(feature_map, path="debug.png"):
    """
    torch.tensor with C, H, W
    """
    feature_map = feature_map.permute(1, 2, 0) # to H, W, C
    feature_map = feature_map.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map.mean(-1), cmap="rainbow")
    plt.savefig(path)

model_name = 'dino'
dino = DINO().eval().cuda()
embed_dim = 384

# task_list = ["shelfplacing", "stacking", "lift", "peginsert", "reachwall", "reach", "pegbox", "push"]
task_list = ["close_jar", "open_drawer", "push_buttons"]
# task_list = ['stack_blocks']
data_root = "/data/yanjieze/projects/nerf-act/data/nerf_data"

for task in task_list:
    data_path = os.path.join(data_root, task, 'all_variations', 'episodes')

    print("data_path: ", data_path)
    episode_dirs = [os.path.join(data_path, x) for x in os.listdir(data_path)]
    scene_paths = []
    for episode_dir in episode_dirs:
        scene_paths += [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]

    scene_count = 0 # count the number of scenes
    for scene_path in tqdm.tqdm(scene_paths):
        scene_count += 1
        img_base_path = os.path.join(scene_path, "images")
        
        # create feature directory
        # feature_base_path = os.path.join(scene_path, "features")
        # if not os.path.exists(feature_base_path):
        #     os.makedirs(feature_base_path)
        all_img = []
        all_img_name = os.listdir(img_base_path)
        all_img_name = natsorted(all_img_name) # very important to sort!!!
        for img_name in all_img_name:
            img_path = os.path.join(img_base_path, img_name)
            # img = torchvision.io.read_image(img_path).div(255).float()
            img = imageio.imread(img_path).astype(np.float32) / 255.
            img = torch.from_numpy(img).float()
            # [128,128,3] -> [3,128,128]
            img = img.permute(2, 0, 1)
            img = img.to('cuda:0')
            all_img.append(img)
        
        all_img = torch.stack(all_img) # [N, C, H, W] N is the number of images in the scene
        # normalize

        features = dino(all_img)
        
        if embed_dim < 384:
            N, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C)
            features = features.cpu().detach().numpy()
            # print("origin dino feature shape:", features.shape)
            pca = PCA(n_components=embed_dim)
            X = pca.fit_transform(features)
            X = torch.Tensor(X).view(N, H, W, embed_dim).permute(0, 3, 1, 2) # [N, 64, H, W]
        else:
            X = features

        
        ###
        # Not interpolate

        ###
        # interpolate to image size
        # N, C, H, W = all_img.shape
        # X = torch.nn.functional.interpolate(X, size=(H, W), mode='bilinear', align_corners=False) # (N, C, H, W)

        # plt.figure()
        # plt.imshow(X[0].mean(0), cmap="rainbow") # [H, W, C] -> [C, H, W] for matplotlib
        # plt.savefig('dino_feature.png') # save feature map

        # store feature
        X = X.cpu().detach().numpy()
        feature_path = os.path.join(scene_path, "features.npz") # feature_path: scene1/features/000000.pt
        np.savez(feature_path, X) # save feature

        
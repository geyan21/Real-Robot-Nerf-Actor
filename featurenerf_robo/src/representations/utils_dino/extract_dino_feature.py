import torchvision
from sklearn.decomposition import PCA
import os

import torch
import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np
from dino import DINO

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

data_path = "../nerf_data/robot_lift_128"
# data_path = "../nerf_data/robot_peginsert_128"
data_path = "../nerf_data/robot_reachwall_128"
# data_path = "../nerf_data/robot_shelfplacing_128"
data_path = "../nerf_data/robot_stacking_128"

print("data_path: ", data_path)
scene_paths = [os.path.join(data_path, x) for x in os.listdir(data_path)] # [scene1, scene2, ...]

scene_count = 0 # count the number of scenes
for scene_path in tqdm.tqdm(scene_paths):
    scene_count += 1
    img_base_path = os.path.join(scene_path, "images")
    # create feature directory
    # feature_base_path = os.path.join(scene_path, "features")
    # if not os.path.exists(feature_base_path):
    #     os.makedirs(feature_base_path)
    all_img = []
    for img_name in os.listdir(img_base_path):
        img_path = os.path.join(img_base_path, img_name)
        img = torchvision.io.read_image(img_path).div(255).float()
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

    
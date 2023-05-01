import torchvision
from sklearn.decomposition import PCA
import os

import torch
import tqdm
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from clip import build_model, load_clip
warnings.filterwarnings("ignore") # ignore warnings
from natsort import natsorted

def save_feature_map(feature_map, path="debug.png"):
    """
    torch.tensor with C, H, W
    """
    feature_map = feature_map.permute(1, 2, 0) # to H, W, C
    feature_map = feature_map.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map.mean(-1), cmap="rainbow")
    plt.savefig(path)

model_name = 'clip'

model, transform = load_clip('RN50', jit=False)
clip_model = build_model(model.state_dict())
clip_model = clip_model.float().cuda()
clip_model.eval()
del model

embed_dim = 384

task_list = ["close_jar", "open_drawer", "push_buttons"]
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

        all_img = []
        all_img_name = os.listdir(img_base_path)
        all_img_name = natsorted(all_img_name) # very important to sort!!!
        
        for img_name in all_img_name:
            img_path = os.path.join(img_base_path, img_name)
            #print(img_path)
            img = Image.open(img_path)
            img = transform(img)
            all_img.append(img)
            #print(img.shape, img.max(), img.min())

        all_img = torch.stack(all_img) # [N, C, H, W] N is the number of images in the scene
        all_img = all_img.cuda()
        #print(all_img.shape)
        # normalize

        with torch.no_grad():
            features = clip_model.encode_image(all_img)

        #
        # if features.shape[1] != embed_dim:
        #     N, C, H, W = features.shape
        #     features = features.permute(0, 2, 3, 1).reshape(-1, C)
        #     features = features.cpu().detach().numpy()
        #     # print("origin dino feature shape:", features.shape)
        #     pca = PCA(n_components=embed_dim)
        #     X = pca.fit_transform(features)
        #     X = torch.Tensor(X).view(N, H, W, embed_dim).permute(0, 3, 1, 2) # [N, emb_dim, H, W]
        #     print(X.shape)
        X = features

        X = X.cpu().detach().numpy()
        feature_path = os.path.join(scene_path, "features_clip_2048.npz")
        np.savez(feature_path, X) # save feature

        
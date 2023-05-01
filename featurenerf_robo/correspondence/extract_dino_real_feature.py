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
import pdb

# set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
pretrain_path = "/data/geyan21/projects/featurenerf-robo/correspondence/dino_deitsmall8_pretrain.pth"
dino = DINO(pretrain_path).eval().cuda()
embed_dim = 384


data_path = "/data/geyan21/projects/featurenerf-robo/Data/Nerf_kitchen_1_real_poses/img/nerfkitchen1_real_poses.npz"
print("data_path: ", data_path)
feature_base_path = "/data/geyan21/projects/featurenerf-robo/Data/Nerf_kitchen_1_real_poses"

# all_img = []
# all_img_name = os.listdir(img_base_path)
# all_img_name = natsorted(all_img_name) # very important to sort!!!
# for img_name in all_img_name:
#     img_path = os.path.join(img_base_path, img_name)
#     # img = torchvision.io.read_image(img_path).div(255).float()
#     img = imageio.imread(img_path).astype(np.float32) / 255.
#     img = torch.from_numpy(img).float()
#     # [128,128,3] -> [3,128,128]
#     img = img.permute(2, 0, 1)
#     img = img.to('cuda:0')
#     all_img.append(img)

realdata = np.load(data_path)
imgs = torch.from_numpy(np.transpose(realdata['images'],(0, 3, 1, 2)))#float32
all_img = []
for im in imgs:
    im = im.cuda()
    all_img.append(im)

all_img = torch.stack(all_img) # [N, C, H, W] N is the number of images in the scene
# normalize

features, cls_attn = dino(all_img)



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
features = X.cpu().detach().numpy()
cls_attn = cls_attn.cpu().detach().numpy()
feature_path = os.path.join(feature_base_path, "features_real.npz") # feature_path: scene1/features/000000.pt
np.savez(feature_path, features=features, cls_attn=cls_attn) # save feature

    
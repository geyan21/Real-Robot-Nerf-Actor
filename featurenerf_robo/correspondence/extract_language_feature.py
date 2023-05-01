import torchvision
from sklearn.decomposition import PCA
import os

import torch
import tqdm
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from clip import build_model, load_clip, tokenize
import clip
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

model_name = 'clip'

model, transform = load_clip('RN50', jit=False)
clip_model = build_model(model.state_dict())
clip_model = clip_model.float().cuda()
clip_model.eval()
del model

embed_dim = 384

task_list = ["close_jar", "open_drawer", "push_buttons"]
# task_list = ["stack_blocks"]
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
        description_path = os.path.join(scene_path, "description.txt")

        with open(description_path, 'r') as f:
            description = f.read()
        description = description.strip()
        description = description.split('\n')
        # onyl select one description
        description = description[0]

        with torch.no_grad():
            tokens = tokenize([description]).numpy()
            tokens = torch.from_numpy(tokens).cuda()
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(tokens)

            sentence_emb = sentence_emb.cpu().detach().numpy()
            token_embs = token_embs.cpu().detach().numpy()


        feat = dict()
        feat["sentence_emb"] = sentence_emb
        feat["token_emb"] =  token_embs

        feature_path = os.path.join(scene_path, "description_feature.npz")
        np.savez(feature_path, feat) # save feature

        
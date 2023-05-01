import torchvision
import matplotlib.pyplot as plt
import torch

def save_feature_map(feature_map, path="debug.png"):
    """
    torch.tensor with C, H, W
    """
    feature_map = feature_map.permute(1, 2, 0) # to H, W, C
    feature_map = feature_map.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map.mean(-1), cmap="rainbow")
    plt.savefig(path)

def save_rgb_image(image, path="debug.png"):
    """
    torch.tensor with C, H, W
    """
    image = image.permute(1, 2, 0) # to H, W, C
    image = image.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.savefig(path)
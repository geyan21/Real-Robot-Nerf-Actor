import os
import mvp
import torchvision.transforms as T
import torch.nn as nn
import torch

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"File {local_path} already exists, skipping download")
        return local_path

    print(f"Downloading {url} to {local_path}")
    os.system(f"wget {url} -O {local_path}")

    print(f"Downloaded {url} to {local_path}")
    return local_path

class MVPEncoder(nn.Module):
    """
    https://arxiv.org/abs/2203.06173
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = mvp.load("vitb-mae-egosoup") # strongest model
        self.model.freeze() 
        self.model = self.model.cuda()
        self.transform = nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
    
    def forward(self, x):
        return self.model(self.transform(x))
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision

import vision_transformer_flexible as vits
import os
import pdb

def download_file(url, local_path):
    import requests
    import os
    import shutil
    import sys
    import time

    if os.path.exists(local_path):
        print(f"File {local_path} already exists, skipping download")
        return local_path

    print(f"Downloading {url} to {local_path}")
    os.system(f"wget {url} -O {local_path}")

    print(f"Downloaded {url} to {local_path}")
    return local_path

class DINO(nn.Module):
    def __init__(self, pretrain_path=None):
        super().__init__()
        self.patch_size = 8
        self.feat_layer = 9
        self.high_res = False

        if self.patch_size == 16:
            self.model_name = "vit_base"
            self.stride = 8
            self.num_patches = 16
            self.padding = 5
            self.pretrain_path = download_file(
                "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
                "dino_vitbase16_pretrain.pth",
            ) if pretrain_path is None else pretrain_path
        elif self.patch_size == 8:
            self.model_name = "vit_small"
            self.stride = 4
            self.num_patches = 32
            self.padding = 2
            self.pretrain_path = download_file(
                "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
                "dino_deitsmall8_pretrain.pth",
            ) if pretrain_path is None else pretrain_path
        else:
            raise ValueError("ViT models only supported with patch sizes 8 or 16")

        if self.high_res:
            self.num_patches *= 2

        self.model = None
        self.load_model()

    def load_model(self):
        model = vits.__dict__[self.model_name](patch_size=self.patch_size)
        state_dict = torch.load(self.pretrain_path, map_location="cpu")
        model.load_state_dict(state_dict)
        # model.to(device)
        model.eval()

        if self.high_res:
            model.patch_embed.proj.stride = (self.stride, self.stride)
            model.num_patches = self.num_patches ** 2
            model.patch_embed.patch_size = self.stride
            model.patch_embed.proj.padding = self.padding
        self.model = model

    def extract_features_and_attn(self, all_images):
        """
        A definition of relevant dimensions {all_b, nh, t, d}:
            image_size: Side length of input images (assumed square)
            all_b: The first dimension size of the input tensor - not necessarily
                the same as "batch size" in high-level script, as we assume that
                reference and target images are all flattened-then-concatenated
                along the batch dimension. With e.g. a batch size of 2, and 5 target
                images, 1 reference image; all_b = 2 * (5+1) = 12
            h: number of heads in ViT, e.g. 6
            t: number of items in ViT keys/values/tokens, e.g. 785 (= 28*28 + 1)
            d: feature dim in ViT, e.g. 64

        Args:
            all_images (torch.Tensor): shape (all_b, 3, image_size, image_size)
        Returns:
            features (torch.Tensor): shape (all_b, nh, t, d) e.g. (12, 6, 785, 64)
            attn (torch.Tensor): shape (all_b, nh, t, t) e.g. (12, 6, 785, 785)
            output_cls_tokens (torch.Tensor): shape (all_b, nh*d) e.g. (12, 384)
        """
        MAX_BATCH_SIZE = 50
        all_images_batch_size = all_images.size(0)
        c, img_h, img_w = all_images.shape[-3:]
        all_images = all_images.view(-1, c, img_h, img_w)

        with torch.no_grad():
            torch.cuda.empty_cache()

            if all_images_batch_size <= MAX_BATCH_SIZE:
                data = self.model.get_specific_tokens(all_images, layers_to_return=(9, 11))
                features = data[self.feat_layer]["k"]
                attn = data[11]["attn"]
                output_cls_tokens = data[11]["t"][:, 0, :]

            # Process in chunks to avoid CUDA out-of-memory
            else:
                num_chunks = np.ceil(all_images_batch_size / MAX_BATCH_SIZE).astype("int")
                data_chunks = []
                for i, ims_ in enumerate(all_images.chunk(num_chunks)):
                    data_chunks.append(self.model.get_specific_tokens(ims_, layers_to_return=(9, 11)))

                features = torch.cat([d[self.feat_layer]["k"] for d in data_chunks], dim=0)
                attn = torch.cat([d[11]["attn"] for d in data_chunks], dim=0)
                output_cls_tokens = torch.cat([d[11]["t"][:, 0, :] for d in data_chunks], dim=0)

        return features, attn, output_cls_tokens

    def forward(self, img):
        features, attn, output_cls_tokens = self.extract_features_and_attn(img)
        #pdb.set_trace()
        features = features[:, :, 1:, :]
       
        id = 2
        probas = features.mean(1).softmax(-1)[id].cpu()
        keep = probas.max(-1).values > 0.04
        vis_indexs = torch.nonzero(keep).squeeze(1)

        features = features.permute(0, 1, 3, 2)
        bsz, nh, d, t = features.shape
        hf, wf = int(np.sqrt(t)), int(np.sqrt(t))
        # hf=7
        # wf=10
        features = features.reshape(bsz, d * nh, hf, wf)  # bsz, d*nh, h, w

        cls_attn = attn[:,:,0,1:].reshape(bsz, -1, hf, wf) 
        #pdb.set_trace()
        
        is_visualize = False
        if is_visualize:
            attention = attn[id,:,1:,1:]
            cls_attn = torch.nn.functional.interpolate(cls_attn, size=(160, 160), mode='bilinear', align_corners=False).cpu().numpy()

            for i in range(bsz):
                Cls_attn_path = f"/data/geyan21/projects/nerf-ssl-master/correspondence/Cls_attn/Cls_{i}.png"
                plt.imsave(fname=Cls_attn_path, arr=cls_attn[i].mean(0), format='png')

            
            Vis_attn_path = f"/data/geyan21/projects/nerf-ssl-master/correspondence/Vis_attn/img_{id}"
            for vis_index in vis_indexs:
                token_dir = os.path.join(Vis_attn_path, 'Dino-Tok-'+str(int(vis_index)))
                if not os.path.exists(token_dir):
                    os.makedirs(token_dir)
                vis_attn = attention[:, vis_index, :] # nh,t
                mean_attention = get_one_query_meanattn(vis_attn, hf, wf)
                mean_attention = mean_attention[0]
                fname = os.path.join(token_dir, "attn-head-mean" + ".png")
                #pdb.set_trace()
                plt.imsave(fname=fname, arr=mean_attention, format='png')
                print(f"{fname} saved.")
                attn = get_one_query_attn(vis_attn, hf, wf, nh)
                for j in range(nh):
                    fname = os.path.join(token_dir, "attn-head" + str(j) + ".png")
                    plt.imsave(fname=fname, arr=attn[j], format='png')
                    print(f"{fname} saved.")

            features_ = torch.nn.functional.interpolate(features, size=(80, 80), mode='bilinear', align_corners=False) # (N, C, H, W)
            feature_map = features_[id].permute(1, 2, 0) # to H, W, C
            feature_map = feature_map.detach().cpu().numpy()


            feature_test_path = f"/data/geyan21/projects/nerf-ssl-master/correspondence/feature_map_{id}.png"
            plt.imsave(fname=feature_test_path, arr=feature_map.mean(-1), format='png')
            
            torchvision.utils.save_image(img[id], f"/data/geyan21/projects/nerf-ssl-master/correspondence/img_{id}.png")
            
            #pdb.set_trace()
            #plot_feature_map_channel(features)

        return features, cls_attn


def get_one_query_meanattn(vis_attn,h_featmap,w_featmap):
    mean_attentions = vis_attn.mean(0).reshape(h_featmap, w_featmap)
    mean_attentions = nn.functional.interpolate(mean_attentions.unsqueeze(0).unsqueeze(0), scale_factor=32, mode="bicubic")[0].cpu().numpy()
    return mean_attentions

def get_one_query_attn(vis_attn, h_featmap, w_featmap, nh):
    attentions = vis_attn.reshape(nh, h_featmap, w_featmap)
    # attentions = vis_attn.sum(0).reshape(h_featmap, w_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=32, mode="bicubic")[0].cpu().numpy()
    return attentions

# def plot_feature_map_channel(features,is_show=False,is_save=True):
#     n, c, h, w = features.size()
#     fig, ax = plt.subplots(1, c)
#     for i in range(c):
#         #out = features[0][i].cpu().data.resize_(h, w)
#         out = features[0][i].cpu().squeeze()
#         imsave(os.path.join("/data/geyan21/projects/nerf-ssl-master/correspondence/feature_map", 'channel_' + str(i) + '.png'), out) if is_save else None
#         if is_show:
#             ax[i].set_title(f'Feature map (channel {i})')
#             ax[i].imshow(features[0][i])
#             # ax[i].imshow(features[0][i], cmap='Blues')

#     if is_show:
#         plt.xticks([]), plt.yticks([])
#         plt.show()

#     return 0

# def imsave(file_name, img):
#     """
#     save a torch tensor as an image
#     :param file_name: 'image/folder/image_name'
#     :param img: c*h*w torch tensor
#     :return: nothing
#     """
#     assert(type(img) == torch.FloatTensor,
#            'img must be a torch.FloatTensor')
#     ndim = len(img.size())
#     assert(ndim == 2 or ndim == 3,
#            'img must be a 2 or 3 dimensional tensor')

#     img = img.numpy()

#     if ndim == 3:
#         plt.imsave(file_name, np.transpose(img, (1, 2, 0)))
#     else:
#         plt.imsave(file_name, img, cmap='gray')


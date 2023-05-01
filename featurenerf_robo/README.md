# Self-Supervised Learning with NeRF for Visual RL


current tasks:
- shelfplacing
- lift
- reachwall
- peginsert
- stacking
- reach
- pegbox
- push

# Usage
## BC

gen keyframe demonstrations: bash scripts/bc/gen_demonstration_keyframe.sh lift

train BC agent with 3D info: bash scripts/bc/train_bc_depth.sh lift pointnet debug

train BC agent with keyframe: bash scripts/bc/train_bc_keyframe.sh lift simple debug


train Diffusion-BC agent: bash scripts/train_bc_diffusion.sh

## pretraining
get dino feature:
```
bash scripts/nerf/extract_dino_feature.sh
```

train nerf+dino:
```
bash scripts/nerf/train_featurenerf.sh
```

train 2d student:
```
bash scripts/nerf/train_2d.sh
```

pointnet2 cls: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth

# New Baselines
1. [MoCo v2](https://arxiv.org/abs/2003.04297) (r50)
2. [Pri3D](https://arxiv.org/abs/2104.11225) (r50)
    - install MinkowskiEngine: `pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps`
3. [DINO](https://arxiv.org/abs/2104.14294) (vit-s/8)


# MuJoCo
package: https://www.roboti.us/download/mujoco200_linux.zip
key: https://www.roboti.us/file/mjkey.txt

# Code Standards

## Config
1. **Usage**. All config files are saved in `configs/`, and are separated into four modes: `bc,rl,nerf,distill`. Each mode has a `default.yaml` file, which is the default config for the corresponding exp. Each exp also has a `*.yaml` file, which is the config for your exp. 
2. **Write your own config**. All config files are in the format of `yaml`. You can use `yaml` to write your own config file. For example, you can write `configs/rl/your_exp.yaml` and specify it by setting `alg_config=your_exp` in the command line and set `mode=rl` in the code.
3. **Config priority**. The priority of config is `configs/default.yaml` < `configs/{mode}/default.yaml` < `config/{mode}/your_exp.yaml` < command line config. The higher priority config will overwrite the lower priority config. 
4. **Config in script**. The config in the script is the highest priority. It will overwrite all other configs. So if you want to change some config, you can change it in the script for simple debugging. We already have nice scripts in `scripts/` for you to use. You can also write your own scripts for your own exp.

## Coding Style
1. Please use `image_size` to denote the size of image instead of `img_size`. If see `img_size`, please change it to `image_size`.
2. Please use `num` to denote the number of something instead of `n`. If see `n`, please change it to `num`.
3. Please do not hard code things as much as possible. For example, if you want to use `image_size=64`, please set it in the config file instead of hard coding it in the code.
4. Please do not write wandb key in the code. Please set it in your `~/.bashrc` file.
5. Please add **two** blank lines between functions, and add **three** blank lines between classes.


## Logging
1. The root log dir is specified by `log_dir_root` in the config file. Please do not change it.
2. To log different exps, please change `log_dir` in the config file. For example, if you want to log an `rl` exp called `sac_debug`, please set `log_dir=sac_debug` in the config file, and the logging system will automatically create a folder `logs/rl/sac_debug` to save the logs.
3. We use a unified log system, in `logger.py`. It is simple to use. It also supports **wandb**. Please refer to `logger.py` for more details. 
4. To use wandb, set `use_wandb=True` and set your `wandb_entity, wandb_group, wandb_project` in the config file.




# References
1. CVPR 2021, pixelNeRF: Neural Radiance Fields from One or Few Images
2. CoRL 2021, Dex-NeRF: Using a Neural Radiance field to Grasp Transparent Objects
3. ICCV 2021, Pri3D: Can 3D Priors Help 2D Representation Learning?
4. CVPR 2020, Momentum Contrast for Unsupervised Visual Representation Learning

# Env

conda create -n nerf python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install imageio
pip install scikit-image
pip install opencv-python
pip install pyhocon
pip install matplotlib
pip install dotmap
pip install tqdm, termcolor
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --upgrade PyMCubes
pip install trimesh
pip install pyrender
pip install -U open3d-python
pip install --upgrade jupyter_client

# Use visdom to vis pointcloud
python -m visdom.server


# dino
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
mkdir ckpts
mv dino_vitbase8_pretrain.pth ckpts/

# Pri3D
gdown 1Whlny5aSH5tqD2Xe79Q7uUP1QD78Wj8t -O ckpts/pri3d.pth

# MoCo v2
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar -O ckpts/mocov2.pth.tar
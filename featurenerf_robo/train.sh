# demo:
#       bash scripts/nerf/train_featurenerf.sh robo_dino_aug

gpu=2

cd featurenerf

# export WANDB_MODE=offline

# datadir="/data/geyan21/projects/nerf-ssl-master/Data/realRobo"
# datadir="/data/geyan21/projects/nerf-ssl-master/Data/NerfSmall"
# datadir="/data/geyan21/projects/nerf-ssl-master/Data/Nerf128"
# datadir="/data/geyan21/projects/nerf-ssl-master/Data/Nerfdata"
datadir="/data/geyan21/projects/nerf-ssl-master/Data/Nerfhl"

config_name=robo_dino_real
# config_name=robo_dino_real_Attn
#config_name=${1}
#config_name=robo_dino_real_Attn

exp_name="${config_name}_nerfhl_bs512"

CUDA_VISIBLE_DEVICES=${gpu} python train/train_embed.py --datadir ${datadir} \
        --name ${exp_name} \
        --conf conf/exp/${config_name}.conf \
        --epochs 100000 \
        --ray_batch_size 512

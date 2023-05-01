# demo:
#       bash scripts/nerf/train_featurenerf.sh robo_dino_aug

gpu=2

cd featurenerf

# export WANDB_MODE=offline

datadir="/data/yanjieze/projects/nerf-act/data/nerf_data"

# config_name=robo_dino_r34_all
# config_name=robo_clip_r34_all
config_name=${1}

exp_name="${config_name}_augnerf_stagesplit"

CUDA_VISIBLE_DEVICES=${gpu} python train/train_embed.py --datadir ${datadir} \
        --name ${exp_name} \
        --conf conf/exp/${config_name}.conf \
        --resume \
        --epochs 40000 \
        --ray_batch_size 128

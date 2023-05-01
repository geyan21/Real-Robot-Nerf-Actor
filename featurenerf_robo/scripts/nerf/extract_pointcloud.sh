# bash scripts/nerf/extract_pointcloud.sh

cd featurenerf

# export WANDB_MODE=offline

datadir="/data/geyan21/projects/nerf-ssl-master/Data/Nerfhl"
gpu_id=3

config_name="robo_dino_real"
exp_name="robo_dino_real_nerfhl_bs512"

CUDA_VISIBLE_DEVICES=${gpu_id} python eval/extract_pointcloud.py --datadir ${datadir} \
        --dataset_format "realrobot" \
        --name ${exp_name} \
        --conf conf/exp/${config_name}.conf \

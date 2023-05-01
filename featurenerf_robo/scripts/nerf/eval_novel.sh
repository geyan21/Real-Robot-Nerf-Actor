# bash scripts/nerf/eval_novel.sh
cd featurenerf

# export WANDB_MODE=offline

datadir="/data/geyan21/projects/nerf-ssl-master/Data/realRobo"

gpu_id=0

config_name="robo_dino_real"
exp_name="robo_dino_real"

CUDA_VISIBLE_DEVICES=${gpu_id} python eval/eval_novel.py --datadir ${datadir} \
        --dataset_format "realrobot" \
        --name ${exp_name} \
        --conf conf/exp/${config_name}.conf \
        --distill_active 1

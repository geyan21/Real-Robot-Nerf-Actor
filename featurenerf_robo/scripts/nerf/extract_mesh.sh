# bash scripts/nerf/extract_mesh.sh

cd featurenerf

# export WANDB_MODE=offline

datadir="../nerf_data/"
gpu_id=1

config_name="robo_dino_r18_all_eval"
exp_name="robo_dino_r18_all"

CUDA_VISIBLE_DEVICES=${gpu_id} python eval/extract_mesh.py --datadir ${datadir} \
        --dataset_format "robo" \
        --name ${exp_name} \
        --conf conf/exp/${config_name}.conf \
        --save-dir ../data/meshes \
        --limit 1.5 \
        --res 200 \
        --iso-level 32 \
        --view-disparity-max-bound 1e0 \
        --batch-size 4096

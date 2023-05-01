# demo:
#       bash scripts/nerf/train_2d.sh

cd featurenerf

# export WANDB_MODE=offline

datadir="../nerf_data/"



### r18 ###
config_name=robo_2d_r18_shelfplacing
# config_name=robo_2d_r18_all
exp_name=${config_name}

CUDA_VISIBLE_DEVICES=4 python train/train_2d.py --datadir ${datadir} \
        --name ${exp_name} \
        --conf conf/exp/${config_name}.conf \
        --resume \
        --epochs 20000 \
        --ray_batch_size 128 

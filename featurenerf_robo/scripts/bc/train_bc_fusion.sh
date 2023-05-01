# demo: 
# 	bash scripts/bc/train_bc_fusion.sh shelfplacing mocov2 indomain
#	bash scripts/bc/train_bc_fusion.sh lift pointnerf 128 1206 

gpu_used=1


use_wandb=False

task_name=${1}
action_space=xyzw
config=${2}
featurenerf_mode=spatial
# featurenerf_mode=global
num_points=${3}
date=${4}

wandb_group="${task_name}_${config}_nump${num_points}_${date}"

log_dir="${wandb_group}"

domain_randomization=False


CUDA_VISIBLE_DEVICES=${gpu_used} python src/train_bc_fusion.py \
	alg_config=${config} \
	log_dir=${log_dir} \
	task_name=${task_name} \
	action_space=${action_space} \
	use_wandb=${use_wandb} \
	wandb_group=${wandb_group} \
	domain_randomization=False \
	num_trajs=100 \
	save_video=False \
	num_epochs=1000 \
	use_robot_state=True \
	num_points=${num_points} \


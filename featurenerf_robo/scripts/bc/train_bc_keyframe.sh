# demo: 
# 	bash scripts/bc/train_bc_keyframe.sh shelfplacing mocov2 indomain
#	bash scripts/bc/train_bc_keyframe.sh lift simple debug

gpu_used=1


use_wandb=False

task_name=${1}
action_space=xyzw
config=${2}
featurenerf_mode=global
date=${3}

wandb_group="${task_name}_${config}_${date}"

log_dir="${wandb_group}"

domain_randomization=False


CUDA_VISIBLE_DEVICES=${gpu_used} python src/train_bc_keyframe.py \
	alg_config=${config} \
	log_dir=${log_dir} \
	task_name=${task_name} \
	action_space=${action_space} \
	use_wandb=${use_wandb} \
	wandb_group=${wandb_group} \
	domain_randomization=False \
	num_trajs=100 \
	save_video=True \
	num_epochs=500 \
	use_robot_state=True \
	freeze_encoder=False \


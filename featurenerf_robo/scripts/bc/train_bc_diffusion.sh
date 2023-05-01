# demo: 
#	bash scripts/bc/train_bc_diffusion.sh peginsert debug

gpu_used=1


use_wandb=False

task_name=${1}
action_space=xyzw
config="diffusion"
date=${2}



wandb_group="${task_name}_${config}_${date}"

log_dir="${wandb_group}"




CUDA_VISIBLE_DEVICES=${gpu_used} python src/train_bc_diffusion.py \
	alg_config=${config} \
	log_dir=${log_dir} \
	task_name=${task_name} \
	action_space=${action_space} \
	use_wandb=${use_wandb} \
	wandb_group=${wandb_group} \
	domain_randomization=False \
	num_trajs=10 \


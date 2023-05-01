# demo: 
# 	bash scripts/bc/nerf_to_pc.sh peginsert


gpu_used=1


use_wandb=True

task_name=${1}
action_space=xyzw

domain_randomization=False


CUDA_VISIBLE_DEVICES=${gpu_used} python src/nerf_to_pc.py \
	alg_config="nerf_to_bc" \
	log_dir="debug" \
	task_name=${task_name} \
	action_space=${action_space} \
	domain_randomization=False \
	use_robot_state=True \


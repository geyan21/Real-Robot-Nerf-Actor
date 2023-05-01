# bash scripts/rl/train_rl.sh dmc finger-spin xyz dino 1000k debug
# bash scripts/rl/train_rl.sh robot peginsert xyzw mvp 1000k debug
# bash scripts/rl/train_rl.sh robot shelfplacing xyzw sacv2 3m morerandom

buffer_capacity=100k

use_wandb=1


domain_name=$1
task_name=$2
action_space=$3
alg_config=${4}
# observation_type=state+image
observation_type=state
train_steps=$5
date=$6


domain_randomization=0


wandb_group="${domain_name}_${task_name}_${alg_config}_${date}"
log_dir="${wandb_group}"


CUDA_VISIBLE_DEVICES=1 python src/train_rl.py \
	alg_config=${alg_config} \
	domain_name=${domain_name} \
	task_name=${task_name} \
	action_space=${action_space} \
	observation_type=${observation_type} \
	buffer_capacity=${buffer_capacity} \
	wandb_group=${wandb_group} \
	use_wandb=${use_wandb} \
	domain_randomization=${domain_randomization} \
	log_dir=${log_dir} \
	train_steps=${train_steps} \
	save_model=1 \



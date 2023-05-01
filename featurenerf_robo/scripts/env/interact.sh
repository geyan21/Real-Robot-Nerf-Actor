# demo:
# 	bash scripts/env/interact.sh peginsert

task_name=${1}

num_static_cameras=1
num_dynamic_cameras=1
camera_mode=static+dynamic
# camera_mode=static
camera_move_range=90


CUDA_VISIBLE_DEVICES=0 python3 src/viz_interact.py \
	alg_config="viz_mujoco" \
	observation_type=state+image \
	domain_name=robot \
    task_name=${task_name} \
	action_space=xyzw \
	log_dir=viz \
	seed=0 \
	image_size=128 \
	camera_move_range=${camera_move_range} \
	save_video=True \
	domain_randomization=0 \
	num_static_cameras=1 \
	num_dynamic_cameras=1 \
	camera_mode=${camera_mode}


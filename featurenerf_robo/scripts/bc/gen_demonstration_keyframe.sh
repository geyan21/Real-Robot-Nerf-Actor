# demo: bash scripts/bc/gen_demonstration_keyframe.sh lift

gpu=1

task_name=${1}
resume_rl="policies/sacv2_${task_name}.pt"

# eval episodes: specify the number of trajectories for each environment
CUDA_VISIBLE_DEVICES=${gpu} python src/gen_demonstration_keyframe.py \
                        alg_config="gen_demonstration_keyframe" \
                        num_trajs=100 \
                        task_name=${task_name} \
                        resume_rl=${resume_rl} \
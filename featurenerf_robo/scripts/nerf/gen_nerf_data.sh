# demo
#   bash scripts/nerf/gen_nerf_data.sh push

task_name=${1}

python src/gen_nerf_data.py \
        alg_config="gen_nerf_data" \
        task_name=${task_name} \
        domain_randomization=False \
        num_scenes=20 \

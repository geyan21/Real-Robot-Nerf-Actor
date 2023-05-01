# Training

```sh
# 1-view, dino
python train/train_embed.py -n feature_plane_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 02691156 --epochs 500
python train/train_embed.py -n feature_table_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 04379243 --epochs 500
python train/train_embed.py -n feature_motorbike_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03790512 --epochs 2000
python train/train_embed.py -n feature_lamp_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03636649 --epochs 1000
python train/train_embed.py -n feature_chair_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03001627 --epochs 500
python train/train_embed.py -n feature_car_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 02958343 --epochs 500
python train/train_embed.py -n feature_bed_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 02818832 --epochs 2000
python train/train_embed.py -n feature_bottle_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02876657 --epochs 1000
python train/train_embed.py -n feature_mug_dino_256_0.25_share_coord_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 03797390 --epochs 2000

# 1-view, diffusion
python train/train_embed.py -n srn_chairs_diff_512_0.25_share_coord_exp -c conf/exp/srn_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/raw/nerf/pixel_nerf_data/srn_chairs/chairs --epochs 500
python train/train_embed.py -n feature_plane_diff_512_0.25_share_coord_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02691156 --epochs 500
python train/train_embed.py -n feature_motorbike_diff_512_0.25_share_coord_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 03790512 --epochs 2000
python train/train_embed.py -n feature_car_diff_512_0.25_share_coord_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02958343 --epochs 500
python train/train_embed.py -n feature_bed_diff_512_0.25_share_coord_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02818832 --epochs 2000
python train/train_embed.py -n feature_bottle_diff_512_0.25_share_coord_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02876657 --epochs 1000

python train/train_embed.py -n srn_cars_diff_512_0.25_share_coord_exp -c conf/exp/srn_diff_512_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/raw/nerf/pixel_nerf_data/srn_cars/cars --epochs 500
python train/train_embed.py -n feature_table_diff_512_0.25_share_coord_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 04379243 --epochs 500
python train/train_embed.py -n feature_lamp_diff_512_0.25_share_coord_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03636649 --epochs 1000

# 2-view, dino
python train/train_embed.py -n srn_chairs_dino_256_0.25_share_coord_2v_exp -c conf/exp/srn_dino_256_0.25_share_coord.conf -D /data2/jianglong/data/raw/nerf/pixel_nerf_data/srn_chairs/chairs -V 2 --epochs 500
python train/train_embed.py -n srn_cars_dino_256_0.25_share_coord_2v_exp -c conf/exp/srn_dino_256_0.25_share_coord.conf -D /data2/jianglong/data/raw/nerf/pixel_nerf_data/srn_cars/cars -V 2 --epochs 500
python train/train_embed.py -n feature_plane_dino_256_0.25_share_coord_2v_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02691156 -V 2 --epochs 500

python train/train_embed.py -n feature_chair_dino_256_0.25_share_coord_2v_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03001627 -V 2 --epochs 500
python train/train_embed.py -n feature_motorbike_dino_256_0.25_share_coord_2v_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03790512 -V 2 --epochs 2000
python train/train_embed.py -n feature_table_dino_256_0.25_share_coord_2v_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 04379243 -V 2 --epochs 500
python train/train_embed.py -n feature_lamp_dino_256_0.25_share_coord_2v_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03636649 -V 2 --epochs 1000
python train/train_embed.py -n feature_bed_dino_256_0.25_share_coord_2v_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 02818832 -V 2 --epochs 2000
python train/train_embed.py -n feature_bottle_dino_256_0.25_share_coord_2v_exp -c conf/exp/feature_dino_256_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 02876657 -V 2 --epochs 1000

# 2-view, diffusion
python train/train_embed.py -n srn_chairs_diff_512_0.25_share_coord_2v_exp -c conf/exp/srn_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/raw/nerf/pixel_nerf_data/srn_chairs/chairs -V 2 --epochs 500
python train/train_embed.py -n feature_motorbike_diff_512_0.25_share_coord_2v_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 03790512 -V 2 --epochs 2000
python train/train_embed.py -n feature_plane_diff_512_0.25_share_coord_2v_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02691156 -V 2 --epochs 500
python train/train_embed.py -n feature_bottle_diff_512_0.25_share_coord_2v_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /data2/jianglong/data/processed/feature_nerf --synset 02876657 -V 2 --epochs 1000

python train/train_embed.py -n srn_cars_diff_512_0.25_share_coord_2v_exp -c conf/exp/srn_diff_512_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/raw/nerf/pixel_nerf_data/srn_cars/cars -V 2 --epochs 500
python train/train_embed.py -n feature_table_diff_512_0.25_share_coord_2v_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 04379243 -V 2 --epochs 500
python train/train_embed.py -n feature_lamp_diff_512_0.25_share_coord_2v_exp -c conf/exp/feature_diff_512_0.25_share_coord.conf -D /mnt/weka/scratch/jianglong.ye/data/processed/feature_nerf/ --synset 03636649 -V 2 --epochs 1000

# 1-view, baseline
python train/train.py -n feature_car_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02958343 --epochs 500
python train/train.py -n feature_plane_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02691156 --epochs 500
python train/train.py -n feature_lamp_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 03636649 --epochs 1000
python train/train.py -n feature_table_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 04379243 --epochs 500
python train/train.py -n feature_bed_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02818832 --epochs 2000
python train/train.py -n feature_bottle_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02876657 --epochs 1000
python train/train.py -n feature_mug_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 03797390 --epochs 2000

# 2-view, baseline
python train/train.py -n feature_chair_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 03001627 -V 2 --epochs 500
python train/train.py -n feature_car_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02958343 -V 2 --epochs 500
python train/train.py -n feature_plane_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02691156 -V 2 --epochs 500
python train/train.py -n feature_lamp_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 03636649 -V 2 --epochs 1000
python train/train.py -n feature_table_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 04379243 -V 2 --epochs 500
python train/train.py -n feature_bed_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02818832 -V 2 --epochs 2000
python train/train.py -n feature_bottle_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 02876657 -V 2 --epochs 1000
python train/train.py -n feature_mug_2v_exp -c conf/exp/feature.conf -D /mnt/truenas/scratch/jianglong.ye/data/processed/feature_nerf --synset 03797390 -V 2 --epochs 2000
```
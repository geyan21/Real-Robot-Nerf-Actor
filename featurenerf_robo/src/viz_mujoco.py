import torch
import os

import numpy as np
import gym
import utils
import time
import wandb
from config import parse_cfg
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import tqdm
from PIL import Image
from torchvision.utils import make_grid, save_image
import torchvision


def visualize_configurations(env, cfg):

	
	frames = []
	figure_num=8
	for i in tqdm.tqdm(range(figure_num)):
		env.seed(i)
		env.reset()
		frame = torch.from_numpy(env.render_obs(mode='rgb_array',
				height=cfg.image_size, width=cfg.image_size, camera_id=None).copy()).squeeze(0)
		frame0 = frame[0].permute(2,0,1).float().div(255)
		frame1 = frame[1].permute(2,0,1).float().div(255)
		frames.append(frame0)
		frames.append(frame1)

		if not os.path.exists("env_imgs"):
			os.mkdir("env_imgs")
	
	save_image(make_grid(torch.stack(frames), nrow=4), f'env_imgs/{cfg.domain_name}_{cfg.task_name}_{str(cfg.image_size)}.png')
		# save_image(frame0, f'env_imgs_paper/{cfg.domain_name}_{cfg.task_name}_{str(cfg.image_size)}_{str(i)}.png')



def videolize_configurations(env, cfg, camera="front"):
	video_dir = "env_videos"
	fps=10
	if not os.path.exists(video_dir):
		os.mkdir(video_dir)

	video = VideoRecorder("env_imgs", height=cfg.image_size, width=cfg.image_size, fps=fps)
	episode_rewards = []
	success_rate = []

	obs = env.reset()
	video.init(enabled=1)
	done = False
	episode_reward = 0
	step=0
	debug = False
	

	# while not done:
	cfg.episode_length = 50
	obses = []
	for count in range(cfg.episode_length):
		action = env.action_space.sample()

		delta = 0.0
		if count < 25:
			gripper_delta = delta
		elif count < 50:
			gripper_delta = delta
		elif count < 75:
			gripper_delta = -delta
		else:	
			gripper_delta = -delta

		action = np.array((0, 0, 0, gripper_delta))
		obs, state, reward, done, info = env.step(action)
		to_store_img = env.render_obs(
                    mode='rgb_array',
                    height=224,
                    width=224,
                    camera_id="camera_dynamic"
                )
		obses.append(torch.from_numpy(obs[3:])) # 3x128x128
		
		print("step: %u | reward: %f"%(step, reward))
		step += 1
		episode_reward += reward

	print("is success:", info['is_success'] )
	if camera is None:
		camera="all"

	obses = torch.stack(obses) # 50x3x128x128
	obses = obses.permute(0,2,3,1) # 50x128x128x3
	torchvision.io.write_video(f"{video_dir}/{cfg.domain_name}_{cfg.task_name}_{str(cfg.image_size)}.mp4", obses, fps=fps) # 50x128x128x3


	print("video saved. episode reward is %f"%episode_reward)

def visualize_env(cfg):
	cameras = "static"
	cameras = "dynamic"
	cameras = None

	utils.set_seed_everywhere(cfg.seed)

	env = make_env(
		domain_name=cfg.domain_name,
		task_name=cfg.task_name,
		seed=cfg.seed,
		episode_length=cfg.episode_length,
		n_substeps=cfg.n_substeps,
		frame_stack=cfg.frame_stack,
		image_size=cfg.image_size,
		render=cfg.render, # Only render if observation type is state
		observation_type="state+image", # state, image, state+image
		action_space=cfg.action_space,
		camera_move_range=cfg.camera_move_range,
		domain_randomization=0,
		cameras=cfg.camera_mode,
		num_static_cameras=cfg.num_static_cameras,
		num_dynamic_cameras=cfg.num_dynamic_cameras,
	)
	

	visualize_configurations(env=env, 
							cfg=cfg)

	if cfg.save_video:
		videolize_configurations(env=env, 
								cfg=cfg,
								camera=cameras)

if __name__=='__main__':
	cfg = parse_cfg(cfg_path="configs", mode="rl")
	visualize_env(cfg)


	

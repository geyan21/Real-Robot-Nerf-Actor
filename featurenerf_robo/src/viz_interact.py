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


def get_action_from_keyboard():
	action = np.array((0., 0., 0., 0.))
	gripper_delta = 0.

	key = input("Enter action: ")
	if "w" in key:
		action[0] += 0.5
	if "s" in key:
		action[0] += -0.5
	if "a" in key:
		action[1] = 0.5
	if "d" in key:
		action[1] = -0.5
	if "q" in key:
		action[2] = 0.5
	if "e" in key:
		action[2] = -0.5
	if "o" in key:
		gripper_delta += -1
	if "p" in key:
		gripper_delta += 1
	action[3] = gripper_delta
	print("action:", action) # [x, y, z, gripper]
	return action


def save_obs(obs):
	obs = obs[:].reshape(2,3,128,128)
	torchvision.utils.save_image(torch.from_numpy(obs).div(255).float(), "env_imgs/interact.png") # 2x3x128x128


def videolize_configurations(env, cfg, camera="front"):
	video_dir = "env_videos"
	fps=10
	if not os.path.exists(video_dir):
		os.mkdir(video_dir)

	episode_rewards = []
	success_rate = []

	obs = env.reset()

	done = False
	episode_reward = 0
	step=0
	debug = False
	

	cfg.episode_length = 1000
	obses = []
	obs, state, info = env.reset() # Reset the environment
	save_obs(obs) # Save the observation

	for count in range(cfg.episode_length):
		action = get_action_from_keyboard() # Get action from keyboard

		obs, state, reward, done, info = env.step(action)
		print("is gripper open:", info["is_gripper_open"])
		save_obs(obs) # Save the observation

		# obses.append(torch.from_numpy(obs[3:])) # 3x128x128
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
		domain_randomization=cfg.domain_randomization,
		cameras=cfg.camera_mode,
		num_static_cameras=cfg.num_static_cameras,
		num_dynamic_cameras=cfg.num_dynamic_cameras,
	)
	
	videolize_configurations(env, cfg, camera=cameras) # Save video of the environment

if __name__=='__main__':
	cfg = parse_cfg(cfg_path="configs", mode="rl")
	visualize_env(cfg)


	

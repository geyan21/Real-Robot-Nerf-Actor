"""
Generat expert demonstrations for a domain and a task.
"""
from ast import arg
import torch
import torchvision
import os
import numpy as np
import gym
import utils
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from tqdm import tqdm
from config import parse_cfg
from env.wrappers import make_env
from algorithms.factory import make_agent
import augmentations
from termcolor import colored
from PIL import Image
import cv2
from mj_pc.mj_point_clouds import PointCloudGenerator

def to_tensor_img(img):
	imgs = []
	for i in range(img[:].shape[0]//3):
		imgs.append(torch.from_numpy(img[i*3:(i+1)*3, :, :]).float().div(255.0))
	return torch.stack(imgs, dim=0)

def to_tensor_depth(depth):
	depths = []
	for i in range(depth.shape[0]):
		depths.append(torch.from_numpy(depth[i:i+1, :, :]).float())
	return torch.stack(depths, dim=0)

def evaluate_and_store(env, agent, num_trajs, cfg):
	episode_rewards = []
	success_rates = []

	# specify the path
	STORE_PATH="expert_demonstrations_keyframe/{}_{}".format(cfg.domain_name, cfg.task_name)
	if not os.path.exists(STORE_PATH):
		os.makedirs(STORE_PATH)


	pc_generator = PointCloudGenerator(env.sim, min_bound=(-1., -1., -1.), max_bound=(1., 1., 1.), \
					img_height=cfg.image_size, img_width=cfg.image_size)


	nominate_function_directory = lambda x: "{0:06}".format(x)
	nominate_function_img = lambda x, idx: "{0:03}".format(x)+"_%u.png"%idx
	nominate_function_depth = lambda x, idx: "{0:03}".format(x)+"_%u_depth.png"%idx
	nominate_function_pc = lambda x: "{0:03}".format(x)+"_pc.npy"

	num_success = 0
	skip_times = 0
	reward_list = []
	print("Generating {} trajectories".format(num_trajs))
	while(num_success < num_trajs):
		i = num_success
		obs, state, info = env.reset() # by default we use image+state env
		state_full = env.get_state_obs()
		done = False
		episode_reward = 0
		episode_success = 0
		episode_length = 0

		# make dir for one trajectory
		traj_idx = nominate_function_directory(i)
		traj_dir = os.path.join(STORE_PATH, traj_idx)
		if not os.path.exists(traj_dir):
			os.makedirs(traj_dir)
		os.makedirs(os.path.join(traj_dir, "rgb"), exist_ok=True)
		os.makedirs(os.path.join(traj_dir, "depth"), exist_ok=True)
		os.makedirs(os.path.join(traj_dir, "video"), exist_ok=True)
		os.makedirs(os.path.join(traj_dir, "transition"), exist_ok=True)
		os.makedirs(os.path.join(traj_dir, "point_cloud"), exist_ok=True)

		# print(colored("Generate trajectory {} in {}".format(traj_idx, traj_dir), "green"))
		
		img_to_save = []
		depth_to_save = []
		pc_to_save = []

		# save first frame
		tensor_img = to_tensor_img(obs) # ?x3ximg_sizeximg_size
		img_to_save.append(tensor_img)
		
		depth_dynamic = pc_generator.captureImage(cam_ind=0) # dynamic camera
		depth_static = pc_generator.captureImage(cam_ind=1) # static camera
		depth_all = np.stack([depth_dynamic, depth_static], axis=0)
		tensor_depth = to_tensor_depth(depth_all) # 2x1ximg_sizeximg_size
		depth_to_save.append(tensor_depth)
		
		point_cloud = pc_generator.generateCroppedPointCloud()
		pc_to_save.append(point_cloud)



		
		
		transition_to_save = []

		# begin to iterat and save
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(state_full, state)

			pre_obs = obs
			pre_state = state
			pre_state_full = state_full
			pre_info = info

			obs, state, reward, done, info = env.step(action)
			state_full = env.get_state_obs()
			transition_to_save.append(
				dict(
					full_state = pre_state_full,
					state=pre_state,
					info=pre_info,
					action=action,
					reward=reward,
					done=done,
					next_full_state=state_full,
					next_state=state,
					next_info=info,
				)
			)
			# print("is_gripper_close: ", info["is_gripper_close"])
			# print("gripper angle: ", info["gripper_angle"])


			episode_reward += reward
			episode_success = episode_success or info['is_success']
			episode_length += 1.0
			
			# save current frame
			tensor_img = to_tensor_img(obs) # ?x3x84x84
			img_to_save.append(tensor_img)
			
			depth_dynamic = pc_generator.captureImage(cam_ind=0) # dynamic camera
			depth_static = pc_generator.captureImage(cam_ind=1) # static camera
			depth_all = np.stack([depth_dynamic, depth_static], axis=0)
			tensor_depth = to_tensor_depth(depth_all) # 2x1ximg_sizeximg_size
			depth_to_save.append(tensor_depth)

			point_cloud = pc_generator.generateCroppedPointCloud()
			pc_to_save.append(point_cloud)


		if num_success < 3 or episode_success:
			num_success += 1
			print(colored("Generate trajectory {} in {} with R {}".format(traj_idx, traj_dir, episode_reward), "cyan"))
			
			

			# store transitions
			np.save(
				os.path.join(traj_dir, "transition", "transitions.npy"), 
				transition_to_save
			)
			# store images
			for time_step in range(len(img_to_save)):
				for i in range(tensor_img.shape[0]):
					img_i = img_to_save[time_step][i]
					torchvision.utils.save_image(img_i,  os.path.join(traj_dir, "rgb", nominate_function_img(int(time_step), i)) )

			# store as video in PIL.Image
			img_to_save = [x[0:1] for x in img_to_save]
			img_to_save = torch.cat(img_to_save, dim=0)
			img_to_save = img_to_save.permute(0, 2, 3, 1).numpy()
			img_to_save = img_to_save * 255
			img_to_save = img_to_save.astype(np.uint8)
			img_to_save = [Image.fromarray(img) for img in img_to_save]
			img_to_save[0].save(os.path.join(traj_dir, "video", "video.gif"), save_all=True, append_images=img_to_save[1:], duration=100, loop=0)

			# store depth
			for time_step in range(len(depth_to_save)):
				for i in range(tensor_depth.shape[0]):
					depth_i = depth_to_save[time_step][i]
					# save depth as original value
					cv2.imwrite(os.path.join(traj_dir, "depth", nominate_function_depth(int(time_step), i)), depth_i.numpy()[0])
					# load and check (there is a loss of precision)
					# depth_i_load = cv2.imread(os.path.join(traj_dir, nominate_function_depth(int(time_step), i)), cv2.IMREAD_UNCHANGED)
			
			# store point cloud
			for time_step in range(len(pc_to_save)):
				np.save(os.path.join(traj_dir, "point_cloud", nominate_function_pc(int(time_step))), pc_to_save[time_step])

		else:	
			skip_times += 1
			print(colored("skip with R {}.".format(episode_reward), "red"))
		

		episode_rewards.append(episode_reward)
		success_rates.append(episode_success)

	return np.nanmean(episode_rewards), np.nanmean(success_rates)


def main(cfg):


	# Set seed
	utils.set_seed_everywhere(cfg.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=cfg.domain_name,
		task_name=cfg.task_name,
		seed=cfg.seed+42,
		episode_length=cfg.episode_length,
		action_space=cfg.action_space,
		n_substeps=cfg.n_substeps,
		frame_stack=cfg.frame_stack,
		image_size=cfg.image_size,
		cameras=cfg.camera_mode, # ["static", "dynamic", "static+dynamic"]
		render=cfg.render, # Only render if observation type is state
		observation_type="state+image",
		camera_move_range=cfg.camera_move_range, # not use full range (360) to avoid the camera move to the same position
		num_static_cameras=cfg.num_static_cameras,
		num_dynamic_cameras=cfg.num_dynamic_cameras,
	)

	
	
	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	print('Observations:', env.observation_space.shape)
	print('Actions:', env.action_space.shape)


	agent = torch.load(cfg.resume_rl)
	agent.train(False)

	reward, success_rate = evaluate_and_store(env, agent, cfg.num_trajs, cfg)
	print('Reward:', int(reward))
	print('Success Rate:', float(success_rate))



if __name__ == '__main__':
	cfg = parse_cfg(cfg_path="configs", mode="bc")
	ckpt_path = cfg.resume_rl

	print(colored("task name: {}".format(cfg.task_name), "green"))
	print(colored("Load checkpoint from {}".format(ckpt_path), "green"))

	main(cfg)

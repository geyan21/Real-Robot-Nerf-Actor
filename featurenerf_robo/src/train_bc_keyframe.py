from representations import make_embedding
import torch
import torchvision
from torch.utils.data import DataLoader
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
import augmentations
import wandb
from algorithms.modules import ContinuousPolicy, RewardPredictor, GripperPolicy
from termcolor import colored
from logger import Logger
import time
import itertools
import matplotlib.pyplot as plt
from analysis.image import save_feature_map, save_rgb_image
import utils_keyframe

def process_raw_obs(raw_obs):
	"""
	process raw obs [0,255] from env into torch tensor [0,1]
	"""
	obs = torch.from_numpy(raw_obs[:]).float().div(255).cuda()
	return obs


def process_raw_state(raw_state):
	"""
	process raw state from env into torch tensor
	"""
	state = torch.from_numpy(raw_state).float().cuda()
	return state


def simple_motion_planning(xyz_current, gripper_current, xyz_pred, gripper_pred, num_steps=10):
	"""
	simple motion planner.
	alg:
	1. if gripper stage change, move gripper first
	2. move gripper to target position and keep gripper state
	"""
	actions = []
	if not gripper_current and gripper_pred: # not close gripper, try to close gripper
		# close the grippe
		for i in range(4):
			actions.append(np.array([0, 0, 0, -1]))
	
	if gripper_current and not gripper_pred: # closed gripper, try to open gripper
		# open the gripper
		for i in range(4):	
			actions.append(np.array([0, 0, 0, 1]))

	delta_xyz = (xyz_pred - xyz_current) / num_steps
	if gripper_pred:
		gripper_action = -1.
	else:
		gripper_action = 1.
	for i in range(num_steps):
		action = [delta_xyz[0], delta_xyz[1], delta_xyz[2], gripper_action]
		actions.append(np.array(action))
	return actions


def evaluate(env, embedding_model, policy, L, step, cfg):
	"""
	evaluation with motion planning
	"""
	num_episodes = 20
	success_rate = []
	returns = []
	for i in range(num_episodes):
		obs, state, info = env.reset()
		xyz_current = state[:3]
		gripper_current = info['is_gripper_close']

		done = False
		episode_reward = 0
		frames_static = []
		success = 0
		if i==0:
			img_obs = env.render_obs(width=128, height=128)
			frames_static.append(img_obs[0])


		episode_length = 100 # longer because of motion planning
		# while not done:
		for t in range(episode_length):
			with torch.no_grad():
				embed = embedding_model(process_raw_obs(obs)[:3].unsqueeze(0))
				if cfg.use_robot_state:
					xyz_pred, gripper_pred = policy(embed, process_raw_state(state).unsqueeze(0))
				else:
					xyz_pred, gripper_pred = policy(embed)
				xyz_pred = xyz_pred.squeeze(0).cpu().numpy()
				gripper_pred = gripper_pred.squeeze(0).cpu().numpy()
				gripper_pred = np.argmax(gripper_pred) # 0: open, 1: close
			
			# direct operate in mujoco
			actions = simple_motion_planning(xyz_current, gripper_current, xyz_pred, gripper_pred)
		
			for action in actions:	
				obs, state, reward, done, info = env.step(action)
				episode_reward += reward
				success = success or info['is_success']
				if i==0:
					img_obs = env.render_obs(width=128, height=128)
					frames_static.append(img_obs[0])
			xyz_current = state[:3]
			gripper_current = info['is_gripper_close']
			
				# frames_dynamic.append(img_obs[1])


		success_rate.append(success)
		returns.append(episode_reward)
	
		# save vide
		if cfg.save_video and i==0:
			L.log_video(frames_static, 'eval_static', step=step, category='eval')
			# L.log_video(frames_dynamic, 'eval_dynamic', step=step, category='eval')

	return np.nanmean(success_rate), np.nanmean(returns)


def to_tensor_img(img):
	img = torch.from_numpy(img[:]).view(2, 3, 84, 84).div(255)
	return img


def get_gripper_label_from_info(info_batch):
	gripper_labels = []
	for i in range(len(info_batch)):
		is_gripper_close = info_batch[i]['is_gripper_close']
		# make it binary
		if is_gripper_close:
			gripper_labels.append(torch.tensor([0., 1.]))
		else:	
			gripper_labels.append(torch.tensor([1., 0.]))
	# stack to tensor
	gripper_labels = torch.stack(gripper_labels).cuda()
	return gripper_labels
		

def main(cfg):

	# prepare dataset and dataloder
	demonstration_data_dir = os.path.join(cfg.dataset_root, "{}_{}".format(cfg.domain_name, cfg.task_name))
	image_size = cfg.image_size
	bc_dataset = utils.BehaviorCloneDataset(root_dir=demonstration_data_dir, episode_length=cfg.episode_length, \
									image_size=image_size, num_trajs=cfg.num_trajs, num_cameras=2)
	keyframe_buffer = utils_keyframe.KeyframeBuffer(bc_dataset)

	
	# prepare agent dimensions
	action_dim_dict = {"xy":(2,), "xyz":(3,), "xyzw":(4,)}
	state_dim_dict = {"xy":(3,), "xyz":(4,), "xyzw":(4,)}
	action_dim = action_dim_dict[cfg.action_space]
	observation_dim = (2*3, image_size, image_size) # default hard-coded value
	state_dim = state_dim_dict[cfg.action_space]
	

	# Set seed
	utils.set_seed_everywhere(cfg.seed + 42)


	# prepare embedding model
	assert torch.cuda.is_available(), 'must have cuda enabled'
	print('Observations:', observation_dim)
	print('Actions:', action_dim)

	embedding_model = make_embedding(model_name=cfg.embedding_name, cfg=cfg)
	embedding_model = embedding_model.cuda()
	if cfg.freeze_encoder:
		embedding_model.eval()
	else:
		embedding_model.train()


	# prepare policy model
	policy = GripperPolicy(input_dim=embedding_model.output_dim, action_dim=action_dim[0], state_dim=state_dim[0])
	policy = policy.cuda()


	# prepare optimizer and loss function
	optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
	optimizer_embed = torch.optim.Adam(embedding_model.parameters(), lr=cfg.lr)
	loss_fn_xyz = torch.nn.MSELoss()
	loss_fn_w = torch.nn.BCELoss()



	# create env for evaluation
	env = make_env(
		domain_name=cfg.domain_name,
		task_name=cfg.task_name,
		seed=cfg.seed+42,
		episode_length=cfg.episode_length,
		n_substeps=cfg.n_substeps,
		frame_stack=cfg.frame_stack,
		image_size=image_size,
		cameras=cfg.camera_mode,
		render=cfg.render, # Only render if observation type is state
		observation_type="state+image", # state, image, state+image
		action_space=cfg.action_space,
		camera_move_range=cfg.camera_move_range,
		domain_randomization=cfg.domain_randomization,
		num_static_cameras=cfg.num_static_cameras,
		num_dynamic_cameras=cfg.num_dynamic_cameras,
	)

	env.seed(cfg.seed)
	env.observation_space.seed(cfg.seed)
	env.action_space.seed(cfg.seed)


	# logger
	log_dir = f"{cfg.log_dir_root}/{cfg.log_dir}"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	L = Logger(log_dir, mode="bc", config=cfg)

	time_start = time.time()

	max3_success_rates = []
	max3_returns = []

	# iterate over epochs
	for it in range(cfg.num_iterations):

		# sample data
		batch = keyframe_buffer.uniform_sample(cfg.batch_size)
				
		obs_mini, state_mini, action_mini, reward_mini, next_obs_mini, next_state_mini, done_mini, info_mini, cummulative_translation = \
			batch['obs'], batch['state'], batch['action'], batch['reward'], batch['next_obs'], batch['next_state'], batch['done'], batch['info'], \
			batch['cummulative_translation']


		# convert [0,1] np array to torch tensor
		obs_mini = torch.from_numpy(obs_mini).float().cuda()
		action_mini = torch.from_numpy(action_mini).float().cuda()
		state_mini = torch.from_numpy(state_mini).float().cuda()
		next_state_mini = torch.from_numpy(next_state_mini).float().cuda()

		# get embed
		embed = embedding_model(obs_mini[:, :3])

		# predict action
		if cfg.use_robot_state:
			xyz_pred, gripper_pred = policy(embed, state_mini)
		else:
			xyz_pred, gripper_pred = policy(embed)


		# get label
		xyz_label = torch.from_numpy(cummulative_translation).float().cuda()
		gripper_label = get_gripper_label_from_info(info_mini)

		# compute loss
		loss_xyz = loss_fn_xyz(xyz_pred, xyz_label)
		loss_w = loss_fn_w(gripper_pred, gripper_label)

		loss = loss_xyz + loss_w

		# update with MSE loss
		optimizer.zero_grad()
		optimizer_embed.zero_grad()
		
		loss.backward()

		# compute grad norm
		grad_norm_embed = torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), 100)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(policy.parameters(), 100)
		
		optimizer.step()
		optimizer_embed.step()

		
		if it % 10 == 0:
			# print('grad norm embed: ', grad_norm_embed)
			# print('grad norm policy: ', grad_norm_policy)
			L.log({'loss':loss.item(), 'total_time':time.time() - time_start}, \
					step=it, category="train")


		# if num_iterations % 100 == 0:
		# 	feature_map = embedding_model.extract_feature_map(obs_mini[:, :3])
		# 	save_feature_map(feature_map[0], f'imgs/bc_{num_iterations}.png')


		# evaluation
		if it % 1000 == 0: 
			success_rate, rewards = evaluate(env, embedding_model, policy, L, it, cfg)

			max3_success_rates.append(success_rate)
			max3_returns.append(rewards)
			max3_success_rates.sort(reverse=True)
			max3_returns.sort(reverse=True)

			if len(max3_success_rates) > 3:
				max3_success_rates = max3_success_rates[:3]
				max3_returns = max3_returns[:3]

			success_rate_pvr = np.mean(max3_success_rates)
			rewards_pvr = np.mean(max3_returns)

			L.log({'success_rate':success_rate, 'reward':np.mean(rewards), \
					'success_rate_pvr':success_rate_pvr, 'reward_pvr':rewards_pvr, \
					'total_time':time.time() - time_start}, step=it, category="eval")

				

if __name__ == '__main__':
	cfg = parse_cfg(cfg_path="configs", mode="bc")
	main(cfg)


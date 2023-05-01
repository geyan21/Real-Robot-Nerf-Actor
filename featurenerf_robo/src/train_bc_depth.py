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
from algorithms.modules import ContinuousPolicy, RewardPredictor
from termcolor import colored
from logger import Logger
import time
import itertools
import matplotlib.pyplot as plt
from analysis.image import save_feature_map, save_rgb_image
from mj_pc.mj_point_clouds import PointCloudGenerator
import visdom


def process_raw_point_cloud(point_cloud, num_points):
	"""
	process raw point cloud from env into torch tensor
	"""
	# downsample
	point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], num_points, replace=False), :]
	point_cloud = torch.from_numpy(point_cloud).float().cuda().unsqueeze(0)
	# permute in [B, C, N]
	point_cloud = point_cloud.permute(0, 2, 1)
	return point_cloud

def process_raw_state(raw_state):
	"""
	process raw state from env into torch tensor
	"""
	state = torch.from_numpy(raw_state).float().cuda()
	return state


def evaluate(env, pc_generator, embedding_model, policy, L, step, cfg):
	num_episodes = 20
	success_rate = []
	returns = []

	# viz = visdom.Visdom()

	for i in range(num_episodes):
		obs, state, info = env.reset()
		point_cloud = pc_generator.generateCroppedPointCloud()

		done = False
		episode_reward = 0
		frames_static = []
		success = 0
		# frames_dynamic = []
		while not done:
			with torch.no_grad():
				embed = embedding_model(process_raw_point_cloud(point_cloud, cfg.num_points))
				if cfg.use_robot_state:
					mu = policy(embed, process_raw_state(state).unsqueeze(0))
				else:
					mu = policy(embed)

			obs, state, reward, done, info = env.step(mu.squeeze(0))
			point_cloud = pc_generator.generateCroppedPointCloud()
			episode_reward += reward
			success = success or info['is_success']
			
			if i==0:
				img_obs = env.render_obs(width=128, height=128)
				frames_static.append(img_obs[0])
				# viz.scatter(point_cloud, win='point_cloud', opts=dict(title='point_cloud', markersize=2))


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



def main(cfg):

	# prepare dataset and dataloder
	demonstration_data_dir = os.path.join(cfg.dataset_root, "{}_{}".format(cfg.domain_name, cfg.task_name))
	image_size = cfg.image_size
	num_cameras = ("dynamic" in cfg.camera_mode) + ("static" in cfg.camera_mode) + cfg.num_static_cameras + cfg.num_dynamic_cameras - 2
	bc_dataset = utils.BehaviorCloneDataset(cfg=cfg, root_dir=demonstration_data_dir, episode_length=cfg.episode_length, \
									image_size=image_size, num_trajs=cfg.num_trajs, num_cameras=num_cameras,
									use_3d=True)
	bc_dataset[0]
	data_loader = DataLoader(dataset=bc_dataset, 
							batch_size=cfg.batch_size, 
							collate_fn=bc_dataset.collect_fn, shuffle=True, num_workers=4, drop_last=False)
	

	# prepare agent dimensions
	action_dim_dict = {"xy":(2,), "xyz":(3,), "xyzw":(4,)}
	state_dim_dict = {"xy":(3,), "xyz":(4,), "xyzw":(4,)}
	action_dim = action_dim_dict[cfg.action_space]
	# observation_dim = (num_cameras*3, image_size, image_size) # default hard-coded value
	observation_dim = (3, cfg.num_points) # 3D point cloud input
	state_dim = state_dim_dict[cfg.action_space]
	

	# iterate over seeds
	for s in range(cfg.num_seeds):

		# Set seed
		utils.set_seed_everywhere(cfg.seed + 42 + s)


		# prepare embedding model
		assert torch.cuda.is_available(), 'must have cuda enabled'
		print('[training] Observations:', observation_dim)
		print('[training] Actions:', action_dim)

		embedding_model = make_embedding(model_name=cfg.embedding_name, cfg=cfg, use_3D=True)
		embedding_model = embedding_model.cuda()
		if cfg.freeze_encoder:
			embedding_model.eval()
		else:
			embedding_model.train()

		# prepare policy model
		policy = ContinuousPolicy(input_dim=embedding_model.output_dim, action_dim=action_dim[0], state_dim=state_dim[0])
		policy = policy.cuda()


		# prepare optimizer and loss function
		optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
		optimizer_embed = torch.optim.Adam(embedding_model.parameters(), lr=cfg.lr)
		loss_fn = torch.nn.MSELoss()

		# prepare reward predictor
		if cfg.use_reward_predictor:
			reward_predictor = RewardPredictor(input_dim=embedding_model.output_dim).cuda()
			optimizer_reward = torch.optim.Adam( itertools.chain(embedding_model.parameters(), reward_predictor.parameters()), lr=cfg.lr)


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
			observation_type="state+state", # we directly get pointcloud so we don't need to render img
			action_space=cfg.action_space,
			camera_move_range=cfg.camera_move_range,
			domain_randomization=cfg.domain_randomization,
			num_static_cameras=cfg.num_static_cameras,
			num_dynamic_cameras=cfg.num_dynamic_cameras,
		)
		pc_generator = PointCloudGenerator(env.sim, min_bound=(-1., -1., -1.), max_bound=(1., 1., 1.))

		env.seed(cfg.seed)
		env.observation_space.seed(cfg.seed)
		env.action_space.seed(cfg.seed)

		# logger
		log_dir = f"{cfg.log_dir_root}/{cfg.log_dir}"
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		L = Logger(log_dir, mode="bc", config=cfg)

		num_iterations = 0
		time_start = time.time()

		max3_success_rates = []
		max3_returns = []

		# iterate over epochs
		for epoch in range(cfg.num_epochs):
    		# iterate over batches of trajectories
			for batch_idx, batch in enumerate(data_loader):
				# load data, consist of batch * traj * single transition
				obs = batch['point_cloud']
				state = batch['state']
				action = batch['action']
				reward = batch['reward']
				next_obs = batch['next_point_cloud']
				next_state = batch['next_state']
				done = batch['done']
				info = batch['info']




				# we use transitions instead of trajectories, thus sample one batch
				# first squeeze the dim
				obs = obs.reshape(obs.shape[0]*obs.shape[1], *obs.shape[2:])
				state = state.reshape(state.shape[0]*state.shape[1], *state.shape[2:])
				action = action.reshape(action.shape[0]*action.shape[1], *action.shape[2:])
				reward = reward.reshape(reward.shape[0]*reward.shape[1], *reward.shape[2:])
				next_obs = next_obs.reshape(next_obs.shape[0]*next_obs.shape[1], *next_obs.shape[2:])
				next_state = next_state.reshape(next_state.shape[0]*next_state.shape[1], *next_state.shape[2:])
				done = done.reshape(done.shape[0]*done.shape[1], *done.shape[2:])
				info = info.reshape(info.shape[0]*info.shape[1], *info.shape[2:])

				# shuffle the data
				shuffle_idx = np.random.permutation(obs.shape[0])
				obs = obs[shuffle_idx]
				state = state[shuffle_idx]
				action = action[shuffle_idx]
				reward = reward[shuffle_idx]
				next_obs = next_obs[shuffle_idx]
				next_state = next_state[shuffle_idx]
				done = done[shuffle_idx]
				info = info[shuffle_idx]


				# iterate over all transitions with batch size
				for minibatch_idx in range(0, obs.shape[0], cfg.batch_size):
					num_iterations += 1
					
					obs_mini, state_mini, action_mini, reward_mini, next_obs_mini, next_state_mini, done_mini, info_mini = \
						obs[minibatch_idx:minibatch_idx+cfg.batch_size], \
						state[minibatch_idx:minibatch_idx+cfg.batch_size], \
						action[minibatch_idx:minibatch_idx+cfg.batch_size], \
						reward[minibatch_idx:minibatch_idx+cfg.batch_size], \
						next_obs[minibatch_idx:minibatch_idx+cfg.batch_size], \
						next_state[minibatch_idx:minibatch_idx+cfg.batch_size], \
						done[minibatch_idx:minibatch_idx+cfg.batch_size], \
						info[minibatch_idx:minibatch_idx+cfg.batch_size]

					# convert [0,1] np array to torch tensor
					obs_mini = torch.from_numpy(obs_mini).float().cuda()
					action_mini = torch.from_numpy(action_mini).float().cuda()
					state_mini = torch.from_numpy(state_mini).float().cuda()

					# get embed
					embed = embedding_model(obs_mini[:, :3])

					# predict action
					if cfg.use_robot_state:
						mu  = policy(embed, state_mini)
					else:
						mu = policy(embed)

					loss = loss_fn(mu, action_mini)
					if cfg.use_reward_predictor:
    					# predict reward
						r_pred = reward_predictor(embed)
						r_target = torch.from_numpy(reward_mini).float().cuda()
						loss_r = loss_fn(r_pred, r_target)
						loss += 0.1*loss_r

					# update with MSE loss
					optimizer.zero_grad()
					optimizer_embed.zero_grad()
					if cfg.use_reward_predictor:
						optimizer_reward.zero_grad()
					
					loss.backward()

					# compute  grad norm
					grad_norm_embed = torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), 100)
					grad_norm_policy = torch.nn.utils.clip_grad_norm_(policy.parameters(), 100)
					
					optimizer.step()
					optimizer_embed.step()
					if cfg.use_reward_predictor:
						optimizer_reward.step()

					

					if num_iterations % 10 == 0:
						# print('grad norm embed: ', grad_norm_embed)
						# print('grad norm policy: ', grad_norm_policy)
						L.log({'loss':loss.item(), 'total_time':time.time() - time_start, 'epoch':epoch}, \
								step=num_iterations, category="train")

					# if num_iterations % 100 == 0:
					# 	feature_map = embedding_model.extract_feature_map(obs_mini[:, :3])
					# 	save_feature_map(feature_map[0], f'imgs/bc_{num_iterations}.png')
			
			eval_freq = 10
			if epoch % eval_freq == 0:
				# evaluate for each eval_freq epoch
				embedding_model.eval()
				success_rate, rewards = evaluate(env, pc_generator,  embedding_model, policy, L, num_iterations, cfg)
				if cfg.freeze_encoder:
					embedding_model.train() # resume training

				max3_success_rates.append(success_rate)
				max3_returns.append(rewards)
				max3_success_rates.sort(reverse=True)
				max3_returns.sort(reverse=True)

				if len(max3_success_rates) > 3:
					max3_success_rates = max3_success_rates[:3]
					max3_returns = max3_returns[:3]

				success_rate_pvr = np.mean(max3_success_rates)
				rewards_pvr = np.mean(max3_returns)
				
				L.log({'success_rate':success_rate, 'reward':np.mean(rewards), 'epoch':epoch, \
						'success_rate_pvr':success_rate_pvr, 'reward_pvr':rewards_pvr, \
						'total_time':time.time() - time_start}, step=num_iterations, category="eval")

					
    						

		
		


if __name__ == '__main__':
	cfg = parse_cfg(cfg_path="configs", mode="bc")
	main(cfg)


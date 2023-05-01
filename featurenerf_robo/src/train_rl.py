from representations import make_embedding
import os
import numpy as np
import gym
import utils
import time
from config import parse_cfg
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import torch
from algorithms.per import EfficientPrioritizedReplayBuffer
import sys
import warnings
warnings.filterwarnings("ignore")
try:
	import wandb
except:
	print('Wandb is not installed in your env. Skip `import wandb`.')
	pass


def evaluate(env, agent, num_episodes, L, step, test_env=False, cfg=None):
	episode_rewards = []
	success_rate = []
	_test_env = '_test_env' if test_env else ''
	for i in range(num_episodes):
		obs, state, info = env.reset()
		if cfg.use_gt_camera:
			agent.save_camera_intrinsic(info["camera_intrinsic"]) # save intrinsic, for repeated usage

		done = False
		episode_reward = 0
		frames_static = []
		# frames_dynamic = []
		while not done:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.select_action(obs, state)
			obs, state, reward, done, info = env.step(action)
			if i==0:
				img_obs = env.render_obs(width=128, height=128)
				frames_static.append(img_obs[0])
				# frames_dynamic.append(img_obs[1])
			episode_reward += reward
		if 'is_success' in info:
			success = float(info['is_success'])
			success_rate.append(success)


		if cfg.use_wandb and i==0 and step%cfg.log_train_video==0:
			L.log_video(frames_static, 'eval_static', step=step, category='eval') # log video to wandb
			# L.log_video(frames_dynamic, 'eval_dynamic', step=step, category='eval') # log video to wandb


		episode_rewards.append(episode_reward)

	episode_rewards = np.nanmean(episode_rewards)
	success_rate = np.nanmean(success_rate)

	L.log({f'episode_reward{_test_env}':episode_reward, \
			f'success_rate{_test_env}': success_rate}, \
			step=step, category='eval') 

		
	return episode_rewards, success_rate


def main(cfg):
	# Set seed
	utils.set_seed_everywhere(cfg.seed+42)
	
	# create embedding
	if cfg.embedding_name != "none":
		embedding_model = make_embedding(model_name=cfg.embedding_name, cfg=cfg)
		embedding_model.eval()
		# freeze embedding
		for param in embedding_model.parameters():
			param.requires_grad = False
	else:
		embedding_model = None

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=cfg.domain_name,
		task_name=cfg.task_name,
		seed=cfg.seed+42,
		episode_length=cfg.episode_length,
		n_substeps=cfg.n_substeps,
		frame_stack=cfg.frame_stack,
		image_size=cfg.image_size,
		# cameras="static+dynamic",
		cameras="static",
		render=cfg.render, # Only render if observation type is state
		observation_type=cfg.observation_type, # state, image, state+image
		action_space=cfg.action_space,
		camera_move_range=cfg.camera_move_range,
		domain_randomization=cfg.domain_randomization,
		embedding_model=embedding_model,
		cfg=cfg,
	)
	env.seed(cfg.seed)
	env.observation_space.seed(cfg.seed)
	env.action_space.seed(cfg.seed)

	# Create working directory
	work_dir = os.path.join(cfg.log_dir_root, cfg.log_dir, str(cfg.seed))
	print('Working directory:', work_dir)
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))

	# create replay buffer
	replay_buffer = EfficientPrioritizedReplayBuffer(
		obs_shape=env.observation_space.shape,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		capacity=cfg.buffer_capacity,
		batch_size=cfg.batch_size,
		prioritized_replay=cfg.use_prioritized_buffer,
		alpha=cfg.prioritized_replay_alpha,
		beta=cfg.prioritized_replay_beta,
		ensemble_size=cfg.ensemble_size,
		episode_length=cfg.episode_length,
		observation_type=cfg.observation_type,
		use_gt_camera=cfg.use_gt_camera,
		imagenet_normalization=cfg.imagenet_normalization,
	)
	
	print("Observation type:", cfg.observation_type)
	print('Observations:', env.observation_space.shape)
	print('Action space:', f'({env.action_space.shape[0]})')


	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	agent = make_agent(
		obs_shape=env.observation_space.shape,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		cfg=cfg
	)

	start_step, episode, episode_reward, info, done, episode_success = 0, 0, 0, {}, True, 0
	L = Logger(work_dir, mode='rl', config=cfg)


	start_time = time.time()
	for step in range(start_step, cfg.train_steps+1):
		if done:
			# Evaluate agent periodically
			if step % cfg.eval_freq == 0:
				print('Evaluating:', work_dir)

				evaluate(env, agent, cfg.eval_episodes, L, step, cfg=cfg)
    						

			# Save agent periodically
			save_model = False
			if save_model:
				if step!=0 and step % 100000==0 or step == cfg.train_steps:
					torch.save(agent, os.path.join(model_dir, f'{step}.pt'))
					if cfg.use_wandb and cfg.save_model:
							wandb.save(os.path.join(model_dir, f'{step}.pt'))

			L.log({'episode_reward': episode_reward,\
					'success_rate':episode_success/cfg.episode_length,
					'episode':episode, 
					'total_time':time.time()-start_time}, 
					step=step, category='train') 


			obs, state, info = env.reset()
			if cfg.use_gt_camera:
				agent.save_camera_intrinsic(info["camera_intrinsic"]) # save intrinsic, for repeated usage
			cur_camera = info["camera_RT"] if cfg.use_gt_camera else None
			done = False

			episode_reward = 0
			episode_step = 0
			episode += 1
			episode_success = 0


		# Sample action and update agent
		if step < cfg.init_steps:
			action = env.action_space.sample()
		else:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.sample_action(obs, state)
			
			num_updates = cfg.init_steps//cfg.update_freq if step == cfg.init_steps else 1
			for i in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, next_state, reward, done, info = env.step(action)
		next_camera = info["camera_RT"] if cfg.use_gt_camera else None

		replay_buffer.add(obs, state, action, reward, next_obs, next_state, cur_camera)
		episode_reward += reward
		obs = next_obs
		state = next_state
		cur_camera = next_camera

		episode_success += float(info['is_success'])
		episode_step += 1
	print('Completed training for', work_dir)

	# save the model
	L.save_model(agent, step) # save the model at the end of training

	# compute training time
	print("Total Training Time: ", round((time.time() - start_time) / 3600, 2), "hrs")


if __name__ == '__main__':
	cfg = parse_cfg(cfg_path="configs", mode="rl")
	
	# parse some integers
	cfg.log_train_video = int(cfg.log_train_video.replace('k','000').replace('m','000000')) # convert to int
	cfg.buffer_capacity = int(cfg.buffer_capacity.replace('k','000').replace('m','000000')) # convert to int
	cfg.train_steps = int(cfg.train_steps.replace('k','000').replace('m','000000')) # convert to int
	cfg.eval_freq = int(cfg.eval_freq.replace('k','000').replace('m','000000')) # convert to int

	if isinstance(cfg.seed, int): # one seed
		main(cfg)
	else: # multiple seeds
		utils.parallel(main, cfg)
	

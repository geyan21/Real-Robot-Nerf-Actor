
"""
use a random policy to collect data
"""
import torch
import torchvision
import os
import glob
import shutil
import numpy as np
import gym
import utils
from tqdm import tqdm
from config import parse_cfg
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder


def render_and_store(env, cfg):
	image_size = cfg.image_size
	STORE_PATH=f"nerf_data/{cfg.domain_name}_{cfg.task_name}_{image_size}"
	if not os.path.exists(STORE_PATH):
		os.makedirs(STORE_PATH)

	nominate_function_image = lambda x: "{0:03}.png".format(x)
	nominate_function_camera = lambda x: "{0:03}.txt".format(x)

	successes = 0
	
	obs, state, _ = env.reset()
	for scene_id in tqdm(range(cfg.num_scenes), desc="scenes"):
		# each timestep, the env is different and is thus stored in a different folder
		
		random_action = env.action_space.sample()
		env.step(random_action)

		# as a different scene for nerf
		img_path = os.path.join(STORE_PATH, str(scene_id), "images")
		camera_path = os.path.join(STORE_PATH, str(scene_id), "cameras")
		if not os.path.exists(img_path):
			os.makedirs(img_path)
		if not os.path.exists(camera_path):
			os.makedirs(camera_path)

		# store the images of all angles
		# loop over different positions
		for camera_idx in range(env.traj_len):
			# change camera position
			env.change_traj_idx(camera_idx)
			env.randomize_camera()

			# save the image from dynamic camera
			img = env.render_obs(width=image_size, height=image_size, mode="rgb_array")[0].transpose(2, 0, 1)
			img = torch.from_numpy(img).float().div(255.0)
			torchvision.utils.save_image(img, os.path.join(img_path, nominate_function_image(camera_idx)) )
			extrinsic_dynamic = env.get_camera_extrinsic()[0] # 4x4
			# env.get_camera_pos_and_euler("camera_dynamic")
			intrinsic_dynamic = env.get_camera_intrinsic() # 3x3
			focal = env.get_focal_length(image_size=image_size)
			# save camera param in txt
			with open( os.path.join(camera_path, nominate_function_camera(camera_idx)), 'w') as file_to_save:
				file_to_save.write("extrinsic\n")
				np.savetxt(file_to_save, extrinsic_dynamic)
				file_to_save.write("intrinsic\n")
				np.savetxt(file_to_save, intrinsic_dynamic)
				file_to_save.write("focal\n")
				np.savetxt(file_to_save, np.array([focal]))
				file_to_save.write("image_size\n")
				np.savetxt(file_to_save, np.array([image_size, image_size]))

		



def main(cfg):

	num_tasks = len(cfg.task_name)
	seed = 42
	num_trajs = cfg.num_trajs # number of trajectories we want to generate

	
	# Set seed
	utils.set_seed_everywhere(seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=cfg.domain_name,
		task_name=cfg.task_name,
		seed=seed,
		episode_length=cfg.episode_length,
		action_space=cfg.action_space,
		n_substeps=cfg.n_substeps,
		frame_stack=cfg.frame_stack,
		image_size=cfg.image_size,
		cameras="dynamic", #['third_person', 'first_person']
		render=cfg.render, # Only render if observation type is state
		observation_type=cfg.observation_type,
		camera_move_range=75, # full range
		domain_randomization=cfg.domain_randomization,
	)


	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	print('Observations:', env.observation_space.shape)
	print('Actions:', env.action_space.shape)
	print("task name:", cfg.task_name)

	agent = None
	render_and_store(env, cfg)



if __name__ == '__main__':
	cfg = parse_cfg(cfg_path="configs", mode="bc")
	main(cfg)
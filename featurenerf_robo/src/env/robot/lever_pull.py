import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path
from .reward_utils import tolerance, hamacher_product

class LeverPullEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		self.lever_init_pos = None
		self.lever_start_init_pos = None

		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz,
		)
		self.state_dim = (26,) if self.use_xyz else (20,)
		self.init_tcp = self.sim.data.get_site_xpos('grasp').copy()
		self.target_pos = self.sim.data.get_site_xpos('goal').copy()
		# self.lever_init_pos = self.sim.data.get_body_xpos("lever").copy()

		utils.EzPickle.__init__(self)

	def compute_reward_leverpull_v2(self, achieved_goal, goal):
		
		gripper =  self.sim.data.get_site_xpos('grasp').copy()
		# lever = self.sim.data.get_body_xpos('lever').copy()
		lever = self.sim.data.get_site_xpos('leverStart').copy()

		# De-emphasize y error so that we get Sawyer's shoulder underneath the
		# lever prior to bumping on against
		scale = np.array([4., 1., 4.])
		# Offset so that we get the Sawyer's shoulder underneath the lever,
		# rather than its fingers
		offset = np.array([.0, .055, .07])

		shoulder_to_lever = (gripper + offset - lever) * scale
		shoulder_to_lever_init = (
			self.init_tcp + offset - self.lever_start_init_pos
		) * scale
		
		# This `ready_to_lift` reward should be a *hint* for the agent, not an
		# end in itself. Make sure to devalue it compared to the value of
		# actually lifting the lever
		ready_to_lift = tolerance(
			np.linalg.norm(shoulder_to_lever),
			bounds=(0, 0.02),
			margin=np.linalg.norm(shoulder_to_lever_init),
			sigmoid='long_tail',
		)

        # The skill of the agent should be measured by its ability to get the
		# lever to point straight upward. This means we'll be measuring the
		# current angle of the lever's joint, and comparing with 90deg.
		lever_angle = -self.sim.data.get_joint_qpos('LeverAxis')
		lever_angle_desired = np.pi / 2.0

		lever_error = abs(lever_angle - lever_angle_desired)

		# We'll set the margin to 15deg from horizontal. Angles below that will
		# receive some reward to incentivize exploration, but we don't want to
		# reward accidents too much. Past 15deg is probably intentional movement
		lever_engagement = tolerance(
			lever_error,
			bounds=(0, np.pi / 48.0),
			margin=(np.pi / 2.0) - (np.pi / 12.0),
			sigmoid='long_tail'
		)

		target = self.target_pos
		obj_to_target = np.linalg.norm(lever - target)
		in_place_margin = (np.linalg.norm(self.lever_start_init_pos - target))

		in_place = tolerance(obj_to_target,
									bounds=(0, 0.04),
									margin=in_place_margin,
									sigmoid='long_tail',)

		# reward = 2.0 * ready_to_lift + 8.0 * lever_engagement
		reward = 10.0 * hamacher_product(ready_to_lift, in_place)
		
		return reward


	def compute_reward(self, achieved_goal, goal, info):

		return self.compute_reward_leverpull_v2(achieved_goal, goal)

	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

		obj_pos = self.sim.data.get_site_xpos('goal') #- cot_pos
		obj_rot = self.sim.data.get_joint_qpos('nail_board:joint')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('nail_target'+ str(self.nail_id)) * dt
		obj_velr = self.sim.data.get_site_xvelr('nail_target'+ str(self.nail_id)) * dt

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]
			obj_pos = obj_pos[:2]
			obj_velp = obj_velp[:2]
			obj_velr = obj_velr[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			self.goal_distance(obj_pos, goal_pos, self.use_xyz),
			self.goal_distance(eef_pos, obj_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, obj_pos, obj_rot, obj_velp, obj_velr, values
		], axis=0)

	def _reset_sim(self):

		return BaseEnv._reset_sim(self)

	def _is_success(self, achieved_goal, desired_goal):
		d = self.goal_distance(achieved_goal, desired_goal, self.use_xyz)
		return (d < 0.01).astype(np.float32)

	def _get_achieved_goal(self):
		return np.squeeze(self.sim.data.get_site_xpos('leverStart').copy())

	def _sample_object_pos(self):
		return None


	def _sample_goal(self, new=True):

		# Randomly sample the position of the lever
		lever_pos = self.sim.data.get_body_xpos("lever")
		
		if self.lever_init_pos is None:
			self.lever_init_pos = lever_pos.copy()
			self.lever_start_init_pos = self.sim.data.get_site_xpos('leverStart').copy()
			
		else:
			lever_pos[0] = self.lever_init_pos[0]
			lever_pos[1] = self.lever_init_pos[1]
			lever_pos[2] = self.lever_init_pos[2]

		if new:
			lever_pos[0] += 0.1*self.np_random.uniform(-0.05, 0.05, size=1)
			lever_pos[1] += 0.1*self.np_random.uniform(-0.1, 0.1, size=1)
			# no need to set, just directly change
		
			
		goal_on_lever = self.sim.data.get_site_xpos('goal')
		goal = goal_on_lever.copy()

		
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		# sample the init pos of gripper
		gripper_target = np.array([1.2561169, 0.3, 0.69603332])
		gripper_target[0] += self.np_random.uniform(-0.05, 0.1, size=1)
		gripper_target[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		if self.use_xyz:
			gripper_target[2] += self.np_random.uniform(-0.05, 0.1, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)

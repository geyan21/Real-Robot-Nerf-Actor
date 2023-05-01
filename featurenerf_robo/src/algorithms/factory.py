from algorithms.sacv2 import SACv2

algorithm = {
	'sacv2': SACv2,
}


def make_agent(obs_shape, state_shape, action_shape, cfg):
	return algorithm[cfg.algorithm](obs_shape, state_shape, action_shape, cfg)

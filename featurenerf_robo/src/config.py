# reference: https://github.com/nicklashansen/tdmpc
import os
import os.path as osp
import re
from omegaconf import OmegaConf


def parse_cfg(cfg_path: str = "configs", mode:str = None) -> OmegaConf:
	"""
    Parses a config file and returns an OmegaConf object.
	Hierarchical file loading.
	mode: rl, bc, distill, nerf
    """
    # read base file and command line arguments
	config_base = OmegaConf.load( osp.join(cfg_path, 'default.yaml'))
	config_cli = OmegaConf.from_cli()
	for k,v in config_cli.items():
		if v == None:
			config_cli[k] = True

	# specify mode in code directly and read default config
	if mode not in ['rl', 'bc', 'distill', 'nerf']:
		raise ValueError(f'Invalid mode: {mode}. Supported modes are: rl, bc, distill, nerf')
	config_mode = OmegaConf.load( osp.join(cfg_path, mode, 'default.yaml'))

	# specify the detailed algorithm config in CLI and read config
	if config_cli.alg_config == None:
		raise ValueError('Please specify file name of alg_config in CLI')
	config_algo = OmegaConf.load( osp.join(cfg_path, mode, config_cli.alg_config + '.yaml'))
	config_mode.merge_with(config_algo)

	# merge base and mode
	config_base.merge_with(config_mode)
	# overwrite base and mode with CLI
	config_base.merge_with(config_cli) 

	return config_base

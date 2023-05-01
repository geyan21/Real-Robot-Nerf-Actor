from collections import defaultdict
import json
import os
import torch
import datetime
from termcolor import colored
import numpy as np
import pandas as pd
import torchvision
from PIL import Image

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), ('step', 'S', 'int'), ('episode_reward', 'R', 'float'),
            ('success_rate', 'SR', 'float'),
            #  ('actor_loss', 'ALOSS', 'float'), ('critic_loss', 'CLOSS', 'float'), ('3d_loss', '3DLOSS', 'float'),
            ('total_time', 'T', 'time')

        ],
        'eval': [
           ('episode', 'E', 'int'), ('step', 'S', 'int'), ('episode_reward', 'ER', 'float'),
            ('episode_reward_test_env', 'ERTEST', 'float'),
            ('total_time', 'T', 'time')
        ]
    },
    
    'distill': {
        'train': [
            ('step', 'S', 'int'),
            ('render_loss', 'Render Loss', 'float'), 
            ('distill_loss', 'Distill Loss', 'float'),
            ('psnr', 'PSNR', 'float'),
            ('total_time', 'T', 'time')
        ],
        'eval': [
            ('step', 'S', 'int'), 
            ('render_loss', 'Render Loss', 'float'), 
            ('psnr', 'PSNR', 'float'),
            ('num_eval_imgs', 'Num of Eval', 'int'),
            ('total_time', 'T', 'time'),
        ]
    },

    'nerf': {
        'train': [
            ('step', 'S', 'int'),
            ('render_loss', 'Render Loss', 'float'), 
            ('psnr', 'PSNR', 'float'),
            ('grad', 'Grad', 'float'),
            ('total_time', 'T', 'time')
        ],
        'eval': [
            ('step', 'S', 'int'),
            ('render_loss', 'Render Loss', 'float'), 
            ('psnr', 'PSNR', 'float'),
            ('num_eval_imgs', 'Num of Eval', 'int'),
            ('total_time', 'T', 'time')
        ]
    },

    'bc':{
        'train': [
            # ('epoch', 'E', 'int'),
            ('step', 'S', 'int'),
            ('loss', 'Loss', 'float'),
            ('total_time', 'T', 'time')
        ],
        'eval': [
            # ('epoch', 'E', 'int'),
            ('step', 'S', 'int'),
           ('success_rate', 'Success', 'float'),
           ('reward', 'Reward', 'float'),
           ('total_time', 'T', 'time')
        ]
    }
}

CONFIG_TO_PRINT = {
    'nerf':  ['nerf_model', 'scene_batch_size', 'ray_batch_size', 'num_scenes_train', 'num_scenes_eval'],

    'distill': ['distill_model', 'scene_batch_size', 'ray_batch_size', 'num_scenes_train', 'num_scenes_eval'],

    'rl': [ 'embedding_name', 'domain_name', 'image_size', 'task_name', 'train_steps'],

    'bc': ['domain_name', 'task_name', 'embedding_name', 'num_trajs', 'freeze_encoder', 'use_robot_state', 'use_reward_predictor', 'batch_size', 'num_epochs','save_video']
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


def print_config(cfg, mode):
    """Pretty-printing of run information. Call at start of training."""
    prefix, color, attrs = '  ', 'green', ['bold']
    def limstr(s, maxlen=32):
        return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
    def pprint(k, v):
        print(prefix + colored(f'{k.capitalize()+":":<20}', color, attrs=attrs), limstr(v))
    kvs = [('log dir root', cfg.log_dir_root),
            ('log dir', cfg.log_dir),
            ('save video', cfg.save_video),
            ('save model', cfg.save_model),
    ]
    kvs += [(k, v) for k, v in cfg.items() if k in CONFIG_TO_PRINT.get(mode, [])]
    w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
    div = '-'*w
    print(div)
    for k,v in kvs:
        pprint(k, v)
    print(div)



class Logger(object):
    def __init__(self, log_dir, mode='rl', config=None):
        self._log_dir = make_dir(log_dir)
        self.config = config
        self._eval = []
        self.print_format = FORMAT_CONFIG[mode]
        print_config(config, mode)

        # wandb
        project, entity, group = config.wandb_project, config.wandb_entity, config.wandb_group
        run_offline = not config.use_wandb or project == 'none' or entity == 'none' or group == 'none'
        if run_offline:
            print(colored('Logs will be saved locally.', 'yellow', attrs=['bold']))
            self._wandb = None
        else:
            try:
                os.environ["WANDB_SILENT"] = "true"
                import wandb
                wandb.init(project=project,
                        entity=entity,
                        name=str(config.seed),
                        group=group,
                        dir=self._log_dir,
                        config=config)
                print(colored('Logs will be synced with wandb.', 'blue', attrs=['bold']))
                self._wandb = wandb
            except:
                print(colored('Warning: failed to init wandb. Logs will be saved locally.', 'yellow', attrs=['bold']))
                self._wandb = None
        
        # saving things
        self._save_model = config.save_model
        self._save_video = config.save_video
        if self._save_video:
            self._video_dir = make_dir(os.path.join(self._log_dir, 'videos'))
        if self._save_model:
            self._model_dir = make_dir(os.path.join(self._log_dir, 'models'))


    def _format(self, key, value, ty):
        if ty == 'int':
            return f'{colored(key+":", "grey")} {int(value):,}'
        elif ty == 'float':
            return f'{colored(key+":", "grey")} {float(value):.03f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "grey")} {value}'
        else:
            raise f'invalid log format type: {ty}'


    def _print(self, d, category):
        console_format = self.print_format[category]
        category = colored(category, 'blue' if category == 'train' else 'green')
        pieces = [f' {category:<14}']
        for k, disp_k, ty in console_format:
            pieces.append(f'{self._format(disp_k, d.get(k, 0), ty)}')
        print(' | '.join(pieces))
    

    def log(self, d:dict, step:int, category='train'):
        assert category in {'train', 'eval'}
        if self._wandb is not None:
            for k,v in d.items():
                self._wandb.log({category + '/' + k: v}, step=step)
        # if category == 'eval': # TODO: add the option to log eval
        #     keys = ['epoch', 'avg_return', 'avg_success', 'success_pvr', 'return_pvr']
        #     self._eval.append(np.array([d[keys[0]], d[keys[1]], d[keys[2]], d[keys[3]], d[keys[4]]]))
        #     pd.DataFrame(np.array(self._eval)).to_csv(self._log_dir / 'eval.log', header=keys, index=None)
        d['step'] = step
        self._print(d, category)


    def log_video(self, frame_list:list, video_name:str, step:int, category:str='train'):
        """
        each frame is: a tensor of shape (H, W, 3), from 0 to 255
        or a numpy array of shape (H, W, 3), from 0 to 255
        """
        if not self._save_video:
            return
        if isinstance(frame_list[0], np.ndarray):
            frame_list = [torch.from_numpy(frame.copy()) for frame in frame_list]
        if self._wandb is not None:
            frame_list = [frame_list[i].permute(2,0,1) for i in range(len(frame_list))] # (H, W, 3) -> (3, H, W)
            self._wandb.log({f"{category}/{video_name}": self._wandb.Video(torch.stack(frame_list).cpu(), fps=10)}, step=step)
        else:
            fp = os.path.join(self._video_dir, f"{step}_{video_name}.gif")
            # save by PIL
            frame_list = [Image.fromarray(frame.numpy().astype(np.uint8)) for frame in frame_list]
            frame_list[0].save(fp, save_all=True, append_images=frame_list[1:], duration=100, loop=0)
            # torchvision.io.write_video(fp, torch.stack(frame_list).cpu(), fps=10)

    
    def save_model(self, model, step):
        if not self._save_model:
            return
        fp = os.path.join(self._model_dir, f"model_{step}.pt")
        torch.save(model, fp)
        print(colored('Saved checkpoints at '+fp, 'cyan'))
        



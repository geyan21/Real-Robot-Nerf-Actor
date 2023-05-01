import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from termcolor import colored


class KeyframeBuffer(torch.utils.data.Dataset):
    """
    A Buffer for Behavior Cloning with Keyframe Discovery and Prediction
    """
    def __init__(self, traj_dataset):
        self.traj_dataset = traj_dataset
        self.keyframe_idx_buffer = []
        self.keyframe_stage_buffer = []


        # get keyframe idx
        for i in range(len(self.traj_dataset)):
            keyframe_idx, keyframe_stage = self._keyframe_discovery(self.traj_dataset[i])
            print(f"Find {len(keyframe_idx)} keyframes in episode {i}, which are {keyframe_idx}")
            self.keyframe_idx_buffer.append(keyframe_idx)
            self.keyframe_stage_buffer.append(keyframe_stage)


        # visualize keyframe for first episode
        example_traj = self.traj_dataset[0]
        example_obs = example_traj['obs']
        example_keyframe_idx = self.keyframe_idx_buffer[0]
        example_keyframe_stage = self.keyframe_stage_buffer[0]
        plt.figure(figsize=(10, 10))
        for i in range(len(example_keyframe_idx)):
            plt.subplot(1, len(example_keyframe_idx), i+1)
            example_img = example_obs[example_keyframe_idx[i]][:3].mul(255).numpy().transpose(1,2,0).astype(np.uint8)
            plt.imshow(example_img)
            plt.title(f"keyframe {example_keyframe_idx[i]}, stage {example_keyframe_stage[i]}")
        plt.tight_layout()
        plt.savefig('example_keyframe.png')
        print(colored(f"Example keyframe saved to example_keyframe.png", 'cyan'))

        # fill buffer
        self.fill_buffer_by_keyframe_prediction()
        print(colored(f"Finish filling buffer with {len(self.buffer)} data", 'cyan'))


    def fill_buffer_by_keyframe_prediction(self):
        """
        fill data buffer using keyframe prediction
        
        """
        self.buffer = []
        for traj_idx in range(len(self.traj_dataset)):
            keyframe_idx = self.keyframe_idx_buffer[traj_idx]
            traj = self.traj_dataset[traj_idx]


            # current: ['obs', 'depth', 'action', 'reward', 'done', 'state', 'full_state', 'info']
            key_next = ['next_obs', 'next_depth', 'next_full_state', 'next_state', 'next_info']
            for step in range(traj['obs'].shape[0]):
                single_data = {}

                # fill original data
                for key in traj.keys():
                    single_data[key] = traj[key][step]

                # find the corresponding keyframe
                keyframe_idx_corresponding = None
                for keyframe in keyframe_idx: # find the closest keyframe
                    if keyframe > step:
                        keyframe_idx_corresponding = keyframe
                        break
                if keyframe_idx_corresponding is None: # no keyframe found, meaning it is the last step
                    keyframe_idx_corresponding = keyframe_idx[-1]

                # replace next state info with keyframe info
                for key in key_next:
                    single_data[key] = traj[key][keyframe_idx_corresponding]

                # compute the cummulative translation action
                single_data['cummulative_translation'] = single_data['action'][:3]
                for i in range(step+1, keyframe_idx_corresponding+1): # no cur step and include keyframe
                    single_data['cummulative_translation'][:3] += traj['action'][i][:3]

                # finish and append
                self.buffer.append(single_data)
            
    
    def _keyframe_discovery(self, traj):
        """"
        key frame discovery for robot manipulation.

        metric:
        stage 0: gripper from close to open (sometimes not necessary)
        stage 1. gripper from open to close
        stage 2. from not success to success
        stage 3. from success to final state (since final state could have max reward)
        """
        episode_length = traj['obs'].shape[0]
        keyframe_idx = []
        keyframe_stage = []
        for step in range(episode_length):
            if step == episode_length - 1:
                keyframe_idx.append(step) # stage 3
                keyframe_stage.append(3)
                break

            info = traj['info'][step]
            next_info = traj['info'][step+1]
            if 'is_success' not in info.keys():
                info['is_success'] = False

            if not info['is_success'] and next_info['is_success']:
                keyframe_idx.append(step) # stage 2
                keyframe_stage.append(2)
            
            if not info['is_gripper_close'] and next_info['is_gripper_close']:
                keyframe_idx.append(step) # stage 1
                keyframe_stage.append(1)
            
            if info['is_gripper_close'] and not next_info['is_gripper_close']:
                keyframe_idx.append(step)   # stage 0
                keyframe_stage.append(0)

        return keyframe_idx, keyframe_stage
    

    def __getitem__(self, idx):
        return self.buffer[idx]
    

    def __len__(self):
        return len(self.buffer)
    

    def uniform_sample(self, batch_size):
        """
        uniform sample from buffer
        """
        idx = np.random.randint(0, len(self), batch_size)
        
        # init
        batch = {}
        for key in self.buffer[0].keys():
            batch[key] = []

        # fill
        for i in idx:
            for key in self.buffer[0].keys():
                batch[key].append(self.buffer[i][key])

        # stack these numpy arrays
        for key in self.buffer[0].keys():
            batch[key] = np.stack(batch[key])
        
        return batch
        


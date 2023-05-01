import torch
import torch.nn as nn
from .diffusion import Diffusion
from .model import MLP

class DiffusionPolicy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ):
        super(DiffusionPolicy, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)

    def forward(self, state):
        return self.actor(state)
    
    def loss(self, action, state):
        return self.actor.loss(action, state)
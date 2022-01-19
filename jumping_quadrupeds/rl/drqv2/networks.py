import torch
import torch.nn as nn
from jumping_quadrupeds.utils import TruncatedNormal
from jumping_quadrupeds.rl import utils

class Encoder(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        assert len(obs_space.shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_space, feature_dim, hidden_dim):
        super().__init__()
        self.action_space = action_space
        self.low = action_space.low[0]
        self.high = action_space.high[0]
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std, low=self.low, high=self.high)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_space, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_space.shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_space.shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

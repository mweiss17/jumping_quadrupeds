import os
import numpy as np
import matplotlib.pyplot as plt
import dill
import math
from itertools import cycle
from gym.spaces import Box

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from jumping_quadrupeds.utils import TruncatedNormal, weight_init
from torch.distributions.one_hot_categorical import OneHotCategorical


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

        self.apply(weight_init)

    def forward(self, obs):
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, n_actions, feature_dim, hidden_dim, log_std_init, use_ln):
        super().__init__()

        if use_ln:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        else:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.Tanh())

        if log_std_init:
            self.log_std = torch.nn.Parameter(log_std_init * torch.ones(n_actions, dtype=torch.float32))
        else:
            self.log_std = None

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

        self.apply(weight_init)

    def dist(self):
        raise NotImplementedError

    def forward(self, obs, std=None):
        h = self.trunk(obs)
        mu = self.policy(h)

        # If we want to learn the std, then we don't pass in a scheduled std
        if not std:
            std = torch.exp(self.log_std)
        std = torch.ones_like(mu) * std

        mu = torch.tanh(mu)
        return self.dist(mu, std)


class ContinuousActor(Actor):
    def __init__(self, action_space=None, **kwargs):
        kwargs["n_actions"] = action_space.shape[0]
        super().__init__(**kwargs)
        self.low = action_space.low[0]
        self.high = action_space.high[0]

    def dist(self, mu, std):
        return TruncatedNormal(mu, std, low=self.low, high=self.high)


class DiscreteActor(Actor):
    def __init__(self, action_space=None, **kwargs):
        kwargs["n_actions"] = action_space.n
        super().__init__(**kwargs)
        self.low = 0
        self.high = action_space.n - 1

    def dist(self, mu, std=None):
        return OneHotCategorical(mu)


class Critic(nn.Module):
    def __init__(self, repr_dim, action_space, feature_dim, hidden_dim, **kwargs):
        super().__init__()
        action_dim = action_space.shape[0] if isinstance(action_space, Box) else action_space.n
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

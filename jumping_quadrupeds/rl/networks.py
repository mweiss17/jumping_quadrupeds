import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
from jumping_quadrupeds.models.encoders import WorldModelsConvEncoder
from jumping_quadrupeds.rl.utils import mlp
from jumping_quadrupeds.utils import layer_init


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class CNNCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # [(W−K+2P)/S]+1
        self.logits_net = nn.Sequential(
            nn.Conv2d(obs_dim, 32, kernel_size=8, stride=4, padding=0),
            # [(84−8+0)/4]+1 = 20
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # [(20−4+0)/2]+1 = 9
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # [(9−3+0)]+1 = 7
            activation(),
            nn.Flatten(),
            nn.Linear(hidden_sizes * 64, act_dim),
        )

    def _distribution(self, obs):
        x = obs.float() / 255
        if len(x.shape) == 3:
            x.unsqueeze_(0)
        logits = self.logits_net(x)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class CNNGaussianActor(Actor):
    def __init__(self, encoder, act_dim, hidden_sizes, log_std):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(64 * hidden_sizes, act_dim)
        log_std = -log_std * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.action_min = torch.tensor([-1.0, 0.0, 0.0], requires_grad=False)
        self.action_max = torch.tensor([1.0, 1.0, 1.0], requires_grad=False)

    def _distribution(self, obs):
        preactivations = self.encoder(obs)
        mu = torch.tanh(self.linear(preactivations))
        mu = self.action_min + ((mu + 1) / 2) * (self.action_max - self.action_min)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class CNNCritic(nn.Module):
    def __init__(self, encoder, hidden_sizes):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(64 * hidden_sizes, 1)

    def forward(self, obs):
        # x = obs.float() / 255.0
        return torch.squeeze(
            self.linear(self.encoder(obs)), -1
        )  # Critical to ensure v has right shape.


class AbstractActorCritic(nn.Module):
    pi: nn.Module  # policy function/network
    v: nn.Module  # value function/network
    encoder: nn.Module  # state encoder function/network

    def step(self, obs):
        """take an observation, return the action, value, log probability of the action under the current policy"""
        pass

    def act(self, obs):
        """same as the `step()` function but only return the action"""
        return self.step(obs)[0]

    def save(self, folder, filename):
        """save the current policy and value functions to a folder."""
        # TODO
        pass

    def load(self, filepath):
        """restore saved policy and value func"""
        # TODO
        pass

    def get_policy_params(self):
        """return the parameters of all networks involved in the policy for the optimizer"""
        pass

    def get_value_params(self):
        """return the parameters of all networks involved in the value function for the optimizer"""
        pass


class ConvActorCritic(AbstractActorCritic):
    def __init__(
        self,
        observation_space,
        action_space,
        shared_encoder=False,
        hidden_sizes=16,
        log_std=0.5,
    ):
        super().__init__()

        channels = observation_space.shape[-1]

        actor_encoder = WorldModelsConvEncoder(channels)
        critic_encoder = (
            actor_encoder if shared_encoder else WorldModelsConvEncoder(channels)
        )
        self.pi = CNNGaussianActor(
            actor_encoder,
            action_space.shape[0],
            hidden_sizes,  # 4 * 4 square scaling factor for car-racing
            log_std,
        )

        # build value function
        self.v = CNNCritic(critic_encoder, hidden_sizes)

    def load_encoder(self, filepath):
        # Load the state encoder
        self.pi.encoder.load_state_dict(torch.load(filepath)["encoder"])

        # Load the state encoder
        self.v.encoder.load_state_dict(torch.load(filepath)["encoder"])

    def freeze_encoder(self):
        for param in self.pi.encoder.parameters():
            param.requires_grad = False
        for param in self.v.encoder.parameters():
            param.requires_grad = False

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy()[0], v.cpu().numpy(), logp_a.cpu().numpy()

    def get_policy_params(self):
        return self.pi.parameters()

    def get_value_params(self):
        return self.v.parameters()

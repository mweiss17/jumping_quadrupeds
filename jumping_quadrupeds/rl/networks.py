import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
from jumping_quadrupeds.encoders import WorldModelsConvEncoder
from jumping_quadrupeds.rl.utils import mlp


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
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
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
    def __init__(self, encoder, act_dim, hidden_sizes):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(64 * hidden_sizes, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, obs):
        if len(obs.shape) == 3:
            obs.unsqueeze_(0)
        preactivations = self.encoder(obs)
        mu = self.linear(torch.sigmoid(preactivations))
        # re-scale mu[0] which is [-1, 1] turn angle
        mu_rescaled = torch.ones_like(mu)
        mu_rescaled[:, 0] = 2.0
        mu_translate = torch.zeros_like(mu)
        mu_translate[:, 0] = 1.0
        mu = mu_rescaled * mu - mu_translate
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
        x = obs.float() / 255.0
        if len(x.shape) == 3:
            x.unsqueeze_(0)
        return torch.squeeze(
            self.linear(self.encoder(x)), -1
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
        self, observation_space, action_space, shared_encoder=False, hidden_sizes=16, activation=nn.Tanh
    ):
        super().__init__()

        channels = observation_space.shape[-1]

        actor_encoder = WorldModelsConvEncoder(channels=channels, activation=activation)
        critic_encoder = actor_encoder if shared_encoder else WorldModelsConvEncoder(channels=channels, activation=activation)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = CNNGaussianActor(
                actor_encoder,
                action_space.shape[0],
                hidden_sizes,  # 4 * 4 square scaling factor for car-racing
            )
        elif isinstance(action_space, Discrete):
            self.pi = CNNCategoricalActor(
                channels, action_space.n, hidden_sizes
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


class ConvSharedActorCritic(AbstractActorCritic):
    def __init__(
        self, observation_space, action_space, encoder_class, hidden_sizes=16, activation=nn.Tanh
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = CNNGaussianActor(
                observation_space.shape[-1],
                action_space.shape[0],
                hidden_sizes,  # 4 * 4 square scaling factor for car-racing
                activation,
            )
        elif isinstance(action_space, Discrete):
            self.pi = CNNCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )
        # build value function
        self.v = CNNCritic(observation_space.shape[-1], hidden_sizes, activation)

    def load_encoder(self, filepath):
        # Load the state encoder
        self.encoder.load_state_dict(torch.load(filepath))

    def freeze_encoder(self):
        for param in self.encoder.parameters():
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


class MLPSharedActorCritic(AbstractActorCritic):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]

        self.shared_encoder = mlp([obs_dim] + list(hidden_sizes), activation)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self._pi = MLPGaussianActor(
                hidden_sizes[-1], action_space.shape[0], [], activation
            )
        elif isinstance(action_space, Discrete):
            self._pi = MLPCategoricalActor(
                hidden_sizes[-1], action_space.n, [], activation
            )

        # build value function
        self._v = MLPCritic(hidden_sizes[-1], [], activation)

    def step(self, obs):
        with torch.no_grad():
            obs_encoded = self.shared_encoder(obs)

            pi = self._pi._distribution(obs_encoded)
            a = pi.sample()
            logp_a = self._pi._log_prob_from_distribution(pi, a)
            v = self._v(obs_encoded)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def pi(self, obs, act=None):
        obs_encoded = self.shared_encoder(obs)
        pi = self._pi._distribution(obs_encoded)
        logp_a = None
        if act is not None:
            logp_a = self._pi._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def v(self, obs):
        obs_encoded = self.shared_encoder(obs)
        return torch.squeeze(self._v.v_net(obs_encoded), -1)

    def get_policy_params(self):
        return list(self._pi.parameters()) + list(self.shared_encoder.parameters())

    def get_value_params(self):
        return list(self._v.parameters()) + list(self.shared_encoder.parameters())

    def train(self, mode: bool = True):
        self._pi.train(mode)
        self._v.train(mode)
        self.shared_encoder.train(mode)

    def eval(self):
        self._pi.eval()
        self._v.eval()
        self.shared_encoder.eval()

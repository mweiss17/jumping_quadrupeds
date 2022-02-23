import torch
from torch import nn

from jumping_quadrupeds.eth.encoders import WorldModelsConvEncoder
from jumping_quadrupeds.utils import TruncatedNormal


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


class AbstractActorCritic(nn.Module):
    pi: nn.Module  # policy function/network
    v: nn.Module  # value function/network
    encoder: nn.Module  # state encoder function/network

    def step(self, obs, eval_mode=False):
        """take an observation, return the action, value, log probability of the action under the current policy"""
        pass

    def act(self, obs, eval_mode=False):
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


class CNNGaussianActor(Actor):
    def __init__(self, encoder, action_space, hidden_sizes, log_std):
        super().__init__()
        self.encoder = encoder
        self.action_space = action_space
        self.low = action_space.low[0]
        self.high = action_space.high[0]
        self.linear = nn.Linear(64 * hidden_sizes, self.action_space.shape[0])
        self.log_std = torch.nn.Parameter(-log_std * torch.ones(self.action_space.shape[0], dtype=torch.float32))

    def _distribution(self, obs):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        preactivations = self.encoder(obs)
        mu = torch.tanh(self.linear(preactivations))
        std = torch.exp(self.log_std)
        return TruncatedNormal(mu, std, low=self.low, high=self.high)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class CNNCritic(nn.Module):
    def __init__(self, encoder, hidden_sizes):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(64 * hidden_sizes, 1)

    def forward(self, obs):
        return torch.squeeze(self.linear(self.encoder(obs)), -1)


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

        channels = observation_space.shape[0]

        actor_encoder = WorldModelsConvEncoder(channels)
        critic_encoder = actor_encoder if shared_encoder else WorldModelsConvEncoder(channels)
        self.pi = CNNGaussianActor(
            actor_encoder,
            action_space,
            hidden_sizes,  # 4 * 4 square scaling factor for base-racing
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

    def step(self, obs, eval_mode=False):
        with torch.no_grad():
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            pi = self.pi._distribution(obs)
            if eval_mode:
                a = pi.mean()
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return (
            a.squeeze(0).cpu().numpy(),
            v.cpu().numpy(),
            logp_a.squeeze(0).cpu().numpy(),
        )

    def get_policy_params(self):
        return self.pi.parameters()

    def get_value_params(self):
        return self.v.parameters()

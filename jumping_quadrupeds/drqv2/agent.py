# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from jumping_quadrupeds.augs import RandomShiftsAug
from jumping_quadrupeds.drqv2.networks import Actor, Critic, Encoder, DiscreteActor
from jumping_quadrupeds.utils import soft_update_params, to_torch, schedule, preprocess_obs, convert_action_to_onehot


class DrQV2Agent:
    def __init__(
        self,
        obs_space,
        action_space,
        device,
        lr,
        critic_lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        log_std_init,
        **kwargs,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.log_std_init = log_std_init
        # models
        self.encoder = Encoder(obs_space).to(device)
        # self.actor = Actor(self.encoder.repr_dim, action_space, feature_dim, hidden_dim, log_std_init).to(device)
        if action_space.__class__.__name__ == "Discrete":
            self.discrete_actions = True
            self.action_dim = action_space.n
            self.actor = DiscreteActor(self.encoder.repr_dim, action_space, feature_dim, hidden_dim, log_std_init).to(device)

        else:
            self.discrete_actions = False
            self.action_dim = action_space.shape[0]
            self.actor = Actor(self.encoder.repr_dim, action_space, feature_dim, hidden_dim, log_std_init).to(device)

        self.critic = Critic(self.encoder.repr_dim, self.action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, self.action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr if critic_lr else lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, action, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev, duration = schedule(self.stddev_schedule, step, self.log_std_init)
        dist = self.actor(obs, stddev)
        if self.discrete_actions:
            if eval_mode:
                action = torch.argmax(dist)
            else:
                action = dist.sample()
                if step < self.num_expl_steps:
                    action = torch.randint(high=3, size=dist.batch_shape).to(self.device)
        else:
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        value = np.array([0.0], dtype=np.float32)
        log_p = dist.log_prob(action).detach().cpu().numpy()[0]
        action = action.detach().cpu().numpy()[0]
        return action, value, log_p

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        with torch.no_grad():
            stddev, duration = schedule(self.stddev_schedule, step, self.log_std_init)
            dist = self.actor(next_obs, stddev)
            if self.discrete_actions:
                next_action = dist.sample()
                next_action = convert_action_to_onehot(next_action, action_dim=self.action_dim)
            else:
                next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_q1"] = Q1.mean().item()
        metrics["critic_q2"] = Q2.mean().item()
        metrics["critic_loss"] = critic_loss.item()
        metrics["action_noise_std_dev"] = (
            stddev.item() if stddev is not None else self.actor.log_std.detach().cpu().numpy()
        )

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev, duration = schedule(self.stddev_schedule, step, self.log_std_init)
        dist = self.actor(obs, stddev)
        if self.discrete_actions:
            action = dist.sample()
        else:
            action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        if self.discrete_actions:
            action = convert_action_to_onehot(action, self.action_dim)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["actor_logprob"] = log_prob.mean().item()
        metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        metrics["update_actor_action_mean"] = action.detach().mean(axis=0).cpu().numpy()
        metrics["update_actor_action_std"] = action.detach().std(axis=0).cpu().numpy()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = to_torch(next(replay_iter), self.device)
        if self.discrete_actions:
            action = convert_action_to_onehot(action, self.action_dim)
        batch_size = obs.shape[0]
        obs = preprocess_obs(obs, self.device)
        next_obs = preprocess_obs(next_obs, self.device)

        # fold the batch in
        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        next_obs = rearrange(next_obs, "b t c h w -> (b t) c h w")

        # augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # expand the timesteps back out
        obs = rearrange(obs, "(b t) c h w -> b t c h w", b=batch_size)
        next_obs = rearrange(next_obs, "(b t) c h w -> b t c h w", b=batch_size)

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        metrics["batch_reward"] = reward.mean().item()
        # metrics["action"] = action.detach().cpu().numpy()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def save_checkpoint(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        print(f"loading checkpoint from: {path}")
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.encoder_opt.load_state_dict(checkpoint["encoder_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])

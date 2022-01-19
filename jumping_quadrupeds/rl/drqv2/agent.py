# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

from jumping_quadrupeds.rl.utils import soft_update_params, to_torch, schedule
from jumping_quadrupeds.rl.drqv2.networks import Actor, Critic, Encoder
from jumping_quadrupeds.rl.drqv2.augs import RandomShiftsAug

class DrQV2Agent:
    def __init__(
        self,
        obs_space,
        action_space,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
            **kwargs
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_space).to(device)
        self.actor = Actor(
            self.encoder.repr_dim, action_space, feature_dim, hidden_dim
        ).to(device)

        self.critic = Critic(
            self.encoder.repr_dim, action_space, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_space, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.detach().cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_reward_producing_critic_target_q"] = reward.mean().item()
        metrics["target_V"] = target_V.mean().item()
        metrics["critic_q1"] = Q1.mean().item()
        metrics["critic_q2"] = Q2.mean().item()
        metrics["critic_loss"] = critic_loss.item()
        metrics["action_noise_std_dev"] = stddev.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
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
        action_mean = action.detach().mean(axis=0).cpu().numpy()
        action_std = action.detach().std(axis=0).cpu().numpy()
        metrics.update({
            "act-mean-turn": action_mean[0],
            "act-mean-gas": action_mean[1],
            "act-mean-brake": action_mean[2],
            "act-std-turn": action_std[0],
            "act-std-gas": action_std[1],
            "act-std-brake": action_std[2]
        })


        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = to_torch(batch.values(), self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def save_checkpoint(self, exp_dir, epoch):
        torch.save(
            self.actor.state_dict(),
            f"{exp_dir}/Weights/actor-{epoch}.pt",
        )
        torch.save(
            self.critic.state_dict(),
            f"{exp_dir}/Weights/critic-{epoch}.pt",
        )
        torch.save(
            self.critic_target.state_dict(),
            f"{exp_dir}/Weights/critic_target-{epoch}.pt",
        )
        torch.save(
            self.encoder.state_dict(),
            f"{exp_dir}/Weights/encoder-{epoch}.pt",
        )
        torch.save(
            self.encoder_opt.state_dict(),
            f"{exp_dir}/Weights/encoder_opt-{epoch}.pt",
        )
        torch.save(
            self.actor_opt.state_dict(),
            f"{exp_dir}/Weights/actor_opt-{epoch}.pt",
        )
        torch.save(
            self.critic_opt.state_dict(),
            f"{exp_dir}/Weights/critic_opt-{epoch}.pt",
        )
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid

from jumping_quadrupeds.augs import RandomShiftsAug
from jumping_quadrupeds.smaq.networks import Actor, Critic, DiscreteActor
from jumping_quadrupeds.utils import soft_update_params, to_torch, schedule, preprocess_obs, convert_action_to_onehot


class SMAQAgent:
    def __init__(
        self,
        action_space,
        model,
        model_ema,
        device,
        lr,
        critic_lr,
        model_lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        model_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        log_std_init,
        use_actor_ln=True,
        weight_decay=0.01,
        critic_mask_aug=False,
        use_model_ema=True,
        use_masked_state_loss=False,
        use_q_approx_loss=False,
        use_drqv2_augs=False,
        smaq_update_optim_step=False,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.model_target_tau = model_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.log_std_init = log_std_init
        self.critic_mask_aug = critic_mask_aug
        self.use_model_ema = use_model_ema
        self.action_space = action_space
        self.use_masked_state_loss = use_masked_state_loss
        self.use_q_approx_loss = use_q_approx_loss
        self.use_drqv2_augs = use_drqv2_augs
        self.smaq_update_optim_step = smaq_update_optim_step

        # models
        self.model = model
        self.model_ema = model_ema
        if action_space.__class__.__name__ == "Discrete":
            self.discrete_actions = True
            self.action_dim = action_space.n
            self.actor = DiscreteActor(self.model.out_dim, action_space, feature_dim, hidden_dim, log_std_init,
                                       use_actor_ln).to(device)

        else:
            self.discrete_actions = False
            self.action_dim = action_space.shape[0]
            self.actor = Actor(self.model.out_dim, action_space, feature_dim, hidden_dim, log_std_init,
                               use_actor_ln).to(device)

        self.critic = Critic(self.model.out_dim, self.action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.model.out_dim, self.action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.model_opt = torch.optim.AdamW(self.model.parameters(), lr=model_lr, weight_decay=weight_decay)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr if critic_lr else lr, weight_decay=weight_decay
        )

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, action, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        if self.discrete_actions:
            action = convert_action_to_onehot(action, action_dim=self.action_dim)

        obs = self.model(obs, action=action, mask_type="None", decode=False)
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

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, obs_aug=None, next_obs_aug=None, masked_state_mse=None
    ):
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
        if obs_aug is not None:
            Q1_aug, Q2_aug = self.critic(obs_aug, action)
            approximation_error = F.mse_loss(Q1, Q1_aug) + F.mse_loss(Q2, Q2_aug)
            metrics["critic_q1_aug_std"] = Q1_aug.std().item()
            metrics["critic_q2_aug_std"] = Q2_aug.std().item()
            metrics["critic_q1_aug"] = Q1_aug.mean().item()
            metrics["critic_q2_aug"] = Q2_aug.mean().item()
            metrics["augmentation_approximation_error"] = approximation_error.item()
            # Q1 = torch.mean(torch.stack([Q1, Q1_aug], dim=0), dim=0)
            # Q2 = torch.mean(torch.stack([Q2, Q2_aug], dim=0), dim=0)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_target_q_std"] = target_Q.std().item()
        metrics["critic_q1_std"] = Q1.std().item()
        metrics["critic_q2_std"] = Q2.std().item()
        metrics["critic_q1"] = Q1.mean().item()
        metrics["critic_q2"] = Q2.mean().item()
        metrics["critic_loss"] = critic_loss.item()
        metrics["action_noise_std_dev"] = (
            stddev.item() if stddev is not None else self.actor.log_std.detach().cpu().numpy()
        )

        return critic_loss, approximation_error, metrics

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

    def smaq_update(self, obs, action):
        recon_loss, pred_pixel_values, masked_indices, unmasked_indices, patches = self.model(
            obs, action=action, mask_type=self.model.auto_encoding_mask_type, decode=True
        )
        gt_img, pred_img, gt_masked_img = self.render_reconstruction(
            pred_pixel_values[0], patches[0], masked_indices[0], obs[0], seq_len=obs.shape[1]
        )
        if self.smaq_update_optim_step:
            self.model_opt.zero_grad(set_to_none=True)
            recon_loss.backward()
            self.model_opt.step()

        return recon_loss, {
            "smaq_loss": recon_loss.detach().cpu().item(),
            "gt_img": gt_img,
            "pred_img": pred_img,
            "gt_masked_img": gt_masked_img,
        }

    def render_reconstruction(self, pred_pixel_values, input, masked_indices, img, seq_len, channels=3):
        h, w = self.model.tokenizer.patch_height, self.model.tokenizer.patch_width
        p1, p2 = img.shape[-2] // h, img.shape[-1] // w
        pred_patches = rearrange(pred_pixel_values, "p (h w c) -> p c h w", c=channels, h=h, w=w)
        input = self.model.tokenizer.patchify(input.unsqueeze(0))[0]
        input = rearrange(input, "p (h w c) -> p c h w", c=channels, h=h, w=w)

        pred_recons = input.clone()
        pred_w_mask = input.clone()
        pred_recons[masked_indices] = pred_patches
        pred_w_mask[masked_indices] = 0.0

        pred_recons = rearrange(
            pred_recons, "(s p1 p2) c h w -> s c (p1 h) (p2 w)", p1=p1, p2=p2, c=3, h=h, w=w, s=seq_len
        )
        pred_w_mask = rearrange(
            pred_w_mask, "(s p1 p2) c h w -> s c (p1 h) (p2 w)", p1=p1, p2=p2, c=3, h=h, w=w, s=seq_len
        )
        gt_img = make_grid(img)
        pred_img = make_grid(pred_recons)
        gt_masked_img = make_grid(pred_w_mask)
        return gt_img, pred_img, gt_masked_img

    def update(self, replay_loader, step):
        metrics = dict()

        obs, action, reward, discount, next_obs = to_torch(next(replay_loader), self.device)
        if self.discrete_actions:
            action = convert_action_to_onehot(action, self.action_dim)
        obs = preprocess_obs(obs, self.device)
        next_obs = preprocess_obs(next_obs, self.device)

        # update model
        recon_loss, smaq_metrics = self.smaq_update(obs, action)
        metrics.update(smaq_metrics)
        batch_size = obs.shape[0]

        # fold the batch in
        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        next_obs = rearrange(next_obs, "b t c h w -> (b t) c h w")

        # augment
        if self.use_drqv2_augs:
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)

        # expand the timesteps back out
        obs = rearrange(obs, "(b t) c h w -> b t c h w", b=batch_size)
        next_obs = rearrange(next_obs, "(b t) c h w -> b t c h w", b=batch_size)

        # encode
        obsenc = self.model(obs, action=action, mask_type="None")
        obs_aug = self.model(obs, action=action, mask_type=self.model.state_encoding_mask_type)

        masked_state_mse = None
        if self.use_masked_state_loss:
            obs = self.model(obs, action=action, mask_type="None")
            masked_state_mse = F.mse_loss(obs, obs_aug)
            metrics["masked_state_mse"] = masked_state_mse.item()
            metrics["encoded_state_gt_vs_masked_l1"] = F.l1_loss(obs.detach(), obs_aug.detach())

        with torch.no_grad():
            if self.use_model_ema:
                next_obs_aug = self.model_ema(next_obs, action=action, mask_type=self.model.state_encoding_mask_type)
                next_obs = self.model_ema(next_obs, action=action, mask_type="None")
            else:
                next_obs_aug = self.model(next_obs, action=action, mask_type=self.model.state_encoding_mask_type)
                next_obs = self.model(next_obs, action=action, mask_type="None")

        # update critic

        critic_loss, approx_err, critic_metrics = self.update_critic(
            obsenc, action, reward, discount, next_obs, step, obs_aug, next_obs_aug
        )
        metrics.update(critic_metrics)
        self.model_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        if self.use_q_approx_loss:
            critic_loss = critic_loss + approx_err
        if self.use_masked_state_loss:
            critic_loss = critic_loss + masked_state_mse
        if not self.smaq_update_optim_step:
            critic_loss = critic_loss + recon_loss
        critic_loss.backward()
        self.critic_opt.step()
        self.model_opt.step()

        # update actor
        metrics.update(self.update_actor(obsenc.detach(), step))

        metrics["batch_reward"] = reward.mean().item()

        # update critic target
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        soft_update_params(self.model, self.model_ema, self.model_target_tau)

        return metrics

    def save_checkpoint(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
            "model_opt": self.model_opt.state_dict(),
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
        self.model.load_state_dict(checkpoint["model"])
        self.model_opt.load_state_dict(checkpoint["model_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])

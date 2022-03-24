# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import os
from einops import rearrange
from jumping_quadrupeds.utils import soft_update_params, to_torch, schedule, preprocess_obs
from jumping_quadrupeds.mae.networks import Actor, Critic, Encoder, MAE, ViT
from jumping_quadrupeds.augs import RandomShiftsAug


class MAEAgent:
    def __init__(
        self,
        obs_space,
        action_space,
        device,
        lr,
        critic_lr,
        encoder_lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        log_std_init,
        patch_size=12,
        mae_encoder_dim=256,
        mae_encoder_head_dim=64,
        mae_encoder_depth=2,
        mae_encoder_heads=8,
        mae_encoder_mlp_dim=512,
        mae_encoder_dropout=0.1,
        mae_encoder_emb_dropout=0.1,
        mae_encoder_nonlinearity="gelu",
        mae_encoder_use_last_ln=True,
        mae_encoder_qkv_bias=False,
        mae_masking_ratio=0.75,
        mae_decoder_dim=512,
        mae_decoder_depth=2,
        mae_decoder_heads=1,
        mae_decoder_dim_head=128,
        use_actor_ln=True,
        weight_decay=0.01,
        **kwargs,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.log_std_init = log_std_init
        vit = ViT(
            obs_space=obs_space,
            tokenizer=tokenizer,
            dim=mae_encoder_dim,
            dim_head=mae_encoder_head_dim,
            depth=mae_encoder_depth,
            heads=mae_encoder_heads,
            mlp_dim=mae_encoder_mlp_dim,
            dropout=mae_encoder_dropout,
            emb_dropout=mae_encoder_emb_dropout,
            encoder_nonlinearity=mae_encoder_nonlinearity,
            use_last_ln=mae_encoder_use_last_ln,
            qkv_bias=mae_encoder_qkv_bias,
        ).to(device)
        mae = MAE(
            encoder=vit,
            patch_size=patch_size,
            masking_ratio=mae_masking_ratio,  # the paper recommended 75% masked patches
            decoder_dim=mae_decoder_dim,  # paper showed good results with just 512
            decoder_depth=mae_decoder_depth,  # anywhere from 1 to 8
            decoder_dim_head=mae_decoder_dim_head,
            decoder_heads=mae_decoder_heads,
            device=device,
        )

        # models
        self.encoder = mae.to(device)
        self.actor = Actor(self.encoder.repr_dim, action_space, feature_dim, hidden_dim, log_std_init, use_actor_ln).to(
            device
        )

        self.critic = Critic(self.encoder.repr_dim, action_space, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_space, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.AdamW(self.encoder.parameters(), lr=encoder_lr, weight_decay=weight_decay)
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
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs)
        stddev, duration = schedule(self.stddev_schedule, step, self.log_std_init)

        dist = self.actor(obs, stddev)
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
        metrics["update_actor_action_mean"] = action.detach().mean(axis=0).cpu().numpy()
        metrics["update_actor_action_std"] = action.detach().std(axis=0).cpu().numpy()
        return metrics

    def mae_update(self, obs):
        recon_loss, pred_pixel_values, masked_indices, unmasked_indices, patches = self.encoder(obs, eval=False)
        self.encoder_opt.zero_grad()
        recon_loss.backward()
        self.encoder_opt.step()
        gt_img, pred_img, gt_masked_img = self.render_reconstruction(
            pred_pixel_values[0], patches[0], masked_indices[0], obs[0], framestack=3
        )  # TODO: get framestack from obs
        return {
            "mae_loss": recon_loss.cpu().item(),
            "gt_img": gt_img,
            "pred_img": pred_img,
            "gt_masked_img": gt_masked_img,
        }

    def render_reconstruction(self, pred_pixel_values, patches, masked_indices, img, framestack=3, channels=3):
        masked_patch_pred = pred_pixel_values.detach().cpu()
        masked_patch_true = patches.cpu()

        # tmp, TODO: remove this once dimensions are correct
        masked_patch_true = rearrange(masked_patch_true, "p (c s) -> (p s) c", s=framestack)
        masked_patch_pred = rearrange(masked_patch_pred, "p (c s) -> (p s) c", s=framestack)

        pred_patches = rearrange(masked_patch_pred, "p (h w c) -> p c h w", c=channels, h=12, w=12)
        gt_patches = rearrange(masked_patch_true, "(p s) (h w c) -> (p s) c h w", c=channels, h=12, w=12, s=framestack)

        pred_recons = gt_patches.clone()
        pred_w_mask = gt_patches.clone()

        # TODO: remove mask repeat once dimensions match the gt
        pred_recons[masked_indices.repeat(framestack)] = pred_patches
        pred_w_mask[masked_indices.repeat(framestack)] = 0.0

        pred_recons = rearrange(
            pred_recons, "(s p1 p2) c h w -> s c (p1 h) (p2 w)", p1=7, p2=7, c=3, h=12, w=12, s=framestack
        )
        pred_w_mask = rearrange(
            pred_w_mask, "(s p1 p2) c h w -> s c (p1 h) (p2 w)", p1=7, p2=7, c=3, h=12, w=12, s=framestack
        )

        img = rearrange(img.cpu(), "(s c) h w -> s c h w", s=framestack)
        gt_img = torchvision.utils.make_grid(img)
        pred_img = torchvision.utils.make_grid(pred_recons.clip(0, 1), nrow=7)
        gt_masked_img = torchvision.utils.make_grid(pred_w_mask, nrow=7 * framestack)
        return gt_img, pred_img, gt_masked_img

    def update(self, replay_loader, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = to_torch(next(replay_loader), self.device)

        obs = preprocess_obs(obs, self.device)
        next_obs = preprocess_obs(next_obs, self.device)

        # update encoder
        # if step % 5 == 0:
        metrics.update(self.mae_update(obs))

        # augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        metrics["batch_reward"] = reward.mean().item()
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

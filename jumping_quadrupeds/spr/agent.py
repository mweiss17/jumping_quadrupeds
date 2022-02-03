# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d

from jumping_quadrupeds.utils import soft_update_params, to_torch, schedule, preprocess_obs
from jumping_quadrupeds.spr.networks import Actor, Critic, Encoder, TransitionModel, Conv2dModel
from jumping_quadrupeds.augs import RandomShiftsAug, Intensity

class SPRAgent:
    def __init__(
        self,
        obs_space,
        action_space,
        device,
        jumps,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        renormalize,
        residual_tm,
        augmentations,
        log_std_init,
        classifier_type,
        use_spr,
        **kwargs
    ):
        self.device = device
        self.jumps = jumps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.renormalize = renormalize
        self.residual_tm = residual_tm
        self.augmentations = augmentations
        self.aug_prob = 0.5
        self.log_std_init = log_std_init
        self.classifier_type = classifier_type
        self.use_spr = use_spr

        # models
        out_channels = 32
        self.encoder = Encoder(obs_space, out_channels).to(device)
        self.actor = Actor(
            self.encoder.repr_dim, action_space, feature_dim, hidden_dim, log_std_init
        ).to(device)

        self.critic = Critic(
            self.encoder.repr_dim, action_space, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_space, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.jumps > 0:
            self.dynamics_model = TransitionModel(channels=out_channels,
                                                  num_actions=action_space.shape[0],
                                                  hidden_size=hidden_dim,
                                                  limit=1,
                                                  renormalize=renormalize,
                                                  residual=residual_tm).to(device)
        else:
            self.dynamics_model = nn.Identity().to(device)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        if augmentations:
            self.transforms = []
            self.eval_transforms = []
            self._build_augmentations()

        if self.use_spr:
            self.momentum_encoder = momentum_encoder
            self.momentum_tau = momentum_tau
            self.shared_encoder = shared_encoder
            self._build_spr()

        if self.momentum_encoder:
            self.target_encoder = copy.deepcopy(self.encoder)
            self.global_target_classifier = copy.deepcopy(self.global_target_classifier)
            self.local_target_classifier = copy.deepcopy(self.local_target_classifier)
            for param in (list(self.target_encoder.parameters())
                          + list(self.global_target_classifier.parameters())
                          + list(self.local_target_classifier.parameters())):
                param.requires_grad = False

        self.train()
        self.critic_target.train()

    def _build_augmentations(self):
        self.uses_augmentation = False
        for aug in self.augmentations:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

    def _build_spr(self):
        assert not (self.shared_encoder and self.momentum_encoder)
        self.local_classifier = self.local_target_classifier = nn.Identity()
        self.global_final_classifier = nn.Identity()
        if self.classifier_type == "mlp":
            self.global_classifier = nn.Sequential(
                nn.Flatten(-3, -1),
                nn.Linear(self.pixels * self.hidden_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            self.global_target_classifier = self.global_classifier
            global_spr_size = 256
        elif self.classifier_type == "q_l1":
            self.global_classifier = QL1Head(self.head, dueling=dueling, type=q_l1_type)
            global_spr_size = self.global_classifier.out_features
            self.global_target_classifier = self.global_classifier
        elif self.classifier_type == "q_l2":
            self.global_classifier = nn.Sequential(self.head, nn.Flatten(-2, -1))
            self.global_target_classifier = self.global_classifier
            global_spr_size = 256
        elif self.classifier_type == "bilinear":
            self.global_classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                   nn.Linear(self.hidden_size * self.pixels,
                                                             self.hidden_size * self.pixels))
            self.global_target_classifier = nn.Flatten(-3, -1)
        elif self.classifier_type == "none":
            self.global_classifier = nn.Flatten(-3, -1)
            self.global_target_classifier = nn.Flatten(-3, -1)

            global_spr_size = self.hidden_size * self.pixels
        if final_classifier == "mlp":
            self.global_final_classifier = nn.Sequential(
                nn.Linear(global_spr_size, global_spr_size * 2),
                nn.BatchNorm1d(global_spr_size * 2),
                nn.ReLU(),
                nn.Linear(global_spr_size * 2, global_spr_size)
            )
        elif final_classifier == "linear":
            self.global_final_classifier = nn.Sequential(
                nn.Linear(global_spr_size, global_spr_size),
            )

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(image, transform,
                                        eval_transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float() / 255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None,
                                                     flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        if len(obs.shape) < 4:
            obs = obs.unsqueeze(0)
        obs = self.encoder(obs)
        stddev, duration = schedule(self.stddev_schedule, step, self.log_std_init)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        value = np.array([0.], dtype=np.float32)
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

        # arrays for predicted values
        log_pred_ps = []
        pred_reward = []
        pred_latents = []

        batch = next(replay_iter)
        obs, action, reward, discount = to_torch(batch.values(), self.device)
        obs = preprocess_obs(obs, self.device)

        batch_size, timesteps, channels, w, h = obs.shape
        first_obs = obs[:, 0]

        first_obs = self.transform(first_obs, augment=True)
        # TODO: encoder should take in prev action and reward
        latent = self.encoder(first_obs, flatten=False)
        if self.renormalize:
            latent = renormalize(latent, -3)

        pred_latents.append(latent)

        if self.jumps > 0:
            pred_rew = self.dynamics_model.reward_predictor(pred_latents[0])
            pred_reward.append(F.log_softmax(pred_rew, -1))
            # for each observation, predict next latent
            for i in range(self.jumps):
                latent, pred_rew = self.dynamics_model(latent, action[:, i])
                pred_rew = pred_rew[:obs.shape[1]]
                pred_latents.append(latent)
                pred_reward.append(F.log_softmax(pred_rew, -1))

        if self.use_spr:
            spr_loss = self.do_spr_loss(pred_latents, observation)
        else:
            spr_loss = torch.zeros((self.jumps + 1, observation.shape[1]), device=latent.device)
        breakpoint()

        return log_pred_ps, pred_reward, spr_loss

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


    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        # Gradients of norrmalized L2 loss and cosine similiarity are proportional.
        # See: https://stats.stackexchange.com/a/146279
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def do_spr_loss(self, pred_latents, observation):
        pred_latents = torch.stack(pred_latents, 1)
        latents = pred_latents[:observation.shape[1]].flatten(0, 1)  # batch*jumps, *
        neg_latents = pred_latents[observation.shape[1]:].flatten(0, 1)
        latents = torch.cat([latents, neg_latents], 0)
        target_images = observation[self.time_offset:self.jumps + self.time_offset+1].transpose(0, 1).flatten(2, 3)
        target_images = self.transform(target_images, True)

        if not self.momentum_encoder and not self.shared_encoder:
            target_images = target_images[..., -1:, :, :]
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            target_latents = self.target_encoder(target_images.flatten(0, 1))
            if self.renormalize:
                target_latents = renormalize(target_latents, -3)

        spr_loss = self.global_spr_loss(latents, target_latents, observation)
        spr_loss = spr_loss.view(-1, observation.shape[1]) # split to batch, jumps

        if self.momentum_encoder:
            update_state_dict(self.target_encoder,
                              self.conv.state_dict(),
                              self.momentum_tau)
            if self.classifier_type != "bilinear":
                update_state_dict(self.global_target_classifier,
                                  self.global_classifier.state_dict(),
                                  self.momentum_tau)
        return spr_loss

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

def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)

def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images

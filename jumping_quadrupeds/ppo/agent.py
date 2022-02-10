from collections import deque

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from jumping_quadrupeds.ppo.networks import ConvActorCritic
from jumping_quadrupeds.utils import preprocess_obs, to_torch

class PPOAgent:
    """
    Proximal Policy Optimization (by clipping),
    with early stopping based on approximate KL
    """

    def __init__(
        self,
        observation_space,
        action_space,
        use_wandb,
        device=None,
        pi_lr=None,
        vf_lr=None,
        rew_smooth_len=None,
        target_kl=None,
        train_pi_iters=None,
        train_v_iters=None,
        ac_checkpoint=None,
        vf_optim_checkpoint=None,
        pi_optim_checkpoint=None,
        shared_encoder=None,
        conv_ac_hidden_scaling=None,
        log_std=None,
        freeze_encoder=None,
        clip_ratio=None,
        **kwargs,
    ) -> None:

        self.device = device
        self.use_wandb = use_wandb
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.ac_checkpoint = ac_checkpoint
        self.vf_optim_checkpoint = vf_optim_checkpoint
        self.pi_optim_checkpoint = pi_optim_checkpoint
        self.ep_rew_mean = deque(maxlen=rew_smooth_len)
        self.ep_len_mean = deque(maxlen=rew_smooth_len)
        self.total_steps = 0
        self.total_episodes = 0

        # policy and value networks
        self.ac = ConvActorCritic(
            observation_space,
            action_space,
            shared_encoder=shared_encoder,
            hidden_sizes=conv_ac_hidden_scaling,
            log_std=log_std,
        )

        # Load checkpoints
        if self.ac_checkpoint:
            self.load_checkpoint()

        # do we want to freeze the encoder?
        if freeze_encoder:
            self.ac.freeze_encoder()

        # Put on device
        self.ac = self.ac.to(self.device)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(
            self.ac.get_policy_params(), lr=pi_lr
        )
        self.vf_optimizer = Adam(
            self.ac.get_value_params(), lr=vf_lr
        )


    def compute_loss_pi(self, obs, act, adv, logp_old):
        """Computes policy loss"""
        obs = preprocess_obs(obs, self.device)

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = (
            torch.clamp(
                ratio,
                1 - self.clip_ratio,
                1 + self.clip_ratio,
            )
            * adv
        )
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(
            1 - self.clip_ratio
        )
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(
            kl=approx_kl, ent=ent, cf=clipfrac, logp=logp, ratio=ratio, adv=adv
        )

        return loss_pi, pi_info

    def compute_loss_v(self, obs, ret):
        """Computes value loss"""
        obs = preprocess_obs(obs, self.device)
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self, replay_loader, step):
        """Updates the policy and value function based on the latest replay buffer"""
        metrics = {}
        obs, action, rew, adv, ret, logp = to_torch(next(iter(replay_iter)).values(), self.device)

        self.ac.train()

        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(obs, action, adv, logp)
            kl = np.mean(pi_info["kl"])
            if kl > self.target_kl:
                tqdm.write(
                    f"Early stopping at step {i}/{self.train_pi_iters} due to reaching max kl."
                )
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(obs, ret)
            loss_v.backward()
            self.vf_optimizer.step()

        metrics["update_actor_action_mean"] = action.detach().mean(axis=0).cpu().numpy()
        metrics["update_actor_action_std"] = action.detach().std(axis=0).cpu().numpy()
        logp_mean = pi_info["logp"].detach().mean(axis=0).cpu().numpy()
        adv_mean = pi_info["adv"].detach().mean().cpu().numpy()
        adv_std = pi_info["adv"].detach().std().cpu().numpy()
        ratio_mean = pi_info["ratio"].detach().mean().cpu().numpy()
        ratio_std = pi_info["ratio"].detach().std().cpu().numpy()
        metrics.update({
                "log-p": logp_mean,
                "loss-pi": loss_pi.detach().item(),
                "loss-v": loss_v.item(),
                "true-return": ret.detach().mean().cpu().numpy(),
                "KL": kl,
                "entropy": np.mean(pi_info["ent"]),
                "clip-frac": np.mean(pi_info["cf"]),
                "adv-mean": adv_mean,
                "adv-std": adv_std,
                "ratio-mean": ratio_mean,
                "ratio-std": ratio_std,
            })
        return metrics


    def save_checkpoint(self, exp_dir, epoch):
        torch.save(
            self.ac.state_dict(),
            f"{exp_dir}/Weights/ac-{epoch}.pt",
        )
        torch.save(
            self.pi_optimizer.state_dict(),
            f"{exp_dir}/Weights/pi-optim-{epoch}.pt",
        )
        torch.save(
            self.vf_optimizer.state_dict(),
            f"{exp_dir}/Weights/vf-optim-{epoch}.pt",
        )

    def load_checkpoint(self):
        self.ac.load_state_dict(torch.load(self.ac_checkpoint))
        self.pi_optimizer.load_state_dict(
            torch.load(self.pi_optim_checkpoint)
        )
        self.vf_optimizer.load_state_dict(
            torch.load(self.vf_optim_checkpoint)
        )


    def act(self, obs, step, eval_mode):
        return self.ac.step(obs, eval_mode)

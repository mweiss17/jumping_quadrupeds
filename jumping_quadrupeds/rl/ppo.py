from collections import deque

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm, trange

from jumping_quadrupeds.rl.buffer import PpoBuffer
from jumping_quadrupeds.rl.networks import AbstractActorCritic


class PPO:
    """
    Proximal Policy Optimization (by clipping),
    with early stopping based on approximate KL
    """

    def __init__(
        self,
        exp,
        env,
        actor_critic: AbstractActorCritic,
        buf: PpoBuffer,
    ) -> None:

        self.exp = exp
        self._config = exp._config
        self.ac = actor_critic
        self.env = env
        self.buf = buf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Random seed
        # this should be done outside
        # torch.manual_seed(params.seed)
        # np.random.seed(params.seed)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(
            self.ac.get_policy_params(), lr=self._config.get("pi_lr")
        )
        self.vf_optimizer = Adam(
            self.ac.get_value_params(), lr=self._config.get("vf_lr")
        )

        # Load checkpoints
        if self._config.get("weight_checkpoint"):
            self.load_checkpoint()

        self.obs = None
        self.ep_rew_mean = deque(maxlen=self._config.get("rew_smooth_len"))
        self.ep_len_mean = deque(maxlen=self._config.get("rew_smooth_len"))
        self.total_steps = 0

    def compute_loss_pi(self, data):
        """Computes policy loss"""

        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = (
            torch.clamp(
                ratio,
                1 - self._config.get("clip_ratio"),
                1 + self._config.get("clip_ratio"),
            )
            * adv
        )
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self._config.get("clip_ratio")) | ratio.lt(
            1 - self._config.get("clip_ratio")
        )
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        """Computes value loss"""

        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self):
        """Updates the policy and value function based on the latest replay buffer"""

        data = self.buf.get()

        ## this is only used for debugging - compute the old loss of policy and value function
        # pi_l_old, pi_info_old = self.compute_loss_pi(data)
        # pi_l_old = pi_l_old.item()
        # v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        if self._config.get("verbose"):
            tqdm.write("Training pi")

        self.ac.train()

        for i in range(self._config.get("train_pi_iters")):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = np.mean(pi_info["kl"])
            if kl > 1.5 * self._config.get("target_kl"):
                if self._config.get("verbose"):
                    tqdm.write(
                        f"Early stopping at step {i}/{self._config.get('train_pi_iters')} due to reaching max kl."
                    )
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        # Value function learning
        if self._config.get("verbose"):
            tqdm.write("Training Val")

        for i in range(self._config.get("train_v_iters")):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

        ## Log changes from update
        # kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        # logger.store(
        #     LossPi=pi_l_old,
        #     LossV=v_l_old,
        #     KL=kl,
        #     Entropy=ent,
        #     ClipFrac=cf,
        #     DeltaLossPi=(loss_pi.item() - pi_l_old),
        #     DeltaLossV=(loss_v.item() - v_l_old),
        # )

    def save_checkpoint(self, epoch):
        torch.save(
            self.ac.state_dict(),
            open(f"{self.exp.experiment_directory}/Weights/ac-{epoch}.pt", "wb"),
        )
        torch.save(
            self.pi_optimizer.state_dict(),
            open(
                f"{self.exp.experiment_directory}/Weights/pi-optim-{epoch}.pt",
                "wb",
            ),
        )
        torch.save(
            self.vf_optimizer.state_dict(),
            open(
                f"{self.exp.experiment_directory}/Weights/vf-optim-{epoch}.pt",
                "wb",
            ),
        )

    def load_checkpoint(self):
        self.ac.load_state_dict(torch.load(self._config.get("weight_checkpoint")))
        self.pi_optimizer.load_state_dict(
            torch.load(self._config.get("pi_optim_checkpoint"))
        )
        self.vf_optimizer.load_state_dict(
            torch.load(self._config.get("vf_optim_checkpoint"))
        )

    def train_loop(self):
        """Automatic training loop for PPO that trains for prespecified number of epochs"""

        # Main loop: collect experience in env and update/log each epoch
        for epoch in trange(self._config.get("epochs")):
            if self._config.get("verbose"):
                tqdm.write("Collecting data")
            self.collect_data()

            # Save model
            if (epoch % self._config.get("save_freq") == 0) or (
                epoch == self._config.get("epochs") - 1
            ):
                self.save_checkpoint(epoch)

            # Perform PPO update!
            if self._config.get("verbose"):
                tqdm.write("Updating PPO")
            self.update()

        # Log info about epoch
        # logger.log_tabular("Epoch", epoch)
        # logger.log_tabular("EpRet", with_min_and_max=True)
        # logger.log_tabular("EpLen", average_only=True)
        # logger.log_tabular("VVals", with_min_and_max=True)
        # logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        # logger.log_tabular("LossPi", average_only=True)
        # logger.log_tabular("LossV", average_only=True)
        # logger.log_tabular("DeltaLossPi", average_only=True)
        # logger.log_tabular("DeltaLossV", average_only=True)
        # logger.log_tabular("Entropy", average_only=True)
        # logger.log_tabular("KL", average_only=True)
        # logger.log_tabular("ClipFrac", average_only=True)
        # logger.log_tabular("StopIter", average_only=True)
        # logger.log_tabular("Time", time.time() - start_time)
        # logger.dump_tabular()

    def collect_data(self):
        """Fill up the replay buffer with fresh rollouts based on the current policy"""

        if self.obs is None:
            self.obs, self.ep_ret, self.ep_len = self.env.reset(), 0, 0
        episode_counter = 0

        ac_step_timers = []
        env_step_timers = []
        env_reset_timers = []

        self.ac.eval()

        for t in trange(self._config.get("steps_per_epoch")):
            self.act, self.val, self.logp = self.ac.step(
                torch.as_tensor(self.obs, dtype=torch.float32).to(self.device)
            )
            self.next_obs, self.rew, self.done, misc = self.env.step(self.act)

            self.total_steps += 1
            self.ep_ret += self.rew
            self.ep_len += 1

            buf_objs = (self.obs, self.act, self.rew, self.val, self.logp)
            self.buf.store(*buf_objs)

            # logger.store(VVals=v)

            # Update obs (critical!)
            self.obs = self.next_obs

            timeout = self.ep_len == self._config.get("max_ep_len")
            terminal = self.done or timeout
            epoch_ended = t == self._config.get("steps_per_epoch") - 1

            if terminal or epoch_ended:
                episode_counter += 1
                if epoch_ended and not terminal and self._config.get("verbose"):
                    tqdm.write(
                        f"Warning: trajectory cut off by epoch at {self.ep_len} steps."
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, self.val, _ = self.ac.step(
                        torch.as_tensor(self.obs, dtype=torch.float32).to(self.device)
                    )
                else:
                    self.val = 0

                self.ep_rew_mean.append(self.ep_ret)
                self.ep_len_mean.append(self.ep_len)

                if episode_counter % self._config.get("log_ep_freq") == 0:
                    if self._config.get("use_wandb"):
                        self.exp.wandb_log(
                            **{
                                "Episode mean reward": np.mean(self.ep_rew_mean),
                                "Env Steps": self.total_steps,
                                "Episode mean length": np.mean(self.ep_len_mean),
                            }
                        )

                self.buf.finish_path(self.val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    # logger.store(EpRet=ep_ret, EpLen=ep_len)
                    pass

                self.obs, self.ep_ret, self.ep_len = self.env.reset(), 0, 0

    def play(self, episodes=3):
        """play n episodes with the current policy and return the observations, rewards, and actions"""

        obs = self.env.reset()
        episode_counter = 0

        obs_buf = []
        rew_buf = []
        act_buf = []

        while True:
            with torch.no_grad():
                obs_buf.append(np.copy(obs))
                act, _, _ = self.ac.step(
                    torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                )
                obs, rew, done, misc = self.env.step(act)
                rew_buf.append(np.copy(rew))
                act_buf.append(np.copy(act))
                if "state" in misc:
                    obs_buf.pop(-1)
                    obs_buf.append(misc["state"])

            if done:
                episode_counter += 1
                if episode_counter == episodes:
                    break
                else:
                    obs = self.env.reset()

        self.obs, self.ep_ret, self.ep_len = self.env.reset(), 0, 0
        return obs_buf, rew_buf, act_buf

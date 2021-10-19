import os
import torch
import numpy as np
from jumping_quadrupeds.rl.utils import combined_shape, discount_cumsum


class PpoBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        steps_per_epoch,
        gamma,
        lam,
        device,
        save_transitions=0,
    ):
        self.steps_per_epoch = steps_per_epoch
        self.total_save_transitions = save_transitions
        self.cur_transition_count = 0
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.obs_buf = np.zeros(
            combined_shape(self.steps_per_epoch, obs_dim), dtype=np.float32
        )
        self.act_buf = np.zeros(
            combined_shape(self.steps_per_epoch, act_dim), dtype=np.float32
        )
        self.adv_buf = np.zeros(
            combined_shape(self.steps_per_epoch, 1), dtype=np.float32
        )
        self.rew_buf = np.zeros(self.steps_per_epoch, dtype=np.float32)
        self.ret_buf = np.zeros(self.steps_per_epoch, dtype=np.float32)
        self.val_buf = np.zeros(self.steps_per_epoch, dtype=np.float32)
        self.logp_buf = np.zeros(
            combined_shape(self.steps_per_epoch, act_dim), dtype=np.float32
        )
        self.ptr, self.path_start_idx = 0, 0

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert (
            self.ptr < self.steps_per_epoch
        )  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = np.expand_dims(
            discount_cumsum(deltas, self.gamma * self.lam), 1
        )

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get_all(self):
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
            for k, v in data.items()
        }

    def advantage_normalize(self):
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

    def reset(self):
        assert (
            self.ptr == self.steps_per_epoch
        )  # buffer has to be full before you can get
        self.minibatch_ptr, self.ptr, self.path_start_idx = 0, 0, 0

    def save(self, path):
        if self.cur_transition_count < self.total_save_transitions:
            data = self.get_all()
            np.save(
                os.path.join(path, "Logs", f"buffer-{self.cur_transition_count}"), data
            )
            self.cur_transition_count += self.steps_per_epoch


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, obs, act, ret, adv, logp):
        self.obs = obs
        self.act = act
        self.ret = ret
        self.adv = adv
        self.logp = logp

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        sample = dict(
            obs=self.obs[idx],
            act=self.act[idx],
            ret=self.ret[idx],
            adv=self.adv[idx],
            logp=self.logp[idx],
        )
        return sample

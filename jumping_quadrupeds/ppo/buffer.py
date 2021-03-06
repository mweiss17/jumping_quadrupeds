from collections import defaultdict

import numpy as np
import scipy.signal
from torch.utils.data import IterableDataset


class OnPolicyReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, data_specs, gae_lambda, discount, **kwargs):
        super().__init__()
        self._replay_dir = replay_dir
        self._data_specs = data_specs
        self._gae_lambda = gae_lambda
        self._discount = discount
        self._buffer = defaultdict(list)
        self.ptr, self.path_start_idx = 0, 0

    def add(self, time_step, val=None, logp=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # buffer has to have room so you can store
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            if time_step.first() and spec.name != "observation":
                continue
            if time_step.last() and spec.name == "observation":
                continue
            self._buffer[spec.name].append(value)
        if val is not None:
            self._buffer["val"].append(val)
        if logp is not None:
            self._buffer["logp"].append(logp)

        if not time_step.first():
            self.ptr += 1
        if time_step.last():
            self.finish_episode()

    @property
    def cur_ep_len(self):
        return self.ptr - self.path_start_idx

    def finish_episode(self):
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
        if self.path_start_idx == self.ptr:
            return
        last_val = 0.0
        rews = np.stack(self._buffer["reward"], axis=1).flatten()[path_slice]
        rews = np.append(rews, last_val)
        vals = np.stack(self._buffer["val"], axis=1).flatten()[path_slice]
        vals = np.append(vals, last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._discount * vals[1:] - vals[:-1]
        self._buffer["adv"].extend(discount_cumsum(deltas, self._discount * self._gae_lambda))

        # the next line computes rewards-to-go, to be targets for the value function
        self._buffer["ret"].extend(discount_cumsum(rews, self._discount)[:-1])

        self.path_start_idx = self.ptr

    def get_obs_dict(self):
        end = len(self._buffer["action"])
        data = (
            np.array(self._buffer["observation"])[:end],
            np.array(self._buffer["action"]),
            np.array(self._buffer["reward"]),
            np.array(self._buffer["ret"]),
            np.array(self._buffer["adv"]),
            np.array(self._buffer["logp"]),
        )
        return data

    def _sample(self, reset=True):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        self.finish_episode()
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self._buffer["adv"]), np.std(self._buffer["adv"])
        self._buffer["adv"] = (self._buffer["adv"] - adv_mean) / adv_std
        data = self.get_obs_dict()
        if reset:
            self.ptr, self.path_start_idx = 0, 0
            self._buffer = defaultdict(list)

        return data

    def __iter__(self):
        while True:
            yield self._sample()


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def episode_len(episode):
    # -1 for dummy transition (first is just an obs)
    return next(iter(episode.values())).shape[0] - 1


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class OffPolicyReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount, fetch_every, save_snapshot, **kwargs):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)

        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)

        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        return obs, action, reward, discount, next_obs

    def __iter__(self):
        while True:
            yield self._sample()


# import traceback
# import numpy as np
# from collections import defaultdict
# from torch.utils.data import IterableDataset

#
# class OffPolicyReplayBuffer(IterableDataset):
#     def __init__(self, replay_dir, data_specs, max_size, discount, nstep, **kwargs):
#         super().__init__()
#         self._replay_dir = replay_dir
#         self._data_specs = data_specs
#         self._max_size = max_size
#         self._discount = discount
#         self._nstep = nstep
#         self._buffer = []
#         self._current_episode = defaultdict(list)
#
#     def add(self, transition):
#         """
#         Append one timestep of agent-environment interaction to the buffer.
#         """
#         if not transition.get("discount"):
#             transition["discount"] = self._discount
#         for spec in self._data_specs:
#             value = transition.get(spec.name, None)
#             if np.isscalar(value):
#                 value = np.full(spec.shape, value, spec.dtype)
#             if value is None:
#                 continue
#             assert spec.shape == value.shape and spec.dtype == value.dtype
#             self._current_episode[spec.name].append(value)
#
#     def finish_episode(self):
#         if len(self._current_episode["obs"]) > self._nstep:
#             self._buffer.append({k: np.array(v) for k, v in self._current_episode.items()})
#             self._current_episode = defaultdict(list)
#
#     def _sample(self, reset=True):
#         """
#         Call this at the end of an epoch to get all of the data from
#         the buffer, with advantages appropriately normalized (shifted to have
#         mean zero and std one). Also, resets some pointers in the buffer.
#         """
#         self.finish_episode()
#         ep_idx = np.random.randint(0, len(self._buffer))
#         episode = self._buffer[ep_idx]
#         step_idx = np.random.randint(0, episode["obs"].shape[0] - self._nstep)
#         obs = episode["obs"][step_idx - 1]
#         action = episode["act"][step_idx]
#         next_obs = episode["obs"][step_idx - 1 + self._nstep]
#         reward = np.zeros_like(episode["rew"][step_idx])
#         discount = np.ones_like(episode["discount"][step_idx])
#         for i in range(self._nstep):
#             step_reward = episode["rew"][step_idx + i]
#             reward += discount * step_reward
#             discount *= self._discount
#
#         sample = {
#             "obs": obs,
#             "act": action,
#             "rew": reward,
#             "discount": discount,
#             "next_obs": next_obs,
#         }
#         return sample
#
#     def __iter__(self):
#         while True:
#             yield self._sample()

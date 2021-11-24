import io
import os
import torch
import random
import datetime
import traceback
from collections import defaultdict, namedtuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from pathlib import Path

from jumping_quadrupeds.rl.utils import combined_shape, discount_cumsum


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        self._replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, step, eps_done):
        # we have to do this because we don't know the discount from dm_control, assume 1.0
        if not step.get("discount"):
            step["discount"] = 1.0
        for spec in self._data_specs:
            value = step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)

        if eps_done:
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        gae_lambda=None,
        compute_adv=False,
        return_logp=False,
    ):
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
        self._gae_lambda = gae_lambda
        self._compute_adv = compute_adv
        self._return_logp = return_logp

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
        if self._compute_adv:
            episode = self.compute_rets_and_advs(episode)

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

    def compute_rets_and_advs(self, episode):
        # add zeros to the end
        rew = np.pad(episode["rew"], ((0, 1), (0, 0)), "constant")
        val = np.pad(episode["val"], ((0, 1), (0, 0)), "constant")

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rew[:-1] + self._discount * val[1:] - val[:-1]
        adv_buf = discount_cumsum(deltas, self._discount * self._gae_lambda)

        # # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        episode["adv"] = adv_buf

        # the next line computes rewards-to-go, to be targets for the value function
        ret_buf = discount_cumsum(rew, self._discount)[:-1]

        episode["ret"] = ret_buf
        return episode

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()

        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode["obs"][idx - 1]
        action = episode["act"][idx]
        next_obs = episode["obs"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["rew"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["rew"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        sample = {
            "obs": torch.as_tensor(obs, dtype=torch.float32) / 255,
            "act": action,
            "rew": reward,
            "discount": discount,
            "next_obs": next_obs,
        }
        if self._return_logp:
            sample["logp"] = episode["logp"][idx]
        if self._compute_adv:
            sample["adv"] = episode["adv"][idx]
            sample["ret"] = episode["ret"][idx]
        return sample

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    gae_lambda,
    compute_adv,
    return_logp,
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        gae_lambda=gae_lambda,
        compute_adv=compute_adv,
        return_logp=return_logp,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader

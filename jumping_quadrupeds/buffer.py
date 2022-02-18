import io
import torch
import random
import datetime
import traceback
import numpy as np
from torch.utils.data import IterableDataset
from collections import defaultdict


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
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, step):
        for spec in self._data_specs:
            if spec.name == "discount":
                value = 0.99
            else:
                value = step.get(spec.name, None)
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            if value is None:
                continue
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def episode_len(self, episode):
        # -1 for dummy transition (first is just an obs)
        return next(iter(episode.values())).shape[0] - 1

    def finish_episode(self):
        episode = dict()
        for spec in self._data_specs:
            value = self._current_episode[spec.name]
            episode[spec.name] = np.array(value, spec.dtype)
        self._current_episode = defaultdict(list)
        eps_idx = self._num_episodes
        eps_len = self.episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)
        print(f"saving to {self._replay_dir / eps_fn}")


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        data_specs,
        max_size,
        num_workers,
        discount,
        save_snapshot,
        fetch_every,
        **kwargs,
    ):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._max_size_per_worker = max_size // self._num_workers
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        try:
            eps_fn = random.choice(self._episode_fns)
        except IndexError as e:
            raise Exception("Need to have finished at least one episode. Increase num_seed_frames.")
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        print(f"storing {eps_fn}")
        try:
            episode = load_episode(eps_fn)
            print("loaded eps.")
        except:
            return False
        eps_len = self.episode_len(episode)
        while eps_len + self._size > self._max_size_per_worker:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= self.episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        print(f"self._size: {self._size}")

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

    def episode_len(self, episode):
        # -1 for dummy transition (first is just an obs)
        return next(iter(episode.values())).shape[0] - 1

    def _sample(self):
        pass

    def __iter__(self):
        while True:
            yield self._sample()

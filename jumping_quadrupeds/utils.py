import os
import re
import random
import torch
import argparse
import numpy as np
import torch.nn as nn
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from torchvision.transforms import transforms
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from collections import defaultdict, namedtuple
from jumping_quadrupeds.ppo.buffer import OnPolicyReplayBuffer
from jumping_quadrupeds.drqv2.buffer import OffPolicyReplayBuffer
from jumping_quadrupeds.spr.buffer import OffPolicySequentialReplayBuffer
from torch.utils.data import IterableDataset

T_co = TypeVar("T_co", covariant=True)


DataSpec = namedtuple("DataSpec", ["name", "shape", "dtype"])
### FILE ADAPTED FROM OPENAI SPINNINGUP, https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo


def schedule(schdl, step, log_std):
    schdl = list(schdl.values())
    schdl = f"{schdl[0]}({float(schdl[1])}, {float(schdl[2])}, {int(schdl[3])})"

    match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
    if match:
        init, final, duration = [float(g) for g in match.groups()]
        mix = np.clip(step / duration, 0.0, 1.0)
        if step > duration and log_std is not None:
            return None

        return (1.0 - mix) * init + mix * final, duration


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def common_img_transforms(with_flip=False, size=84):
    out = [transforms.Resize(size)]
    if with_flip:
        out.append(transforms.RandomHorizontalFlip())
    out.append(transforms.ToTensor())
    return transforms.Compose(out)


def abs_path(x):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(x)))


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def preprocess_obs(obs, device):
    assert obs.dtype == np.uint8 or obs.dtype == torch.uint8
    if obs.dtype == np.uint8:
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
    obs = obs.to(device)
    obs = obs / 255.0 - 0.5
    return obs


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    return seed


class BufferedShuffleDataset(IterableDataset[T_co]):
    r"""Dataset shuffled from the original dataset.
    This class is useful to shuffle an existing instance of an IterableDataset.
    The buffer with `buffer_size` is filled with the items from the dataset first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.
    `buffer_size` is required to be larger than 0. For `buffer_size == 1`, the
    dataset is not shuffled. In order to fully shuffle the whole dataset, `buffer_size`
    is required to be greater than or equal to the size of dataset.
    When it is used with :class:`~torch.utils.data.DataLoader`, each item in the
    dataset will be yielded from the :class:`~torch.utils.data.DataLoader` iterator.
    And, the method to set up a random seed is different based on :attr:`num_workers`.
    For single-process mode (:attr:`num_workers == 0`), the random seed is required to
    be set before the :class:`~torch.utils.data.DataLoader` in the main process.
        >>> ds = BufferedShuffleDataset(dataset)
        >>> random.seed(...)
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
    For multi-process mode (:attr:`num_workers > 0`), the random seed is set by a callable
    function in each worker.
        >>> ds = BufferedShuffleDataset(dataset)
        >>> def init_fn(worker_id):
        ...     random.seed(...)
        >>> print(list(torch.utils.data.DataLoader(ds, ..., num_workers=n, worker_init_fn=init_fn)))
    Arguments:
        dataset (IterableDataset): The original IterableDataset.
        buffer_size (int): The buffer size for shuffling.
    """
    dataset: IterableDataset[T_co]
    buffer_size: int

    def __init__(self, dataset: IterableDataset[T_co], buffer_size: int) -> None:
        super(BufferedShuffleDataset, self).__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[T_co]:
        buf: List[T_co] = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()


def buffer_loader_factory(type=None, batch_size=None, **kwargs):
    def _worker_init_fn(worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(seed)

    if type == "on-policy":
        buffer = OnPolicyReplayBuffer(**kwargs)
    elif type == "off-policy":
        buffer = OffPolicyReplayBuffer(**kwargs)
    elif type == "off-policy-sequential":
        buffer = OffPolicySequentialReplayBuffer(**kwargs)
    else:
        raise ValueError(
            f"Unknown replay buffer name: {name}. Have you specified your buffer correctly, a la `--macro templates/buffer/ppo.yml'?"
        )
    breakpoint()

    buffer = BufferedShuffleDataset(buffer, buffer_size=batch_size * kwargs.get("num_workers"))

    loader = torch.utils.data.DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=kwargs.get("num_workers"),
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader

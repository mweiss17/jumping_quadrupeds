import os
import random
import re
import PIL
from collections import namedtuple
from typing import TypeVar, Iterator, List

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.utils.data import IterableDataset
from torchvision.transforms import transforms

from jumping_quadrupeds.buffer import OffPolicyReplayBuffer
from jumping_quadrupeds.ppo.buffer import OnPolicyReplayBuffer

# from jumping_quadrupeds.spr.buffer import OffPolicySequentialReplayBuffer

T_co = TypeVar("T_co", covariant=True)
DataSpec = namedtuple("DataSpec", ["name", "shape", "dtype"])


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


def fold_timesteps_and_channels_if_needed(embeddings, timesteps, folded):
    # fold the timesteps and channels together for linear processing
    if folded:
        return rearrange(embeddings, "(b t) c -> b (t c)", t=timesteps)
    else:
        return rearrange(embeddings, "t c -> (t c)")


def fold_timesteps_if_needed(obs):
    if len(obs.shape) == 3:
        obs = obs.unsqueeze(0)
    folded = False
    if len(obs.shape) == 5:
        folded = True
        obs = rearrange(obs, "b t c w h -> (b t) c w h")
    return obs, folded


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


def buffer_loader_factory(buffer_type=None, batch_size=None, **kwargs):
    def _worker_init_fn(worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(seed)

    if buffer_type == "on-policy":
        buffer = OnPolicyReplayBuffer(**kwargs)
        buffer_size = batch_size
    elif buffer_type == "off-policy":
        buffer = OffPolicyReplayBuffer(**kwargs)
        buffer_size = batch_size * kwargs.get("num_workers")
    elif buffer_type == "off-policy-sequential":
        buffer = OffPolicySequentialReplayBuffer(**kwargs)
        buffer_size = batch_size * kwargs.get("num_workers")

    else:
        raise ValueError(
            f"Unknown replay buffer name: {buffer_type}. Have you specified your buffer correctly, a la `--macro templates/buffer/ppo.yml'?"
        )
    if buffer_type == "off-policy":
        buffer = BufferedShuffleDataset(buffer, buffer_size=buffer_size)

    loader = torch.utils.data.DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=kwargs.get("num_workers"),
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )
    return loader


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def py2pil(a):
    return np2pil(py2np(a))


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a, mode=guess_mode(a))


def py2np(a):
    return a.permute(1, 2, 0).detach().cpu().numpy()


def guess_mode(data):
    if data.shape[-1] == 1:
        return "L"
    if data.shape[-1] == 3:
        return "RGB"
    if data.shape[-1] == 4:
        return "RGBA"
    raise ValueError("Un-supported shape for image conversion %s" % list(data.shape))


def convert_action_to_onehot(action, action_dim):
    onehot_converter = torch.eye(action_dim).to(device=action.device)
    action = onehot_converter[action.long()]
    return action

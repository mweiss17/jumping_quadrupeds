import os
import re
import random
import torch
import argparse
import numpy as np
import torch.nn as nn
from torchvision.transforms import transforms
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from collections import defaultdict, namedtuple
from jumping_quadrupeds.ppo.buffer import OnPolicyReplayBuffer
from jumping_quadrupeds.drqv2.buffer import OffPolicyReplayBuffer
from jumping_quadrupeds.spr.buffer import OffPolicySequentialReplayBuffer

DataSpec = namedtuple("DataSpec", ["name", "shape", "dtype"])
### FILE ADAPTED FROM OPENAI SPINNINGUP, https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


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


def common_img_transforms(with_flip=False):
    out = [transforms.Resize(64)]
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
        obs = torch.tensor(obs.copy(), dtype=torch.float32)
    obs = obs.to(device)
    obs = obs / 255.0
    return obs

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    return seed

def build_loader(replay_dir, name=None, batch_size=None, **kwargs):
    if name == "on-policy":
        iterable = OnPolicyReplayBuffer(replay_dir, **kwargs)
    elif name == "off-policy":
        iterable = OffPolicyReplayBuffer(replay_dir, **kwargs)
    elif name == "off-policy-sequential":
        iterable = OffPolicySequentialReplayBuffer(replay_dir, **kwargs)
    else:
        raise ValueError(
            f"Unknown replay buffer name: {name}. Have you specified your buffer correctly, a la `--macro templates/buffer/ppo.yml'?")

    def _worker_init_fn(worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(seed)

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=kwargs.get("num_workers"),
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return iter(loader)

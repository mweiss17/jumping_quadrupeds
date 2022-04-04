import math

import torch
from einops import repeat
from einops.layers.torch import Rearrange
from einops import rearrange

from torch import nn

from jumping_quadrupeds.utils import pair


class InputTokenizer(nn.Module):
    @property
    def output_dim(self):
        raise NotImplementedError


class SpatioTemporalTokenizer(InputTokenizer):
    def __init__(
        self,
        obs_space,
        dim,
        patch_size,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.num_timesteps = obs_space.shape[0]
        image_width = obs_space.shape[-1]
        channels = obs_space.shape[1]
        self.num_patches = image_width // patch_size
        self.patch_height, self.patch_width = pair(patch_size)

        self.patch_dim = channels * self.patch_height * self.patch_width
        self.patchify = Rearrange(
            "b s c (p1 h) (p2 w) -> b (s p1 p2) (h w c)",
            h=self.patch_height,
            w=self.patch_width,
            s=self.num_timesteps,
        )
        self.to_embedding = torch.nn.Linear(self.patch_dim, dim)

    def output_dim(self):
        return self.num_patches * self.repr_dim

    def vit_positional_encoding(self, n, dim, device):
        position = torch.arange(n, device=device).unsqueeze(1)
        div_term_even = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        div_term_odd = torch.exp(torch.arange(1, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(n, 1, dim, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term_even)
        pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
        return pe.transpose(0, 1)

    def forward(self, x):
        x = self.to_embedding(x)
        b, n, c = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vit_positional_encoding((n + 1), c, x.device)  # [:, : (n + 1)]

        return x

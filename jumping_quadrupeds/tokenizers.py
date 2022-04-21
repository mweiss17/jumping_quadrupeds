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
        pe_method: str = "vit-1d-pe",
        pe_basis: str = "sin_cos_xy",
        pe_max_freq: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.pe_method = pe_method
        self.pe_basis = pe_basis
        self.pe_max_freq = pe_max_freq

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
        in_dim = self.patch_dim
        if self.pe_basis == "sin_cos_xy" and self.pe_method == "nerf-2d-pe":
            in_dim = (channels + self.pe_max_freq * 2 * 2 + 2) * self.patch_width * self.patch_height
        if self.pe_method == "nerf-3d-pe-raw":
            in_dim = self.patch_dim * 2
        self.to_embedding = torch.nn.Linear(in_dim, dim)

    def output_dim(self):
        return self.num_patches * self.repr_dim

    def vit_pe_1d(self, n, device):
        position = torch.arange(n, device=device).unsqueeze(1)
        div_term_even = torch.exp(torch.arange(0, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim))
        div_term_odd = torch.exp(torch.arange(1, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(n, 1, self.dim, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term_even)
        pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
        return pe.transpose(0, 1)

    def nerf_pe(self, xy_coords, L=10, basis_function="sin_cos", device="cpu"):
        # L is max frequency. 2 * 2 * L = number of output dimensions
        if basis_function == "raw_xy":
            return xy_coords
        elif basis_function == "sin_cos":
            l = torch.arange(L, device=device).reshape(1, L, 1, 1)
            x = 2**l * torch.pi * xy_coords[:, 0:1]
            y = 2**l * torch.pi * xy_coords[:, 1:2]
            xy = torch.cat([x, y], 1)
            pe = torch.cat([torch.sin(xy), torch.cos(xy)], 1)
        elif basis_function == "sinc":
            l = torch.arange(L, device=device).reshape(1, L, 1, 1)
            x = 2**l * torch.pi * xy_coords[:, 0:1]
            y = 2**l * torch.pi * xy_coords[:, 1:2]
            xy = torch.cat([x, y], 1)
            pe = torch.cat([torch.special.sinc(xy), torch.special.sinc(xy + torch.pi / 2.0)], 1)
        elif basis_function == "sin_cos_xy":
            l = torch.arange(L, device=device).reshape(1, L, 1, 1)
            x = 2**l * torch.pi * xy_coords[:, 0:1]
            y = 2**l * torch.pi * xy_coords[:, 1:2]
            xy = torch.cat([x, y], 1)
            pe = torch.cat([torch.sin(xy), torch.cos(xy), xy_coords], 1)
        return pe

    def embed_x(self, x):
        x = self.to_embedding(x)
        b, n, c = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward(self, x):
        if self.pe_method == "vit-1d-pe":
            x = self.embed_x(self.patchify(x))
            x = x + self.vit_pe_1d(x.shape[1], x.device)
        elif self.pe_method == "nerf-2d-pe":
            xy_coords = xy_meshgrid(x.shape[-1], x.shape[-2], batch_size=x.shape[0], device=x.device)
            pos_enc = self.nerf_pe(xy_coords, L=self.pe_max_freq, basis_function=self.pe_basis, device=x.device)
            pos_enc = repeat(pos_enc, "b c h w -> b s c h w", s=self.num_timesteps)
            x = torch.cat([x, pos_enc], dim=2)
            x = self.embed_x(self.patchify(x))
        elif self.pe_method == "nerf-3d-pe-raw":
            xyz_coords = xyz_meshgrid(x.shape[-1], x.shape[-2], x.shape[1], batch_size=x.shape[0], device=x.device)
            x = torch.cat([x, xyz_coords], dim=2)
            x = self.embed_x(self.patchify(x))
        else:
            raise NotImplementedError

        return x


def xyz_meshgrid(
    width,
    height,
    depth,
    minval_x=-1.0,
    maxval_x=1.0,
    minval_y=-1.0,
    maxval_y=1.0,
    minval_z=-1.0,
    maxval_z=1.0,
    batch_size=1,
    device="cpu",
):
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.linspace(minval_x, maxval_x, width, device=device),
        torch.linspace(minval_y, maxval_y, height, device=device),
        torch.linspace(minval_z, maxval_z, depth, device=device),
        indexing="xy",
    )
    xyz_coords = torch.stack([x_coords, y_coords, z_coords], 0)
    xyz_coords = rearrange(xyz_coords, "c h w s -> s c h w")
    xyz_coords = repeat(xyz_coords, "s c h w -> b s c h w", b=batch_size)
    return xyz_coords


def xy_meshgrid(width, height, minval_x=-1.0, maxval_x=1.0, minval_y=-1.0, maxval_y=1.0, batch_size=1, device="cpu"):
    x_coords, y_coords = torch.meshgrid(
        torch.linspace(minval_x, maxval_x, width, device=device),
        torch.linspace(minval_y, maxval_y, height, device=device),
        indexing="xy",
    )
    xy_coords = torch.stack([x_coords, y_coords], 0)  # [2, height, width]
    xy_coords = torch.unsqueeze(xy_coords, 0)  # [1, 2, height, width]
    xy_coords = torch.tile(xy_coords, [batch_size, 1, 1, 1])
    return xy_coords  # +x right, +y down, xy-indexed

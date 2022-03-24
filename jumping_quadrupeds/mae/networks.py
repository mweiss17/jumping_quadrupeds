import torch
import torch.nn as nn
from jumping_quadrupeds.utils import TruncatedNormal, weight_init
import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import dill
import math
from itertools import cycle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# from torchmeta.datasets.helpers import miniimagenet
# from torchmeta.utils.data import NonEpisodicWrapper

from jumping_quadrupeds.eth.dataset import Hdf5ImgDataset
from jumping_quadrupeds.utils import common_img_transforms, abs_path
from torch.utils.data import RandomSampler, DataLoader, Subset

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange


# helpers
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob


def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    max_masked = math.ceil(prob * seq_len)

    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    new_mask = torch.zeros((batch, seq_len), device=device)
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()


class Encoder(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        assert len(obs_space.shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_space, feature_dim, hidden_dim, log_std, use_actor_ln):
        super().__init__()
        self.action_space = action_space
        self.low = action_space.low[0]
        self.high = action_space.high[0]
        if use_actor_ln:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        else:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.Tanh())

        self.log_std = None
        if log_std:
            self.log_std = torch.nn.Parameter(log_std * torch.ones(self.action_space.shape[0], dtype=torch.float32))

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.shape[0]),
        )

        self.apply(weight_init)

    def forward(self, obs, std=None):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)

        # If we want to learn the std, then we don't pass in a scheduled std
        if not std:
            std = torch.exp(self.log_std)

        # do it this way to backprop thru
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std, low=self.low, high=self.high)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_space, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_space.shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_space.shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForwardGEGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, encoder_nonlinearity="gelu", qkv_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        ffn = FeedForward if encoder_nonlinearity == "gelu" else FeedForwardGEGLU
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, qkv_bias=qkv_bias)
                        ),
                        PreNorm(dim, ffn(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


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
        channels,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        num_timesteps = obs_space.shape[0]
        image_width = obs_space.shape[-1]
        self.num_patches = (num_timesteps * image_width // patch_size) ** 2
        self.patch_height, self.patch_width = pair(patch_size)

        patch_dim = channels * self.patch_height * self.patch_width

        self.vectorize = Rearrange("b c h w -> b (h w) c")
        self.patchify = Rearrange("b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=self.patch_height, p2=self.patch_width)
        self.to_embedding = torch.nn.Linear(patch_dim, dim)
        self.unpatchify = Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_height, p2=self.patch_width)
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_height, p2=self.patch_width),
        #     nn.Linear(patch_dim, dim),
        # )

    def output_dim(self):
        return self.num_patches * self.repr_dim

    def vit_positional_encoding(self, n, dim):
        position = torch.arange(n, device=self.device).unsqueeze(1)
        div_term_even = torch.exp(torch.arange(0, dim, 2, device=self.device) * (-math.log(10000.0) / dim))
        div_term_odd = torch.exp(torch.arange(1, dim, 2, device=self.device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(n, 1, dim, device=self.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term_even)
        pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
        return pe.transpose(0, 1)

    def forward(self, x):
        x = self.vectorize(x)
        x = self.patchify(x)
        x = self.to_embedding(x)
        b, n, c = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vit_positional_encoding((n + 1), c)  # [:, : (n + 1)]

        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        obs_space,
        tokenizer,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        encoder_nonlinearity="gelu",
        use_last_ln=True,
        qkv_bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.obs_space = obs_space
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, encoder_nonlinearity, qkv_bias)

        if use_last_ln:
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch_dim))
        else:
            self.mlp_head = nn.Sequential(nn.Linear(dim, patch_dim))

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        mask_strat="all",
        device=None,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio
        self.mask_strat = mask_strat
        self.device = device

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.repr_dim = self.encoder.dim  # num_patches * self.encoder.dim

        # decoder parameters
        self.enc_to_dec = nn.Linear(self.encoder.dim, decoder_dim) if self.encoder.dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )

        self.decoder_pos_emb = nn.Embedding(self.encoder.tokenizer.num_patches, decoder_dim)

    def compute_masks(self, eval, batch):
        if eval:
            masked_indices = torch.arange(0, device=self.device)
            unmasked_indices = torch.arange(self.num_patches, device=self.device)
            num_masked = 0
        elif self.mask_strat == "all":
            num_masked = int(self.masking_ratio * self.num_patches)
            rand_indices = torch.rand(batch, self.num_patches, device=self.device).argsort(dim=-1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        return masked_indices, unmasked_indices, num_masked

    def forward(self, batch, eval=True):
        # breakpoint()
        # TODO:
        # 1. add temporal encoding
        # 2. stack images along patch dimension
        # 3. add masking options (next-frame prediction, all-frames prediction)
        # 4. want to have a specific framestacking dimension (will make life easier in MAE) render_reconstruction

        if len(batch.shape) == 4:
            batch = batch.unsqueeze(0)
        # patches = self.to_patch(img)
        # batch, num_patches, *_ = patches.shape
        #
        # # patch to encoder tokens and add positions
        # tokens = self.patch_to_emb(patches)
        # tokens = tokens + self.encoder.pos_embedding[:, 1 : (num_patches + 1)]
        breakpoint()
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        masked_indices, unmasked_indices, num_masked = self.compute_masks(eval, batch.shape[0])

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch.shape[0], device=self.device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)
        breakpoint()

        # use for control stuff
        global_code = encoded_tokens[:, 0]

        patches = self.mlp_head(encoded_tokens)

        # If we're in eval mode, we just return the encoded image
        if eval:
            return encoded_tokens.view(encoded_tokens.shape[0], -1)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss, pred_pixel_values, masked_indices, unmasked_indices, patches

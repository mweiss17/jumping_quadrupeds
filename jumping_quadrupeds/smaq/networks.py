import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.distributions.categorical import Categorical

from jumping_quadrupeds.utils import TruncatedNormal, weight_init


# from torchmeta.datasets.helpers import miniimagenet
# from torchmeta.utils.data import NonEpisodicWrapper


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


class DiscreteActor(nn.Module):
    def __init__(self, repr_dim, action_space, feature_dim, hidden_dim, log_std, use_actor_ln):
        super().__init__()
        self.action_space = action_space
        if use_actor_ln:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        else:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.Tanh())

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.n),
        )

        self.apply(weight_init)

    def forward(self, obs, std=None):
        h = self.trunk(obs)
        action_dist = self.policy(h)
        action_dist = F.softmax(action_dist / (std*10.0), dim=-1)  # at std=0.1 (min_val by default), no temp is applied
        action_dist = Categorical(action_dist)
        return action_dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
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
    def __init__(self, dim, mult=4, dropout=0.0):
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
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.0, qkv_bias=False):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim**-0.5

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


class ViT(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        head_dim,
        dropout=0.0,
        emb_dropout=0.0,
        gelu_mult=4,
        qkv_bias=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout, qkv_bias=qkv_bias)
                        ),
                        PreNorm(dim, FeedForwardGEGLU(dim, gelu_mult, dropout=dropout)),
                    ]
                )
            )
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.dropout(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SequentialMaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        encoder,
        decoder,
        action_dim,
        masking_ratio=0.75,
        device=None,
        use_cls_token=False,
        use_last_ln=False,
        auto_encoding_mask_type="random",
        state_encoding_mask_type="random",
        action_encoding_dim=16,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio
        self.auto_encoding_mask_type = auto_encoding_mask_type
        self.state_encoding_mask_type = state_encoding_mask_type
        self.use_cls_token = use_cls_token
        self.action_dim = action_dim
        self.device = device
        self.tokenizer = tokenizer
        self.total_num_patches = (self.tokenizer.num_patches**2) * self.tokenizer.num_timesteps
        self.enc = encoder
        self.dec = decoder

        # decoder parameters
        self.enc_to_dec = nn.Linear(self.enc.dim, self.dec.dim) if self.enc.dim != self.dec.dim else nn.Identity()
        self.project_action = nn.Linear(self.action_dim, action_encoding_dim)
        self.mask_token = nn.Parameter(torch.randn(self.dec.dim))
        self.dec_pos_emb = nn.Embedding(self.total_num_patches, self.dec.dim)

        # We can optionally use the cls token as input to the actor and critic
        if self.use_cls_token:
            self.out_dim = self.enc.dim
        elif not self.use_cls_token and self.state_encoding_mask_type == "smaq-uniform":
            self.out_dim = int((1 - self.masking_ratio) * self.total_num_patches) * self.enc.dim
        elif not self.use_cls_token and self.state_encoding_mask_type == "temporal-multinoulli":
            self.out_dim = int(self.total_num_patches // self.tokenizer.num_timesteps) * self.enc.dim
        else:
            raise ValueError("what")

        if use_last_ln:
            self.from_embedding = nn.Sequential(
                nn.LayerNorm(self.dec.dim), nn.Linear(self.dec.dim, self.tokenizer.patch_dim)
            )
        else:
            self.from_embedding = nn.Sequential(nn.Linear(self.dec.dim, self.tokenizer.patch_dim))

    def compute_masks(self, batch_size, seq_len, mask_type):

        if mask_type == "smaq-uniform":
            num_masked = int(self.masking_ratio * self.total_num_patches)
            rand_indices = torch.rand(batch_size, self.total_num_patches, device=self.device).argsort(dim=-1)
            masked_indices = rand_indices[:, :num_masked]
            unmasked_indices = rand_indices[:, num_masked:]
        elif mask_type == "temporal-multinoulli":
            num_masked = self.total_num_patches // seq_len
            dist = Categorical(torch.ones(seq_len, device=self.device) / seq_len)
            masked_indices = dist.sample((batch_size, num_masked))
            masked_indices = F.one_hot(masked_indices, num_classes=seq_len)
            masked_indices = rearrange(masked_indices, "b p s -> b (p s)")
            unmasked_indices = rearrange(
                (masked_indices - 1).nonzero()[:, 1], "(b m) -> b m", m=self.total_num_patches - num_masked
            )
            masked_indices = rearrange(masked_indices.nonzero()[:, 1], "(b m) -> b m", m=num_masked)
        elif mask_type == "nfp":
            num_masked = self.total_num_patches // seq_len
            indices = torch.arange(0, self.total_num_patches, device=self.device).repeat(batch_size, 1)
            unmasked_indices = indices[:, : self.total_num_patches - num_masked]
            masked_indices = indices[:, self.total_num_patches - num_masked :]
        elif mask_type == "hybrid-nfp-smaq-uniform":
            # initialize the mask
            num_patches_minus_last_f = self.total_num_patches - self.total_num_patches // seq_len
            num_to_mask = int(self.masking_ratio * (self.total_num_patches - self.total_num_patches // seq_len))

            # Mask smaq uniform
            rand_indices = torch.rand(batch_size, num_patches_minus_last_f, device=self.device).argsort(dim=-1)
            masked_indices = rand_indices[:, :num_to_mask]
            unmasked_indices = rand_indices[:, num_to_mask:]

            # Mask nfp and concatenate
            last_frame_patches = torch.arange(
                num_patches_minus_last_f, self.total_num_patches, device=self.device
            ).repeat(batch_size, 1)
            masked_indices = torch.cat([masked_indices, last_frame_patches], dim=1)
        elif mask_type == "None":
            masked_indices = torch.LongTensor([]).repeat(batch_size, 1)
            unmasked_indices = torch.arange(self.total_num_patches, device=self.device).repeat(batch_size, 1)

        else:
            raise ValueError("invalid mask type")

        if self.use_cls_token:
            cls_token_index = torch.ones([1], device=self.device, dtype=torch.bool).repeat(batch_size, 1)
            unmasked_indices = torch.cat([cls_token_index, unmasked_indices], dim=1)
        return masked_indices, unmasked_indices

    def forward(self, input, action=None, decode=False, mask_type="None"):
        # input: (batch_size, num_timesteps, c, h ,w)
        if len(input.shape) == 4:
            input = input.unsqueeze(0)
        if action is None:
            action = torch.zeros(input.shape[0], self.action_dim, device=self.device)

        masked_indices, unmasked_indices = self.compute_masks(
            batch_size=input.shape[0], seq_len=input.shape[1], mask_type=mask_type
        )
        num_masked = masked_indices.shape[1]
        batch_range = torch.arange(input.shape[0], device=self.device)[:, None]

        raw_patches = self.tokenizer.patchify(input)
        masked_patches = raw_patches[batch_range, masked_indices]

        embeds = self.tokenizer(input.clone())
        unmasked_embeds = embeds[batch_range, unmasked_indices]

        action = self.project_action(action.to(self.device))
        if len(action.shape) == 1:
            action = repeat(action, "c -> b t c", b=unmasked_embeds.shape[0], t=unmasked_embeds.shape[1])
        else:
            action = repeat(action, "b c -> b t c", t=unmasked_embeds.shape[1])
        unmasked_embeds = torch.cat((unmasked_embeds, action), dim=-1)

        # attend with vision transformer
        encoded_tokens = self.enc(unmasked_embeds)

        # Return an embedding for control
        if not decode:
            if self.use_cls_token:
                return encoded_tokens[:, 0]
            else:
                return encoded_tokens

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        decoder_tokens = decoder_tokens + self.dec_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=input.shape[0], n=num_masked)

        mask_tokens = mask_tokens + self.dec_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.dec(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.from_embedding(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss, pred_pixel_values, masked_indices, unmasked_indices, input

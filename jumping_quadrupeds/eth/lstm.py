import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList
import torch.nn.functional as F


class EthLstm(nn.Module):
    def __init__(
        self,
        lstm_layers=2,
        seq_len=200,
        batch_size=2,
        lstm_size=128,
        input_size=32,
        output_size=16,
    ):
        super().__init__()
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.lstms = ModuleList()
        self.output_dims = []
        for l in range(self.lstm_layers):
            input_dim = self.input_size if l == 0 else self.lstm_size
            output_dim = (
                self.output_size if l == self.lstm_layers - 1 else self.lstm_size
            )
            self.output_dims.append(output_dim)
            self.lstms.append(nn.LSTMCell(input_dim, output_dim))

        self.mu_net = nn.Linear(output_size, output_size)
        self.sigma_net = nn.Linear(output_size, output_size)

    def init_hidden(self, batch_size, device):
        self.ht = []
        self.ct = []

        self.h = []
        self.c = []
        for layer_idx in range(self.lstm_layers):
            h = torch.zeros((batch_size, self.output_dims[layer_idx]))
            c = torch.zeros((batch_size, self.output_dims[layer_idx]))
            torch.nn.init.xavier_normal_(h, gain=1.0)
            torch.nn.init.xavier_normal_(c, gain=1.0)
            self.h.append(h.to(device))
            self.c.append(c.to(device))
            self.ht.append(
                [
                    None,
                ]
                * self.seq_len
            )
            self.ct.append(
                [
                    None,
                ]
                * self.seq_len
            )

    def forward(self, vae_latent, frame_idx, displacement=None):
        # in theory, we should concatenate the movement info/odometry here
        # x = cat(vae_latent, displacement)
        x = vae_latent

        # step the LSTM layers
        for layer_idx, l in enumerate(self.lstms):
            input = x if layer_idx == 0 else self.ht[layer_idx - 1][frame_idx]
            if frame_idx == 0:
                self.ht[layer_idx][frame_idx], self.ct[layer_idx][frame_idx] = l(
                    input, (self.h[layer_idx], self.c[layer_idx])
                )
            else:
                self.ht[layer_idx][frame_idx], self.ct[layer_idx][frame_idx] = l(
                    input,
                    (
                        self.ht[layer_idx][frame_idx - 1],
                        self.ct[layer_idx][frame_idx - 1],
                    ),
                )

        lstm_mu = self.mu_net(self.ht[self.lstm_layers - 1][frame_idx])
        lstm_log_sigma = self.sigma_net(self.ht[self.lstm_layers - 1][frame_idx])

        return {
            "lstm_mu": lstm_mu,
            "lstm_log_sigma": lstm_log_sigma,
            "vae_latent": vae_latent,
        }

    def loss(self, lstm_mu, lstm_log_sigma, vae_latent):
        # make mu/sigma into distribution
        lstm_normal = Normal(lstm_mu, lstm_log_sigma.exp())
        sampled_next_latent = lstm_normal.rsample()  # reparameterization trick

        l2 = F.mse_loss(sampled_next_latent, vae_latent, size_average=False)

        kl = -0.5 * torch.sum(
            1 + 2 * lstm_log_sigma - lstm_mu.pow(2) - (2 * lstm_log_sigma).exp()
        )

        return {"l2": l2, "kl": kl, "loss": kl + l2}

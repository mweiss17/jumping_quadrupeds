import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList
import torch.nn.functional as F


class EthLstm(nn.Module):
    def __init__(self, lstm_layers=2, seq_len=200, batch_size=2, lstm_size=128, input_size=32, output_size=32):
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
            output_dim = self.output_size if l == self.lstm_layers - 1 else self.lstm_size
            self.output_dims.append(output_dim)
            self.lstms.append(nn.LSTMCell(input_dim, output_dim))

        self.mu_net = nn.Linear(output_size, output_size)
        self.sigma_net = nn.Linear(output_size, output_size)

        self.loss_func = nn.MSELoss()  # not sure about this # FIXME

    def init_hidden(self, device):
        self.ht = []
        self.ct = []

        self.h = []
        self.c = []
        for layer_idx in range(self.lstm_layers):
            h = torch.zeros((self.batch_size, self.output_dims[layer_idx]))
            c = torch.zeros((self.batch_size, self.output_dims[layer_idx]))
            torch.nn.init.xavier_normal_(h, gain=1.0)
            torch.nn.init.xavier_normal_(c, gain=1.0)
            self.h.append(h)
            self.c.append(c)
            self.ht.append([None, ] * self.seq_len)
            self.ct.append([None, ] * self.seq_len)

    # def init_hidden(self, device):
    #     self.h0 =
    #     self
    #     self.ht = []
    #     self.ct = []
    #     for layer_idx in range(self.lstm_layers):
    #         ht = torch.zeros(self.seq_len, self.batch_size, self.output_dims[layer_idx]).to(device)
    #         ct = torch.zeros(self.seq_len, self.batch_size, self.output_dims[layer_idx]).to(device)
    #         # torch.nn.init.xavier_normal_(ht, gain=1.0)
    #         # torch.nn.init.xavier_normal_(ct, gain=1.0)
    #         self.ht.append(ht)
    #         self.ct.append(ct)

    def forward(self, vae_latent, frame_idx, displacement=None):
        # in theory, we should concatenate the movement info/odometry here
        # x = cat(vae_latent, displacement)
        # FIXME
        x = vae_latent

        # step the LSTM layers

        for layer_idx, l in enumerate(self.lstms):
            input = x if layer_idx == 0 else self.ht[layer_idx - 1][frame_idx]
            if frame_idx == 0:
                self.ht[layer_idx][frame_idx], self.ct[layer_idx][frame_idx] = l(input, (self.h[layer_idx], self.c[layer_idx]))
            else:
                self.ht[layer_idx][frame_idx], self.ct[layer_idx][frame_idx] = l(input, (self.ht[layer_idx][frame_idx - 1], self.ct[layer_idx][frame_idx - 1]))

        last_idx = len(self.lstms) - 1
        h_next = self.ht[last_idx][frame_idx]
        mu = self.mu_net(h_next)
        logsigma = self.sigma_net(h_next)

        return mu, logsigma, h_next

    def loss(self, mu, logsigma, h_next, l_next_gt):
        # make mu/sigma into distribution
        sigma = logsigma.exp()
        norm = Normal(mu, sigma)
        l_next = norm.rsample()  # reparameterization trick

        l2 = self.loss_func(l_next, l_next_gt)

        sample = Normal(0, 1).sample([self.output_size])  # comparing to a unit gaussian
        kl = F.kl_div(norm.log_prob(h_next), sample, reduce="batchmean").mean()

        return {"l2": kl, "kl": kl, "loss":  kl}

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList
import torch.nn.functional as F


class EthLstm(nn.Module):
    def __init__(self, lstm_layers=2, lstm_size=128, input_size=32, output_size=32):
        super().__init__()
        self.output_size = output_size

        self.lstms = ModuleList()
        self.hidden_template = []
        self.cells_template = []
        for l in range(lstm_layers):
            input_dim = input_size if l == 0 else lstm_size
            output_dim = output_size if l == lstm_layers - 1 else lstm_size

            self.lstms.append(nn.LSTMCell(input_dim, output_dim))
            self.hidden_template.append(torch.zeros(output_dim))
            self.cells_template.append(torch.zeros(output_dim))
        self.hiddens = []
        self.cells = []

        self.mu_net = nn.Linear(output_size, output_size)
        self.sigma_net = nn.Linear(output_size, output_size)

        self.loss_func = nn.MSELoss()  # not sure about this # FIXME

    def reset_hidden(self, minibatch, device):
        self.hiddens = [h.expand(minibatch, -1).to(device) for h in self.hidden_template]
        self.cells = [c.expand(minibatch, -1).to(device) for c in self.cells_template]

    def forward(self, vae_latent, displacement=None):
        # in theory, we should concatenate the movement info/odometry here
        # x = cat(vae_latent, displacement)
        # FIXME
        x = vae_latent

        # step the LSTM layers
        # hidden_0, cell_0 = self.lstm_0(x, (hidden_0, cell_0))
        # hidden_1, cell_1 = self.lstm_0(hidden_0, (hidden_1, cell_1))
        for i, l in enumerate(self.lstms):
            lstm_input = x if i == 0 else self.hiddens[i - 1]
            self.hiddens[i], self.cells[i] = l(lstm_input, (self.hiddens[i], self.cells[i]))

        last_idx = len(self.lstms) - 1
        h_next = self.hiddens[last_idx]
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

        return {"l2": l2, "kl": kl, "loss": l2 + kl}

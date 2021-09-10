import torch
import torch.nn as nn
import torch.nn.functional as F
from vae import VectorVAE

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))
        return x
# The VAE has a latent space dimension of 32. The LSTM
# networks have 2 layers and a hidden state of dimension 128,
# which is mapped to the parameters of the latent distribution
# using two 1 layer MLPs with a linear activation. We train the
# networks using the Adam [43] optimizer with a learning rate
# of 1e-3.

class JumpingQuadruped(nn.Module):
    def __init__(self, input_size, output_size):
        super(JumpingQuadruped, self).__init__()
        vae_latent_dim = 100
        lstm_input_dim = 2 * vae_latent_dim
        lstm_output_dim = input_size
        self.mlp = MLP(input_size=input_size, output_size=output_size)
        self.vae = VanillaVAE(in_channels=3, latent_dim=vae_latent_dim)
        # TODO : put back to 2x vae_latent dim when we start using VAE
        self.lstm = torch.nn.LSTM(input_size, lstm_output_dim, 1) # input size, hidden size, num layers
        self.h_n = torch.randn(1, 1, lstm_output_dim)
        self.c_n = torch.randn(1, 1, lstm_output_dim)

    def forward(self, xs):

        # No VAE while we just use this 24-dim obs on bipedalwalker
        # # decoded image, input image, latent variable z's mu and sigma
        # decoded, input, mu_z, log_var_z = self.vae(torch.FloatTensor(x))
        #
        # # cat them together and get another dim for the LSTM
        # latent = torch.cat([mu_z, log_var_z], dim=1)
        # xs = latent.unsqueeze(1)
        for x in xs:
            # run the latent through the LSTM
            if len(x.shape) < 3:
                x = x.unsqueeze(0).unsqueeze(0)
            output, (self.h_n, self.c_n) = self.lstm(x, (self.h_n, self.c_n))
        # process the lstm's hidden state to get an action
        action = self.mlp(output)
        return action[0][0]


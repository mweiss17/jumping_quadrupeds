import torch
import torch.nn as nn
import torch.nn.functional as F


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

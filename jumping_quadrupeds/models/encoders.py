import torch
from torch.nn import functional as F
from typing import List, Any, TypeVar
from torch import nn


class WorldModelsConvEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """World Models' encoder"""

    def __init__(self, channels, activation=nn.ReLU):
        super(WorldModelsConvEncoder, self).__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        # self.conv5 = nn.Conv2d(256, 256, 2, stride=1) # for 84x84
        self.flatten = nn.Flatten()

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        # x = self.activation(self.conv5(x))
        x = self.flatten(x)
        return x


class FlosConvEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Florian's Encoder"""

    def __init__(self, channels, activation=nn.ReLU):
        super(FlosConvEncoder, self).__init__()

        self.mu_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            # [(64−8+0)/4]+1 = 15
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # [(15−4+0)/2]+1 = 6
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # [(6−3+0)]+1 = 4
            activation(),
            nn.Flatten(),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.mu_net(x)
        return x

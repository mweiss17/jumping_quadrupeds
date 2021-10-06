import torch
from torch.nn import functional as F
from typing import List, Any, TypeVar
from torch import nn


class ConvEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """World Models' encoder"""

    def __init__(self, channels):
        super(ConvEncoder, self).__init__()
        self.img_channels = channels
        self.conv1 = nn.Conv2d(channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        return x


class ConvEncoder2(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Florian's Encoder"""

    def __init__(self, channels, activation=nn.ReLU):
        super(ConvEncoder2, self).__init__()

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
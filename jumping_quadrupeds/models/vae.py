import torch
from torch.nn import functional as F
from typing import List, Any, TypeVar

Tensor = TypeVar("torch.tensor")
from torch import nn
from abc import abstractmethod

## All credit to https://github.com/AntixK/PyTorch-VAE


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


#
# class ConvVAE(BaseVAE):
#     def __init__(
#         self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs
#     ) -> None:
#         super(ConvVAE, self).__init__()
#
#         self.latent_dim = latent_dim
#
#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]
#
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(
#                         in_channels,
#                         out_channels=h_dim,
#                         kernel_size=3,
#                         stride=2,
#                         padding=1,
#                     ),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU(),
#                 )
#             )
#             in_channels = h_dim
#
#         self.encoder = nn.Sequential(*modules)
#         self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
#
#         # Build Decoder
#         modules = []
#
#         self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
#
#         hidden_dims.reverse()
#
#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(
#                         hidden_dims[i],
#                         hidden_dims[i + 1],
#                         kernel_size=3,
#                         stride=2,
#                         padding=1,
#                         output_padding=1,
#                     ),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU(),
#                 )
#             )
#
#         self.decoder = nn.Sequential(*modules)
#
#         self.final_layer = nn.Sequential(
#             nn.ConvTranspose2d(
#                 hidden_dims[-1],
#                 hidden_dims[-1],
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             ),
#             nn.BatchNorm2d(hidden_dims[-1]),
#             nn.LeakyReLU(),
#             nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
#             nn.Sigmoid(),
#         )
#
#     def encode(self, input: Tensor) -> List[Tensor]:
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)
#
#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)
#
#         return [mu, log_var]
#
#     def decode(self, z: Tensor) -> Tensor:
#         """
#         Maps the given latent codes
#         onto the image space.
#         :param z: (Tensor) [B x D]
#         :return: (Tensor) [B x C x H x W]
#         """
#         result = self.decoder_input(z)
#         result = result.view(-1, 512, 2, 2)
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result
#
#     def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
#
#     def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         mu, log_var = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return [self.decode(z), input, mu, log_var]
#
#     def loss_function(self, *args, **kwargs) -> dict:
#         """
#         Computes the VAE loss function.
#         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]
#
#         kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
#         recons_loss = F.mse_loss(recons, input, size_average=False)
#
#         # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
#         # loss = recons_loss + kld_loss * kld_weight
#         kld_loss = -0.5 * torch.sum(1 + 2 * log_var - mu.pow(2) - (2 * log_var).exp())
#         loss = recons_loss + kld_loss
#
#         return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": -kld_loss}
#
#     def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
#         """
#         Samples from the latent space and return the corresponding
#         image space map.
#         :param num_samples: (Int) Number of samples
#         :param current_device: (Int) Device to run the model
#         :return: (Tensor)
#         """
#         z = torch.randn(num_samples, self.latent_dim)
#
#         z = z.to(current_device)
#
#         samples = self.decode(z)
#         return samples
#
#     def generate(self, x: Tensor, **kwargs) -> Tensor:
#         """
#         Given an input image x, returns the reconstructed image
#         :param x: (Tensor) [B x C x H x W]
#         :return: (Tensor) [B x C x H x W]
#         """
#
#         return self.forward(x)[0]
#


class Decoder(nn.Module):
    """VAE decoder"""

    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """VAE encoder"""

    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        # self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_logsigma = nn.Linear(2 * 2 * 256, latent_size)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class ConvVAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, img_channels, latent_size):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logsigma):
        """VAE loss function"""
        BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return {"BCE": BCE, "KLD": KLD, "loss": BCE + KLD}

    def forward(self, x):  # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma

import os
from os.path import expanduser, expandvars
import sys
import time
import torch
from tqdm import tqdm
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import RandomSampler, DataLoader, Subset

# to add jumping_quadrupeds to python path
sys.path.append(os.getcwd())

from jumping_quadrupeds.models.vae import ConvVAE
from jumping_quadrupeds.models.encoders import WorldModelsConvEncoder, FlosConvEncoder
from jumping_quadrupeds.dataset import Hdf5ImgDataset
from jumping_quadrupeds.utils import common_img_transforms, abs_path

# pip install -e speedrun from https://github.com/inferno-pytorch/speedrun
from speedrun import (
    BaseExperiment,
    WandBSweepMixin,
    IOMixin,
    SweepRunner,
    register_default_dispatch,
)

import submitit
from submitit.core.utils import CommandFunction


class TrainVAE(
    WandBSweepMixin, IOMixin, submitit.helpers.Checkpointable, BaseExperiment
):
    WANDB_ENTITY = "jumping_quadrupeds"
    WANDB_PROJECT = "vae-tests"

    def __init__(self):
        super(TrainVAE, self).__init__()

        self.auto_setup()

        if self.get("use_wandb"):
            self.initialize_wandb()

        self.transform = {
            "train": common_img_transforms(with_flip=True),
            "valid": common_img_transforms(),
        }

        self._build_dataset()
        self._build_model()
        self._build_optimizer()

    def _build_dataset(self):
        ds_path = abs_path(expanduser(expandvars(self.get("paths/rollouts"))))

        train = Hdf5ImgDataset(ds_path, transform=self.transform["train"], flat=True)
        valid = Hdf5ImgDataset(ds_path, transform=self.transform["valid"], flat=True)

        split_percent = 0.8
        num_samples = train.episodes * train.steps
        sample_indices = np.arange(num_samples)
        mid = int(num_samples * split_percent)
        num_samples_train = sample_indices[0:mid]
        num_samples_valid = sample_indices[mid:num_samples]

        train_ds = Subset(train, num_samples_train)
        sampler = RandomSampler(train_ds)
        self.train = DataLoader(train_ds, sampler=sampler, **self.get("dataloader"))

        valid_ds = Subset(valid, num_samples_valid)
        sampler = RandomSampler(valid_ds)
        self.valid = DataLoader(valid_ds, sampler=sampler, **self.get("dataloader"))

    def _build_model(self):
        img_channels = 3
        if self.get("encoder_type") == "flo":
            encoder = FlosConvEncoder(img_channels)
        else:
            encoder = WorldModelsConvEncoder(img_channels)
        self.model = ConvVAE(encoder, img_channels=img_channels, latent_size=32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.get("lr"))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

    def plot_samples(self, true_image, generated_image):
        if self.get("use_wandb"):
            img = np.zeros((3, 64, 64 * 2 + 2), dtype=np.float32)
            img[:, :, :64] = np.rollaxis(true_image, 2, 0)
            img[:, :, 66:] = np.rollaxis(generated_image, 2, 0)
            self.wandb_log_image("L: true, R: generated", img)
        else:
            plt.imsave(
                os.path.join(
                    self.experiment_directory, f"Logs/sample-{self.epoch}.jpg"
                ),
                img,
            )

    @register_default_dispatch
    def __call__(self):
        if os.path.isfile(self.checkpoint_path):
            self.model = self.model.load_state_dict(
                torch.load(self.checkpoint_path)["model"]
            )

        for epoch in self.progress(range(self.get("num_epochs")), desc="epochs..."):
            # Train the model
            for imgs in tqdm(self.train, desc="batches..."):
                imgs = imgs.to(self.device)
                x_hat, mu, log_var = self.model(imgs)
                loss = self.model.loss_function(x_hat, imgs, mu, log_var)
                self.optimizer.zero_grad()
                loss["loss"].backward()
                self.optimizer.step()
                self.next_step()
                if self.get("use_wandb"):
                    self.wandb_log(
                        **{
                            "train_loss": loss,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )

            self.next_epoch()
            self.scheduler.step(loss["loss"])
            # get a sample
            sampleid = np.random.choice(range(0, len(imgs)))
            true_image = imgs[sampleid].detach().cpu().moveaxis(0, 2).numpy()
            generated_image = x_hat[sampleid].detach().cpu().moveaxis(0, 2).numpy()
            self.plot_samples(true_image, generated_image)

            if self.get("use_wandb"):
                # log gradients once per epoch
                self.wandb_watch(self.model, loss, log_freq=1)

                # record latent variables for this sample
                self.wandb_log(
                    **{
                        "mu": mu[sampleid],
                        "var": torch.exp(0.5 * log_var)[sampleid],
                    }
                )

            self.checkpoint_if_required()

            # Run Validation
            if epoch % self.get("valid_every") == 0:
                self.model.eval()

                for imgs in tqdm(self.valid, "valid batches..."):
                    imgs = imgs.to(self.device)
                    x_hat, mu, log_var = self.model(imgs)
                    valid_loss = self.model.loss_function(x_hat, imgs, mu, log_var)
                    if self.get("use_wandb"):
                        self.wandb_log(**{"valid_loss": valid_loss})
                        self.wandb_log(**{"lr": self.optimizer.param_groups[0]["lr"]})

                self.model.train()
        return valid_loss

    def checkpoint_if_required(self):
        if self.epoch % self.get("checkpoint_every") == 0:
            info = {
                "encoder": self.model.encoder.state_dict(),
                "model": self.model.state_dict(),
            }
            torch.save(info, f"{self.experiment_directory}/Weights/vae-{self.epoch}.pt")


if __name__ == "__main__":
    # Default cmdline args Flo
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "experiments/vae", "--inherit", "templates/vae"]

    TrainVAE().run()

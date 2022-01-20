import os
import sys
import torch
from speedrun import BaseExperiment, WandBMixin, IOMixin, WandBSweepMixin
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from jumping_quadrupeds.dataset import Hdf5ImgSeqDataset
from jumping_quadrupeds.eth.lstm import EthLstm
from jumping_quadrupeds.eth.vae import ConvVAE
from jumping_quadrupeds.eth.encoders import WorldModelsConvEncoder, FlosConvEncoder
from jumping_quadrupeds.utils import common_img_transforms, abs_path


class TrainLSTM(WandBMixin, IOMixin, BaseExperiment):
    WANDB_ENTITY = "jumping_quadrupeds"
    WANDB_PROJECT = "lstm-tests"
    WANDB_GROUP = "lstm-exploration"

    def __init__(self):
        super().__init__()
        self.auto_setup()

        if self.get("use_wandb"):
            self.initialize_wandb()

        self.transforms = {
            "train": common_img_transforms(with_flip=True),
            "valid": common_img_transforms(),
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load dataset
        ds = Hdf5ImgSeqDataset(
            abs_path(self.get("paths/rollouts")), self.transforms["train"]
        )  # FIXME: this applies training transforms to both
        # TODO: cut down sequence length - make 4 segments of 50 frames each or summin like that
        split = int(len(ds) * 0.8)
        ds_train_raw, ds_valid_raw = torch.utils.data.random_split(
            ds, [split, len(ds) - split]
        )

        self.train = DataLoader(ds_train_raw, **self.get("dataloader"))
        self.valid = DataLoader(ds_valid_raw, **self.get("dataloader"))

        # load VAE
        img_channels = 3
        if self.get("encoder_type") == "flo":
            encoder = FlosConvEncoder(img_channels)
        else:
            encoder = WorldModelsConvEncoder(img_channels)

        self.vae = ConvVAE(
            encoder, img_channels=img_channels, latent_size=self.get("vae/latent")
        ).to(self.device)

        self.vae.load_state_dict(
            torch.load(self.get("paths/vae_weights"), map_location=self.device)["model"]
        )

        self.lstm = EthLstm(**self.get("lstm")).to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.lstm.parameters(), lr=self.get("lr")
        )  # apply optimizer only to LSTM

    def run(self):
        for epoch in trange(self.get("num_epochs"), desc="epochs..."):
            self.train_loop()
            self.valid_loop()

    def step_model(self, frames, train=True):
        batch_len, seq_len, channels, w, h = frames.size()
        latents = []
        results = {}

        for frame_idx in trange(seq_len, desc="steps..."):
            _, mu_vae, logsigma_vae = self.vae(frames[:, frame_idx])
            norm = Normal(
                mu_vae, logsigma_vae.exp()
            )  # I think we need sigma here, not logsigma

            # this is the target for the LSTM in the next timestep
            vae_latent = (
                norm.rsample()
            )  # reparameterization trick so we can differentiate through

            # Not the Last timestep, because on the last step we
            # don't have a loss/target for the the LSTM that's predicting the future
            if frame_idx < seq_len - 1:
                latents.append(self.lstm(vae_latent, frame_idx))
            # First timestep, don't train
            if frame_idx == 0:
                continue

        if train:
            accumulated_loss = defaultdict(list)
            for latent in latents:
                loss = self.lstm.loss(**latent)
                for k, v in loss.items():
                    accumulated_loss[k].append(v)
            torch.stack(accumulated_loss["loss"]).mean().backward(retain_graph=True)

            self.optimizer.step()
            self.optimizer.zero_grad()

            for k, v in accumulated_loss.items():
                results[k] = torch.mean(torch.stack(v).detach())
        results["latents"] = latents
        return results

    def train_loop(self):
        self.lstm.train()
        self.vae.train()
        for batch in tqdm(self.train, desc="batches..."):
            batch = batch.to(self.device)
            self.lstm.init_hidden(batch.size()[0], self.device)

            state = self.step_model(batch, train=True)

            self.next_step()

            # TODO step scheduler
            # self.scheduler.step(loss["loss"])
            # TODO: log
            self.wandb_log(**state)

    def valid_loop(self):
        self.lstm.eval()
        self.vae.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid, desc="validation...")):
                batch = batch.to(self.device)

                self.lstm.init_hidden(batch.size()[0], self.device)

                state = self.step_model(batch, train=False)
            self.reconstruct_latents(batch, state["latents"])
            self.reconstruct_latents(batch, state["latents"], frame_idx=100)

    def reconstruct_latents(self, gt, latents, frame_idx=0):
        lstm_latent = torch.cat(
            [
                latents[frame_idx]["lstm_mu"],
                latents[frame_idx]["lstm_log_sigma"],
            ],
            dim=1,
        )
        self.write_sample(
            gt[0][frame_idx],
            latents[frame_idx]["vae_latent"],
            lstm_latent,
            frame_idx,
        )

    def write_sample(self, gt, vae_latent, lstm_latent, frame_idx):
        with torch.no_grad():
            plt.imsave(
                os.path.join(
                    self.experiment_directory,
                    f"Logs/gt-frame-{frame_idx}-step-{self.step}.png",
                ),
                gt.moveaxis(0, 2).cpu().numpy(),
            )
            plt.imsave(
                os.path.join(
                    self.experiment_directory,
                    f"Logs/vae-frame-{frame_idx}-step-{self.step}.png",
                ),
                self.vae.decoder(vae_latent)[0].cpu().moveaxis(0, 2).numpy(),
            )
            plt.imsave(
                os.path.join(
                    self.experiment_directory,
                    f"Logs/lstm-frame-{frame_idx}-step-{self.step}.png",
                ),
                self.vae.decoder(lstm_latent)[0].cpu().moveaxis(0, 2).numpy(),
            )


if __name__ == "__main__":
    # Default cmdline args Flo
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "experiments/lstm",
            "--inherit",
            "templates/lstm",
        ]

    TrainLSTM().run()

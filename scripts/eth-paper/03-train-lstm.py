import os
import sys
import torch
from speedrun import BaseExperiment, WandBMixin, IOMixin, WandBSweepMixin
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from jumping_quadrupeds.models.dataset import Hdf5ImgSeqDataset
from jumping_quadrupeds.models.lstm import EthLstm
from jumping_quadrupeds.models.vae import ConvVAE
from jumping_quadrupeds.encoders import WorldModelsConvEncoder, FlosConvEncoder
from jumping_quadrupeds.utils import common_img_transforms, abs_path


class TrainLSTM(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super().__init__()
        self.auto_setup()
        # TODO reenable wandb once everything is working
        # WandBSweepMixin.WANDB_ENTITY = "jumping_quadrupeds"
        # WandBSweepMixin.WANDB_PROJECT = "lstm-tests"
        # WandBSweepMixin.WANDB_GROUP = "lstm-exploration"
        #
        # if self.get("use_wandb"):
        #     self.initialize_wandb()

        self.transforms = {"train": common_img_transforms(with_flip=True), "valid": common_img_transforms()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load dataset
        ds_path = abs_path(self.get("paths/rollouts"))
        ds = Hdf5ImgSeqDataset(ds_path, self.transforms["train"])  # FIXME: this applies training transforms to both

        # TODO: cut down sequence length - make 4 segments of 50 frames each or summin like that
        split = int(len(ds) * 0.8)
        ds_train_raw, ds_valid_raw = torch.utils.data.random_split(ds, [split, len(ds) - split])

        self.train = DataLoader(ds_train_raw, **self.get("dataloader"))
        self.valid = DataLoader(ds_valid_raw, **self.get("dataloader"))

        latent_size = self.get("vae/latent")

        # load VAE
        img_channels = 3
        if self.get("encoder_type") == "flo":
            encoder = FlosConvEncoder(img_channels)
        else:
            encoder = WorldModelsConvEncoder(img_channels)
        self.vae = ConvVAE(encoder, img_channels=img_channels, latent_size=latent_size).to(self.device)

        # load VAE weights
        self.vae.load_state_dict(torch.load(self.get("paths/vae_weights"), map_location=self.device)['model'])
        self.vae.eval()

        # load LSTM
        self.lstm = EthLstm(lstm_layers=2, seq_len=200, batch_size=self.get("dataloader/batch_size"), lstm_size=128, input_size=latent_size, output_size=latent_size)

        # optimizer
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.get("lr"))  # apply optimizer only to LSTM

    def run(self):
        for epoch in trange(self.get("num_epochs"), desc="epochs..."):
            self.train_loop()
            self.valid_loop()

    def valid_loop(self):
        self.lstm.eval()
        self.vae.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid, desc="validation...")):
                loss, (mu_lstm, sigma_lstm, l_0, l_1) = self.step_model(batch, train=False)
                self.write_sample(batch[0, 100], l_0.view(2, -1).detach(), mu_lstm.view(2, -1))

    def step_model(self, frames, train=True):
        torch.autograd.set_detect_anomaly(True)
        batch_len, seq_len, channels, w, h = frames.size()
        for frame_idx in trange(seq_len, desc="steps..."):
            _, mu_vae, logsigma_vae = self.vae(frames[:, frame_idx])
            norm = Normal(mu_vae, logsigma_vae.exp())  # I think we need sigma here, not logsigma

            # this is the target for the LSTM in the next timestep
            l_0 = norm.rsample()  # reparameterization trick so we can differentiate through

            # Not the first timestep, then compute loss between cur vae output and prev lstm output
            if frame_idx != 0:
                # compute loss of vae latent at t=T against lstm latent at t=T-1
                loss = self.lstm.loss(mu_lstm, sigma_lstm, l_1, l_0)

            # Not the Last timestep, because on the last step we
            # don't have a loss/target for the the LSTM that's predicting the future
            if frame_idx < seq_len - 1:
                mu_lstm, sigma_lstm, l_1 = self.lstm(l_0, frame_idx)

            # First timestep, don't train
            if frame_idx == 0:
                continue

            if train:

                loss["loss"].backward(retain_graph=True)  # accumulate gradient over all timesteps in rollout

                if (frame_idx + 1) % self.get("lstm/backprop_after") == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # TODO add wandb tracking of different loss parts

        return loss, (mu_lstm, sigma_lstm, l_0, l_1)

    def train_loop(self):
        self.lstm.train()
        self.vae.train()
        for batch in tqdm(self.train, desc="batches..."):
            batch = batch.to(self.device)
            self.lstm.init_hidden(self.device)

            loss, latents = self.step_model(batch, train=True)

            # TODO step scheduler
            # self.scheduler.step(loss["loss"])
            # TODO: log
            self.next_step()

            # TODO step scheduler
            # self.scheduler.step(loss["loss"])
            # TODO: log
            print(loss)

    def write_sample(self, gt, reconstruction_1, reconstruction_2):
        with torch.no_grad():
            plt.imsave(
                os.path.join(self.experiment_directory, f"Logs/gt-{self.step}.png"),
                gt.moveaxis(0, 2).cpu().numpy(),
            )
            plt.imsave(
                os.path.join(self.experiment_directory, f"Logs/vae-{self.step}.png"),
                self.vae.decoder(reconstruction_1)[0].cpu().moveaxis(0, 2).numpy(),
            )
            plt.imsave(
                os.path.join(self.experiment_directory, f"Logs/lstm-{self.step}.png"),
                self.vae.decoder(reconstruction_2)[0].cpu().moveaxis(0, 2).numpy(),
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


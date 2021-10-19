import torch
from speedrun import BaseExperiment, WandBMixin, IOMixin, WandBSweepMixin
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from jumping_quadrupeds.models.dataset import Hdf5ImgSeqDataset
from jumping_quadrupeds.models.lstm import EthLstm
from jumping_quadrupeds.models.vae import ConvVAE
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
        self.vae = ConvVAE(img_channels=3, latent_size=latent_size).to(self.device)

        # load VAE weights
        self.vae.load_state_dict(torch.load(self.get("paths/vae_weights")))
        self.vae.eval()

        # load LSTM
        self.lstm = EthLstm(lstm_layers=2, lstm_size=128, input_size=latent_size, output_size=latent_size)

        # optimizer
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.get("lr"))  # apply optimizer only to LSTM

    def run(self):
        for epoch in trange(self.get("num_epochs"), desc="epochs..."):  # TODO
            self.train_loop()

            # TODO: test loop
            # TODO: write sample

    def train_loop(self):
        for imgs in tqdm(self.train, desc="batches..."):
            imgs = imgs.to(self.device)
            minibatch = len(imgs[:])
            frames = len(imgs[0, :])
            self.lstm.reset_hidden(minibatch, self.device)

            mu_lstm = None
            sigma_lstm = None
            l_1 = None
            loss_sum = 0

            self.optimizer.zero_grad()

            for frame in trange(frames, desc="steps..."):
                _, mu_vae, logsigma_vae = self.vae(imgs[:, frame])
                # FIXME we need to sample here, right? just confirming
                norm = Normal(mu_vae, logsigma_vae.exp())  # I think we need sigma here, not logsigma

                # this is the target for the LSTM in the next timestep
                l_0 = norm.rsample()  # reparameterization trick so we can differentiate through

                if mu_lstm is not None:
                    loss = self.lstm.loss(mu_lstm, sigma_lstm, l_1, l_0)
                    loss["loss"].backward(retain_graph=True)  # accumulate gradient over all timesteps in rollout
                    loss_sum += loss["loss"].item()

                    if frame + 1 % self.get("lstm/backprop_after") == 0:
                        self.optimizer.step()

                    # TODO add wandb tracking of different loss parts

                if frame < frames - 1:
                    # because on the very last step,
                    # we don't have a loss/target for the the LSTM that's predicting the future
                    mu_lstm, sigma_lstm, l_1 = self.lstm(l_0)

            # just to make sure we're not missing anything at the end
            self.optimizer.step()

            # TODO step scheduler
            # self.scheduler.step(loss["loss"])
            # TODO: log
            self.next_step()

    def write_sample(self):
        # TODO use VAE reconstruction to compare
        # reconstruction(lstm's predicted latent) to
        # reconstruction(next observed latent)
        pass


TrainLSTM().run()

import torch
import os
from tqdm import tqdm
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from jumping_quadrupeds.models.dataloader import Dataset
from jumping_quadrupeds.models.vae import ConvVAE
from speedrun import BaseExperiment, WandBMixin, IOMixin


class TrainVAE(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(TrainVAE, self).__init__()
        self.auto_setup()
        WandBMixin.WANDB_ENTITY = "jumping_quadrupeds"
        WandBMixin.WANDB_PROJECT = "vae-tests"
        WandBMixin.WANDB_GROUP = "vae-exploration"

        if self.get("use_wandb"):
            self.initialize_wandb()

        self._build()

    def _build(self):
        # CUDA for PyTorch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Generators
        dataset = Dataset(
            self.get("paths/rollouts"),
            max_num_samples=self.get("max_num_samples", None),
        )

        split = dataset.data.shape[0] // 8
        self.train = torch.utils.data.DataLoader(
            dataset[split:], **self.get("dataloader")
        )
        self.valid = torch.utils.data.DataLoader(
            dataset[:split], **self.get("dataloader")
        )

        self.model = ConvVAE(in_channels=3, latent_dim=32)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def run(self):
        # train loop
        for epoch in tqdm(range(self.get("num_epochs")), desc="epochs..."):

            # Train the model
            for batch in tqdm(self.train, desc="batches..."):
                batch = batch.to(self.device)
                x_hat, input, mu, log_var = self.model(batch)
                loss = self.model.loss_function(
                    x_hat, input, mu, log_var, M_N=self.get("dataloader/batch_size")
                )
                self.optimizer.zero_grad()
                loss["loss"].backward()
                self.optimizer.step()
                self.next_step()
                if self.get("use_wandb"):
                    self.wandb_log(**{"train_loss": loss})
                    self.wandb_log(**{"lr": self.scheduler.get_lr()[0]})

            self.next_epoch()

            # log gradients once per epoch
            if self.get("use_wandb"):
                self.wandb_watch(self.model, loss, log_freq=1)

            self.scheduler.step()

            # Plot samples
            sampleid = np.random.choice(range(0, len(batch)))
            true_image = batch[sampleid].detach().cpu().moveaxis(0, 2).numpy()
            generated_image = x_hat[sampleid].detach().cpu().moveaxis(0, 2).numpy()
            if self.get("use_wandb"):
                self.wandb_log_image("true image", np.rollaxis(true_image, 2, 0))
                self.wandb_log_image(
                    "generated image", np.rollaxis(generated_image, 2, 0)
                )
            else:
                true_sample_path = os.path.join(
                    self.experiment_directory,
                    f"Logs/true-epoch{epoch}.jpg",
                )
                generated_sample_path = os.path.join(
                    self.experiment_directory,
                    f"Logs/generated-epoch{epoch}.jpg",
                )
                plt.imsave(true_sample_path, true_image)
                plt.imsave(generated_sample_path, generated_image)

            # Checkpoint
            if epoch % self.get("checkpoint_every") == 0:
                torch.save(
                    self.model.state_dict(),
                    open(f"{self.experiment_directory}/Weights/vae-{epoch}.pt", "wb"),
                )

            # Run Validation
            if epoch % self.get("valid_every") == 0:
                self.model.eval()
                for valbatch in tqdm(self.valid, "valid batches..."):
                    valbatch = valbatch.to(self.device)
                    x_hat, input, mu, log_var = self.model(valbatch)
                    loss = self.model.loss_function(
                        x_hat, input, mu, log_var, M_N=self.get("dataloader/batch_size")
                    )
                    if self.get("use_wandb"):
                        self.wandb_log(**{"valid_loss": loss})
                        self.wandb_log(**{"lr": self.scheduler.get_lr()})

                self.model.train()


TrainVAE().run()

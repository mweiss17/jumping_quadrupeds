import torch
import os
import sys
from tqdm import tqdm
import numpy as np
from torch import optim
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder

sys.path.append(os.getcwd())

from jumping_quadrupeds.models.vae import ConvVAE

# pip install -e speedrun from https://github.com/inferno-pytorch/speedrun
from speedrun import BaseExperiment, WandBMixin, IOMixin

class TrainVAE(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(TrainVAE, self).__init__()
        self.auto_setup()
        WandBMixin.WANDB_PROJECT = "jumping_quadrupeds"
        WandBMixin.WANDB_ENTITY = "jumping_quadrupeds"
        if self.get("use_wandb"):
            self.initialize_wandb()

        # Make output dirs
        os.makedirs(self.get("paths/samples"), exist_ok=True)
        os.makedirs(self.get("paths/checkpoints"), exist_ok=True)

        self._build()

    def _build(self):
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        # Generators
        dataset_path = os.path.abspath(os.path.expanduser(os.path.expandvars(self.get("paths/rollouts"))))
        dataset = ImageFolder(dataset_path)

        split = dataset.data.shape[0] // 8
        self.train = torch.utils.data.DataLoader(dataset[split:], **self.get("dataloader"))
        self.valid = torch.utils.data.DataLoader(dataset[:split], **self.get("dataloader"))

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
                loss = self.model.loss_function(x_hat, input, mu, log_var, M_N=self.get("dataloader/batch_size"))
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
                true_sample_path = os.path.join(self.get("paths/samples"), f"true-epoch{epoch}.jpg")
                generated_sample_path = os.path.join(self.get("paths/samples"), f"generated-epoch{epoch}.jpg")
                plt.imsave(true_sample_path, true_image)
                plt.imsave(generated_sample_path, generated_image)

            # Checkpoint
            if epoch % self.get("checkpoint_every") == 0:
                torch.save(self.model.state_dict(), open(f"results/vae-checkpoints/vae-{epoch}.pt", "wb"))

            # Run Validation
            if epoch % self.get("valid_every") == 0:
                self.model.eval()
                for valbatch in tqdm(self.valid, "valid batches..."):
                    valbatch = valbatch.to(self.device)
                    x_hat, input, mu, log_var = self.model(valbatch)
                    loss = self.model.loss_function(x_hat, input, mu, log_var, M_N=self.get("dataloader/batch_size"))
                    if self.get("use_wandb"):
                        self.wandb_log(**{"valid_loss": loss})
                        self.wandb_log(**{"lr": self.scheduler.get_lr()})

                self.model.train()


if __name__ == "__main__":
    TrainVAE().run()
import os
import sys
import time
import torch
from tqdm import tqdm
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pathlib import Path

# to add jumping_quadrupeds to python path
sys.path.append(os.getcwd())

from jumping_quadrupeds.models.vae import ConvVAE
from jumping_quadrupeds.models.dataset import Box2dRollout, MySubset

# pip install -e speedrun from https://github.com/inferno-pytorch/speedrun
from speedrun import (
    BaseExperiment,
    WandBSweepMixin,
    IOMixin,
    SweepRunner,
    register_default_dispatch,
)

# FAIR's SLURM-based automated task launcher
import submitit


class TrainVAE(
    BaseExperiment, WandBSweepMixin, IOMixin, submitit.helpers.Checkpointable
):
    def __init__(self, args=None, kwargs=None):
        super(TrainVAE, self).__init__()
        WandBSweepMixin.WANDB_ENTITY = "jumping_quadrupeds"
        WandBSweepMixin.WANDB_PROJECT = "vae-tests"
        WandBSweepMixin.WANDB_GROUP = "vae-exploration"

        # this is necessary for running sweeps
        if kwargs:
            sys.argv = kwargs

        self.auto_setup()

        self.transform_dict = {
            "train": transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                ]
            ),
        }

        # Dataset
        dataset_path = os.path.abspath(
            os.path.expanduser(os.path.expandvars(self.get("paths/rollouts")))
        )
        dataset = Box2dRollout(dataset_path)

        # Split dataset into train and validation splits
        split_idx = len(dataset) // 8
        self.valid, self.train = torch.utils.data.random_split(
            dataset, [split_idx, len(dataset) - split_idx]
        )
        self.train = MySubset(self.train, self.transform_dict["train"])
        self.valid = MySubset(self.valid, self.transform_dict["valid"])

        self.train = torch.utils.data.DataLoader(self.train, **self.get("dataloader"))
        self.valid = torch.utils.data.DataLoader(self.valid, **self.get("dataloader"))

        self.model = ConvVAE(img_channels=3, latent_size=32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.get("lr"))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

    @register_default_dispatch
    def __call__(self):
        # if checkpoint_path and Path(checkpoint_path).exists():
        #     print("in checkpoint")
        #     self.model = self.model.load_state_dict(torch.load(checkpoint_path))

        # CUDA for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.get("use_wandb"):
            self.initialize_wandb()

        for epoch in tqdm(range(self.get("num_epochs")), desc="epochs..."):
            # Train the model
            for imgs, rollout_envs in tqdm(self.train, desc="batches..."):
                imgs, rollout_envs = imgs.to(self.device), rollout_envs.to(self.device)
                x_hat, mu, log_var = self.model(imgs)
                loss = self.model.loss_function(x_hat, imgs, mu, log_var)
                self.optimizer.zero_grad()
                loss["loss"].backward()
                self.scheduler.step(loss["loss"])
                self.next_step()
                if self.get("use_wandb"):
                    self.wandb_log(
                        **{
                            "train_loss": loss,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )
            self.next_epoch()

            # log gradients once per epoch
            if self.get("use_wandb"):
                self.wandb_watch(self.model, loss, log_freq=1)

            # Plot samples
            sampleid = np.random.choice(range(0, len(imgs)))
            true_image = imgs[sampleid].detach().cpu().moveaxis(0, 2).numpy()
            generated_image = x_hat[sampleid].detach().cpu().moveaxis(0, 2).numpy()
            if self.get("use_wandb"):
                img = np.zeros((3, 64, 64 * 2 + 2), dtype=np.float32)
                img[:, :, :64] = np.rollaxis(true_image, 2, 0)
                img[:, :, 66:] = np.rollaxis(generated_image, 2, 0)
                self.wandb_log_image("L: true, R: generated", img)
                self.wandb_log(
                    **{
                        "mu": mu[sampleid],
                        "var": torch.exp(0.5 * log_var)[sampleid],
                    }
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
                torch.save(
                    self.model.encoder.state_dict(),
                    open(f"{self.experiment_directory}/Weights/enc-{epoch}.pt", "wb"),
                )

            # Run Validation
            if epoch % self.get("valid_every") == 0:
                self.model.eval()

                for imgs, rollout_envs in tqdm(self.valid, "valid batches..."):
                    imgs, rollout_envs = imgs.to(self.device), rollout_envs.to(
                        self.device
                    )
                    x_hat, mu, log_var = self.model(imgs)
                    valid_loss = self.model.loss_function(x_hat, imgs, mu, log_var)
                    if self.get("use_wandb"):
                        self.wandb_log(**{"valid_loss": valid_loss})
                        self.wandb_log(**{"lr": self.optimizer.param_groups[0]["lr"]})

                self.model.train()
        return valid_loss


class SweepVAE(SweepRunner, WandBSweepMixin, IOMixin):
    def __init__(self):
        self.auto_setup()
        super(SweepVAE, self).__init__(TrainVAE, None, sys.argv)

    @register_default_dispatch
    def __call__(self):
        WandBSweepMixin.WANDB_ENTITY = "jumping_quadrupeds"
        WandBSweepMixin.WANDB_PROJECT = "vae-tests"
        WandBSweepMixin.WANDB_GROUP = "vae-exploration"
        self.parse_experiment_directory()
        self.read_config_file()
        if self.get_arg("wandb.sweep", False):
            self.update_configuration_from_wandb(dump_configuration=True)

        self.run()


if __name__ == "__main__":
    folder = os.path.join(sys.argv[1], "Configurations", "Logs")
    ex = submitit.AutoExecutor(folder=folder)

    # setup the executor parameters based on the cluster location
    if ex.cluster == "slurm":
        ex.update_parameters(
            mem_gb=16,
            cpus_per_task=12,
            timeout_min=1000,
            tasks_per_node=1,
            nodes=1,
            slurm_partition="main",
            gres="gpu:rtx8000:1",
        )
    elif ex.cluster == "local":
        ex.update_parameters(timeout_min=1000)
    else:
        raise ("Where the hell am I?")

    if "--wandb.sweep" in sys.argv:
        assert "--njobs" in sys.argv
        njobs = int(sys.argv[sys.argv.index("--njobs") + 1])
        exp_path_root = sys.argv[1]
        job_array = []
        for i in range(njobs):
            sys.argv[1] = os.path.join(exp_path_root, f"{i}")
            func = SweepVAE()
            job_array.append(func)
        executor.update_parameters(slurm_array_parallelism=2)
        jobs = executor.map_array(job_array)  # just a list of jobs

    else:
        func = TrainVAE()
        jobs = [ex.submit(func)]

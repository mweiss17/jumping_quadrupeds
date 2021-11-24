import gym
import torch
import sys
import numpy as np
import submitit
from tqdm import tqdm, trange
from pathlib import Path
from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
from jumping_quadrupeds.rl.buffer import make_replay_loader, ReplayBufferStorage
from jumping_quadrupeds.rl.networks import ConvActorCritic
from jumping_quadrupeds.env import make_env
from jumping_quadrupeds.utils import DataSpec


class TrainPPOConv(
    BaseExperiment, WandBMixin, IOMixin, submitit.helpers.Checkpointable
):
    WANDB_ENTITY = "jumping_quadrupeds"
    WANDB_PROJECT = "rl-encoder-test"

    def __init__(self):
        super(TrainPPOConv, self).__init__()
        self.auto_setup()

        if self.get("use_wandb"):
            self.initialize_wandb()

        SEED = self.get("SEED", 58235)
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)

        # env setup
        env = make_env(
            env_name=self.get("env_name"),
            seed=SEED,
        )
        device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy and value networks
        ac = ConvActorCritic(
            env.observation_space,
            env.action_space,
            shared_encoder=self.get("shared_encoder", False),
            hidden_sizes=self.get("conv_ac_hidden_scaling", 16),
            log_std=self.get("log_std", 0.5),
        )

        if self.get("vae_enc_checkpoint"):
            print(f"Loading saved encoder checkpoint to the state encoder")
            ac.load_encoder(self.get("vae_enc_checkpoint"))

        # do we want to freeze the encoder?
        if self.get("freeze_encoder"):
            ac.freeze_encoder()

        # Put on device
        ac = ac.to(device)

        if self.get("use_wandb"):
            self.wandb_watch(ac.pi, log_freq=1)
            self.wandb_watch(ac.v, log_freq=1)

        data_specs = [
            DataSpec("obs", env.observation_space.shape, np.uint8),
            DataSpec("act", env.action_space.shape, np.float32),
            DataSpec("rew", (1,), np.float32),
            DataSpec("val", (1,), np.float32),
            DataSpec("discount", (1,), np.float32),
            DataSpec("logp", env.action_space.shape, np.float32),
        ]

        replay_dir = Path(self.experiment_directory + "/Logs/buffer")
        buf = ReplayBufferStorage(data_specs, replay_dir)

        self.replay_loader = make_replay_loader(replay_dir, **self.get("buffer/kwargs"))
        self._replay_iter = None
        if self.get("agent/name") == "ppo":
            from jumping_quadrupeds.rl.ppo import PPO

            self.agent = PPO(self, env, ac, buf, device=device)
        elif self.get("agent/name") == "drqv2":
            from jumping_quadrupeds.rl.drqv2 import DRQV2Agent

            self.agent == DRQV2Agent(self, env, ac, buf, device=device)

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def checkpoint_now(self):
        if (self.epoch % self.get("save_freq") == 0) or (
            epoch == self.get("epochs") - 1
        ):
            return True
        return False

    @register_default_dispatch
    def __call__(self):

        # Main loop: collect experience in env and update/log each epoch
        for epoch in trange(self.get("epochs")):
            self.agent.act()

            if self.checkpoint_now:
                self.agent.save_checkpoint(epoch)

            self.agent.update(self.replay_iter)

            # Record video of the updated policy
            self.send_wandb_video()
            self.next_epoch()


if __name__ == "__main__":
    # Default cmdline args Flo
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "experiments/ppo-car",
            "--inherit",
            "templates/ppo-car",
        ]

    TrainPPOConv().run()

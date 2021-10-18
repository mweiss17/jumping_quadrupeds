import gym
import torch
import sys
import numpy as np
import submitit
from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
from jumping_quadrupeds.rl.buffer import PpoBuffer
from jumping_quadrupeds.rl.networks import ConvActorCritic, ConvSharedActorCritic
from jumping_quadrupeds.rl.ppo import PPO
from jumping_quadrupeds.env import make_env


class TrainPPOConv(BaseExperiment, WandBMixin, IOMixin, submitit.helpers.Checkpointable):
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
            env_name=self.get("ENV", "CarRacing-v0"),
            seed=SEED,
            render_mode=False,
            full_ep=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy and value networks
        ac = ConvActorCritic(env.observation_space, env.action_space, shared_encoder=self.get("shared_encoder", False))

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

        # buffer
        buf = PpoBuffer(
            (
                env.observation_space.shape[2],
                env.observation_space.shape[1],
                env.observation_space.shape[0],
            ),
            env.action_space.shape,
            self.get("steps_per_epoch"),
            self.get("gamma"),
            self.get("lam"),
            device,
            self.get("save_transitions", 0),
        )
        self.ppo = PPO(self, env, ac, buf, device=device)
    
    @register_default_dispatch
    def __call__(self):
        self.ppo.train_loop()


if __name__ == "__main__":
    # Default cmdline args Flo
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "experiments/ppo-car", "--inherit", "templates/ppo-car"]

    TrainPPOConv().run()



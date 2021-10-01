import gym
import torch
import numpy as np
from speedrun import BaseExperiment, WandBMixin, IOMixin
from jumping_quadrupeds.rl.buffer import PpoBuffer
from jumping_quadrupeds.rl.networks import ConvActorCritic
from jumping_quadrupeds.rl.ppo import PPO
from jumping_quadrupeds.env import make_env


class TrainPPOConv(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(TrainPPOConv, self).__init__()
        self.auto_setup()
        WandBMixin.WANDB_ENTITY = "jumping_quadrupeds"
        WandBMixin.WANDB_PROJECT = "rl-tests"
        WandBMixin.WANDB_GROUP = "vnav-car-racing-ppo-conv"

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
            full_episode=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy and value networks
        ac = ConvActorCritic(env.observation_space, env.action_space)
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
        )
        self.ppo = PPO(self, env, ac, buf)

    def run(self):
        self.ppo.train_loop()


TrainPPOConv().run()
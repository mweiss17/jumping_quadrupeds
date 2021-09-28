import gym
import torch
import numpy as np
import wandb
from jumping_quadrupeds.rl.buffer import PpoBuffer
from jumping_quadrupeds.rl.networks import MLPActorCritic
from jumping_quadrupeds.rl.params import PpoParams
from jumping_quadrupeds.rl.ppo import PPO


class TrainPPO(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(TrainPPO, self).__init__()
        self.auto_setup()
        WandBMixin.WANDB_ENTITY = "jumping_quadrupeds"
        WandBMixin.WANDB_PROJECT = "rl-tests"
        WandBMixin.WANDB_GROUP = "vnav-cartpole"

        if self.get("use_wandb"):
            self.initialize_wandb()

        SEED = self.get("SEED", 58235)
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)

        # env setup
        self.env = gym.make(self.get("ENV", "CartPole-v0"))
        self.env.seed(SEED)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # policy and value networks
        self.ac = MLPActorCritic(self.env.observation_space, self.env.action_space)
        wandb.watch(self.ac.pi)
        wandb.watch(self.ac.v)

        # buffer
        buf = PpoBuffer(
            self.env.observation_space.shape, self.env.action_space.shape, self
        )

        self.ppo = PPO(self, self.env, self.ac, buf)

    def run(self):
        self.ppo.train_loop()

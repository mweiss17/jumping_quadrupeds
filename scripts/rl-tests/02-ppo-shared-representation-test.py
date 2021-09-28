import gym
import torch
import numpy as np
from speedrun import BaseExperiment, WandBMixin, IOMixin
from jumping_quadrupeds.rl.buffer import PpoBuffer
from jumping_quadrupeds.rl.networks import MLPSharedActorCritic
from jumping_quadrupeds.rl.ppo import PPO


class TrainPPO(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(TrainPPO, self).__init__()
        self.auto_setup()
        WandBMixin.WANDB_ENTITY = "jumping_quadrupeds"
        WandBMixin.WANDB_PROJECT = "rl-tests"
        WandBMixin.WANDB_GROUP = "vnav-cartpole-shared-v2"

        if self.get("use_wandb"):
            self.initialize_wandb()

        SEED = self.get("SEED", 58235)
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)

        # env setup
        env = gym.make(self.get("ENV", "CartPole-v0"))
        env.seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy and value networks
        ac = MLPSharedActorCritic(env.observation_space, env.action_space)
        ac = ac.to(device)

        if self.get("use_wandb"):
            self.wandb_watch(ac._pi, log_freq=1)
            self.wandb_watch(ac._v, log_freq=1)
            self.wandb_watch(ac.shared_encoder, log_freq=1)

        # buffer
        buf = PpoBuffer(
            env.observation_space.shape,
            env.action_space.shape,
            self.get("steps_per_epoch"),
            self.get("gamma"),
            self.get("lam"),
            device,
        )
        self.ppo = PPO(self, env, ac, buf)

    def run(self):
        self.ppo.train_loop()


TrainPPO().run()

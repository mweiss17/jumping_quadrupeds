import gym
import torch
import numpy as np
import wandb
from jumping_quadrupeds.rl.buffer import PpoBuffer
from jumping_quadrupeds.rl.networks import MLPActorCritic
from jumping_quadrupeds.rl.params import PpoParams
from jumping_quadrupeds.rl.ppo import PPO

SEED = 123
ENV = "CartPole-v0"
WANDB_PROJ = "rl-tests"
WANDB_GROUP = "vnav-cartpole"

params = PpoParams(env_name=ENV, seed=SEED, verbose=True)
wandb.init(project=WANDB_PROJ, group=WANDB_GROUP, config=params)

# seed stuff
np.random.seed(SEED)
torch.random.manual_seed(SEED)

# env setup
env = gym.make(ENV)

# policy and value networks
ac = MLPActorCritic(env.observation_space, env.action_space)
wandb.watch(ac.pi)
wandb.watch(ac.v)

# buffer
buf = PpoBuffer(env.observation_space.shape, env.action_space.shape, params)

ppo = PPO(env, ac, params, buf, wandb)

ppo.train_loop()

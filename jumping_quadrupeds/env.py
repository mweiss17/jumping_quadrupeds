import numpy as np
import gym

from PIL import Image
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

SCREEN_X = 64
SCREEN_Y = 64


def _process_frame(frame):
    obs = np.array(Image.fromarray(np.rollaxis(frame, 0, 2)).resize((64, 64)))
    obs = obs.astype(np.float32) / 255.0
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    obs = np.expand_dims(obs, 0)
    obs = np.transpose(obs, (0, 3, 2, 1))
    return obs


class CarRacingWrapper(CarRacing):
    def __init__(self, full_episode=False):
        super(CarRacingWrapper, self).__init__()
        self.full_episode = full_episode
        self.observation_space = Box(
            low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3)
        )  # , dtype=np.uint8

    def step(self, action):
        obs, reward, done, _ = super(CarRacingWrapper, self).step(action)
        if self.full_episode:
            return _process_frame(obs), reward, False, {}
        return _process_frame(obs), reward, done, {}


def make_env(env_name, seed=-1, render_mode=False, full_episode=False):
    env = CarRacingWrapper(full_episode=full_episode)
    if seed >= 0:
        env.seed(seed)
    """
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  """
    return env

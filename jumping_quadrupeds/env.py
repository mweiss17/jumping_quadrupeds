import numpy as np
import gym
import wandb

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
    def __init__(self, full_ep=False):
        super(CarRacingWrapper, self).__init__()
        self.full_episode = full_ep
        self.observation_space = Box(
            low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3)
        )  # , dtype=np.uint8

    def step(self, action):
        obs, reward, done, _ = super(CarRacingWrapper, self).step(action)
        if self.full_episode:
            return _process_frame(obs), reward, False, {}
        return _process_frame(obs), reward, done, {}


class VideoWrapper(gym.Wrapper):
    """Gathers up the frames from an episode and allows to upload them to Weights & Biases
    Thanks to @cyrilibrahim for this snippet
    """

    def __init__(self, env, update_freq=25):
        super(VideoWrapper, self).__init__(env)
        self.episode_images = []
        # we need to store the last episode's frames because by the time we
        # wanna upload them, reset() has juuust been called, so the self.episode_rewards buffer would be empty
        self.last_frames = None

        # we also only render every 20th episode to save framerate
        self.episode_no = 0
        self.render_every_n_episodes = update_freq  # can be modified

    def reset(self, **kwargs):
        self.episode_no += 1
        if self.episode_no == self.render_every_n_episodes:
            self.episode_no = 0
            self.last_frames = self.episode_images[:]
            self.episode_images.clear()

        state = self.env.reset()

        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.episode_no + 1 == self.render_every_n_episodes:
            frame = np.copy(self.env.render("rgb_array"))
            self.episode_images.append(frame)

        return state, reward, done, info

    def send_wandb_video(self):
        if self.last_frames is None or len(self.last_frames) == 0:
            print("Not enough images for GIF. continuing...")
            return

        lf = np.array(self.last_frames)
        print(lf.shape)
        frames = np.swapaxes(lf, 1, 3)
        frames = np.swapaxes(frames, 2, 3)
        wandb.log({"video": wandb.Video(frames, fps=10, format="gif")})
        print("=== Logged GIF")


def make_env(env_name, seed=-1, render_mode=False, full_ep=False, render_every=1):
    env = VideoWrapper(CarRacingWrapper(full_ep=full_ep), update_freq=render_every)
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

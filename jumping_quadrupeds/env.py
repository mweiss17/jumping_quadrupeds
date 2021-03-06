import torch
import numpy as np
import wandb
import gym
import importlib
import cv2
import dm_env
import enum

from collections import deque
from einops import rearrange
from typing import Any, NamedTuple

from jumping_quadrupeds.frame_stack import FrameStack

try:
    import dm_control
    import dmc2gym
    from dm_control.suite.wrappers import action_scale, pixels
except ImportError:
    dm_control = None

try:
    import SEVN_gym
except ImportError:
    SEVN_gym = None


class StepType(enum.IntEnum):
    """Defines the status of a `TimeStep` within a sequence."""

    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST


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

    def step(self, action, render_type="state_pixels"):  # rgb_array
        state, reward, done, info = self.env.step(action)

        if self.episode_no + 1 == self.render_every_n_episodes:

            if len(state.shape) == 4:
                frame = state[0]
            else:
                frame = state
            self.episode_images.append(np.copy(frame))

        return state, reward, done, info

    def send_wandb_video(self):
        if self.last_frames is None or len(self.last_frames) == 0:
            print("Not enough images for GIF. continuing...")
            return

        frames = np.array(self.last_frames)
        print(frames.shape)
        # frames = np.swapaxes(lf, 1, 3)
        # frames = np.swapaxes(frames, 2, 3)

        wandb.log({"video": wandb.Video(frames, fps=10, format="gif")})
        print("=== Logged GIF")
        self.last_frames = None


class ActionScale(gym.core.Wrapper):
    def __init__(self, env, new_min, new_max):
        super().__init__(env)
        orig_min = self.env.action_space.low
        orig_max = self.env.action_space.high
        new_min = np.array(new_min)
        new_max = np.array(new_max)
        self._transform = lambda a: orig_min + (orig_max - orig_min) / (new_max - new_min) * (a - new_min)
        self.env.action_space.low = np.repeat([new_min], env.action_space.shape)
        self.env.action_space.high = np.repeat([new_max], env.action_space.shape)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(self._transform(action))


class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        observation = observation.transpose(2, 0, 1)
        return observation


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, resize_w=80, resize_h=80):
        gym.ObservationWrapper.__init__(self, env)
        self.resize_h = resize_h
        self.resize_w = resize_w
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[1, 1, 1],
            [obs_shape[0], resize_h, resize_w],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation

    def reset(self):
        obs = gym.ObservationWrapper.reset(self)
        obs = cv2.resize(
            obs.swapaxes(0, 2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC
        ).swapaxes(0, 2)
        return obs

    def step(self, actions):
        obs, reward, done, info = gym.ObservationWrapper.step(self, actions)
        return (
            cv2.resize(
                obs.swapaxes(0, 2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC
            ).swapaxes(0, 2),
            reward,
            done,
            info,
        )


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        observation = time_step[0]
        discount = np.array([1.0], dtype=np.float32)
        if len(time_step) == 1:
            step_type = StepType.FIRST
            reward = np.array([0.0], dtype=np.float32)
        if len(time_step) > 1:
            reward = np.array([time_step[1]], dtype=np.float32)
            done = time_step[2]
            if done:
                step_type = StepType.LAST
            else:
                step_type = StepType.MID
        return ExtendedTimeStep(
            observation=observation,
            step_type=step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_space

    def action_spec(self):
        return self._env.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(FrameSkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            # Take a step
            obs, reward, done, info = self.env.step(action)
            # Update the total reward by summing the (reward obtained from the step taken) + (the current
            # total reward)
            total_reward += reward
            # If the game ends, break the for loop
            if done:
                break

        return obs, total_reward, done, info


def make_env(seed=-1, name=None, action_repeat=1, frame_stack=1, w=84, h=84, render_every=None, discrete_action=False, **kwargs):
    if "Duckietown" in name:
        import gym_duckietown

    if name.startswith("dm-"):
        domain, task = name[3:].split("_")
        camera_id = dict(quadruped=2).get(domain, 0)
        env = dmc2gym.make(
            domain,
            task,
            from_pixels=True,
            visualize_reward=False,
            width=w,
            height=h,
            frame_skip=action_repeat,
            camera_id=camera_id,
        )
    elif name.startswith("gym-"):
        env_name = name[4:]
        env = gym.make(env_name)
        env = ResizeWrapper(PyTorchObsWrapper(env), resize_w=w, resize_h=h)
        env = FrameSkipEnv(env, skip=action_repeat)
    else:
        raise ValueError(f"Unknown environment name: {name}.")

    if discrete_action and "Duckietown" in name:
        env = gym_duckietown.wrappers.DiscreteWrapper(env)
    else:
        env = ActionScale(env, new_min=-1.0, new_max=1.0)

    env = FrameStack(env, frame_stack)
    env = VideoWrapper(env, update_freq=render_every)
    env = ExtendedTimeStepWrapper(env)
    env.seed(seed)
    return env

from collections import deque
from typing import Optional

import numpy as np
from gym.spaces import Box
from gym import ObservationWrapper


class LazyFrames:
    r"""Ensures common frames are only stored once to optimize memory use.
    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.
    .. note::
        This object should only be converted to numpy array just before forward pass.
    Args:
        lz4_compress (bool): use lz4 to compress the frames internally
    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames) * self.frame_shape[0],) + self.frame_shape[1:]
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.concatenate([self._check_decompress(f) for f in self._frames[int_or_slice]])

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(self.frame_shape)
        return frame


class FrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> envs = gym.make('PongNoFrameskip-v0')
        >>> envs = FrameStack(envs, 4)
        >>> envs.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)
        low = np.repeat(self.observation_space.low, num_stack, axis=0)
        high = np.repeat(self.observation_space.high, num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    @property
    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.concatenate(self.frames)  # LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return (self.observation, reward, done, info)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return (self.observation,)
import numpy as np
import wandb
import gym
import cv2
try:
    import dm_control
    import dmc2gym
except ImportError:
    dm_control = None


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
        self.last_frames = None


class ActionScale(gym.core.Wrapper):
    def __init__(self, env, new_min, new_max):
        super().__init__(env)
        orig_min = self.env.action_space.low
        orig_max = self.env.action_space.high
        new_min = np.array(new_min)
        new_max = np.array(new_max)
        self._transform = lambda a: orig_min + (orig_max - orig_min) / (
            new_max - new_min
        ) * (a - new_min)
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


def make_env(env_name, action_repeat=1, w=84, h=84, seed=-1, render_every=2):
    if "Duckietown" in env_name:
        import gym_duckietown

    # TODO add framestacking https://github.com/facebookresearch/drqv2/blob/7ad7e05fa44378c64998dc89586a9703b74531ab/dmc.py
    if env_name.startswith("dm-"):
        domain, task = env_name[3:].split("_")
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
    elif env_name.startswith("gym-"):
        env_name = env_name[4:]
        env = gym.make(env_name)
        env = ResizeWrapper(PyTorchObsWrapper(env), resize_w=w, resize_h=h)
    else:
        raise ValueError("Unknown environment name: {}".format(env_name))
    env = ActionScale(env, new_min=-1.0, new_max=1.0)
    env = VideoWrapper(env, update_freq=render_every)
    if seed >= 0:
        env.seed(seed)

    return env

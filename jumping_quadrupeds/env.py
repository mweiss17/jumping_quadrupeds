import numpy as np
import gym
import wandb
import gym_duckietown
from gym_duckietown.wrappers import PyTorchObsWrapper, ResizeWrapper
from PIL import Image
from gym.spaces.box import Box

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


def make_env(env_name, action_repeat=1, w=64, h=64, seed=-1, render_every=25):
    # TODO add framestacking https://github.com/facebookresearch/drqv2/blob/7ad7e05fa44378c64998dc89586a9703b74531ab/dmc.py
    if env_name.startswith("dm-"):
        domain, task = env_name[3:].split("_")
        camera_id = dict(quadruped=2).get(domain, 0)

        env = dmc2gym.make(
            domain_name,
            task_name,
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

    env = VideoWrapper(env, update_freq=render_every)
    if seed >= 0:
        env.seed(seed)

    return env

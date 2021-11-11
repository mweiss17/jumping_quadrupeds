import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import multiprocessing as mp
from PIL import Image
from io import BytesIO
import h5py
from tqdm import trange
import shutil
import click
from typing import Callable


def data_generator(
    env_name: str,
    num_episodes: int,
    num_processes: int,
    seed: int,
    output_path: str,
    initial_cutoff: int,
    max_timesteps_per_episode: int,
    get_action: Callable,
):
    np.random.seed(seed)
    env = gym.make(env_name)

    start = seed * num_episodes // num_processes  # shouldn't this be
    end = num_episodes // num_processes + seed * num_episodes // num_processes

    for i in range(start, end + 1):
        obs = env.reset()
        rollout_path = os.path.join(output_path, "raw", str(i))
        os.makedirs(rollout_path, exist_ok=True)

        for t in range(max_timesteps_per_episode + initial_cutoff):
            action = get_action(obs)
            obs, rew, done, _ = env.step(action)
            if t < initial_cutoff:
                continue
            im = Image.fromarray(obs)
            im.save(f"{rollout_path}/{t - initial_cutoff}.png", "PNG")
    print(f"gen_data {seed} finished")
    return


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument("max_timesteps_per_episode", type=int, default=200)
@click.argument("output_path", type=str, default="random-rollouts-50k/")
@click.argument("initial_cutoff", type=int, default=30)
@click.option("--env_name", type=str, default="CarRacing-v0")
@click.option("--num_episodes", type=int, default=256)
@click.option("--num_processes", type=int, default=8)
def main(
    env_name,
    max_timesteps_per_episode,
    output_path,
    initial_cutoff,
    num_episodes,
    num_processes,
):
    if "CarRacing" in env_name:

        def get_action(obs):
            return np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), 0])

    simulators = [
        mp.Process(
            target=data_generator,
            args=(
                env_name,
                seed,
                num_episodes,
                num_processes,
                output_path,
                initial_cutoff,
                max_timesteps_per_episode,
                get_action,
            ),
        )
        for seed in range(num_processes)
    ]
    for s in simulators:
        s.start()

    for s in simulators:
        s.join()
    print("done! Now writing to hdf5")
    env = gym.make(env_name)

    filename = f"{env_name}_rollouts_e{num_episodes}_t{max_timesteps_per_episode}.hdf5"
    filepath = os.path.join(output_path, filename)
    f = h5py.File(filepath, "w")
    states = f.create_dataset(
        "states",
        (num_episodes, max_timesteps_per_episode, *env.observation_space.shape),
        dtype=np.uint8,
        compression="lzf",
        chunks=(1, 1, *env.observation_space.shape),
        shuffle=True,
    )
    for episode in trange(num_episodes):
        episode_buffer = np.zeros(
            (max_timesteps_per_episode, *env.observation_space.shape), np.uint8
        )  # to reduce i/o
        for step in range(max_timesteps_per_episode):
            im = plt.imread(
                os.path.join(output_path, "raw", f"{episode}", f"{step}.png")
            )
            episode_buffer[step] = np.copy(im) * 255
        states[episode] = episode_buffer

    f.flush()
    f.close()
    print(f"done, wrote file {filepath}")
    shutil.rmtree(os.path.join(output_path, "raw"))


if __name__ == "__main__":
    main()

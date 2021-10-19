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

num_episodes = 256  # 1024 # 2^10
max_timesteps_per_episode = 200
output_path = "random-rollouts-50k/"
num_processes = os.cpu_count()  # multiple of 2, ideally
# num_processes = 4  # multiple of 2, ideally
initial_cutoff = 30


def data_generator(seed):
    np.random.seed(seed)
    env = gym.make("CarRacing-v0")

    def get_action(obs):
        return np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), 0])

    start = seed * num_episodes // num_processes  # shouldn't this be
    end = num_episodes // num_processes + seed * num_episodes // num_processes
    # print(f"seed {seed}, start {start}, end {end}")
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
            im.save(f"{rollout_path}/{t-initial_cutoff}.png", "PNG")
    print(f"gen_data {seed} finished")
    return


#
num_todo = num_episodes * max_timesteps_per_episode
simulators = [mp.Process(target=data_generator, args=(seed,)) for seed in range(num_processes)]
for s in simulators:
    s.start()

for s in simulators:
    s.join()
print("done! Now writing to hdf5")

filename = f"rollouts_e{num_episodes}_t{max_timesteps_per_episode}.hdf5"
filepath = os.path.join(output_path, filename)
f = h5py.File(filepath, "w")
states = f.create_dataset(
    "states",
    (num_episodes, max_timesteps_per_episode, 96, 96, 3),
    dtype=np.uint8,
    compression="lzf",
    chunks=(1, 1, 96, 96, 3),
    shuffle=True,
)
for episode in trange(num_episodes):
    episode_buffer = np.zeros((max_timesteps_per_episode, 96, 96, 3), np.uint8)  # to reduce i/o
    for step in range(max_timesteps_per_episode):
        im = plt.imread(os.path.join(output_path, "raw", f"{episode}", f"{step}.png"))
        episode_buffer[step] = np.copy(im)
    states[episode] = episode_buffer

f.flush()
f.close()
print(f"done, wrote file {filepath}")
shutil.rmtree(os.path.join(output_path, "raw"))

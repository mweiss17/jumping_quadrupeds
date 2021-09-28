import gym
import numpy as np
import os
import zipfile
import multiprocessing as mp
from PIL import Image
from io import BytesIO

num_episodes = 256  # 1024 # 2^10
max_timesteps_per_episode = 200
output_path = "random-rollouts-50k/"
num_processes = 32  # multiple of 2, ideally


def data_generator(seed):
    np.random.seed(seed)
    env = gym.make("CarRacing-v0")

    def get_action(obs):
        return np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), 0])

    start = seed * num_episodes // os.cpu_count()
    end = num_episodes // os.cpu_count() + seed * num_episodes // os.cpu_count()
    for i in range(start, end):
        obs = env.reset()
        rollout_path = os.path.join(output_path, str(i))
        os.makedirs(rollout_path, exist_ok=True)

        for t in range(max_timesteps_per_episode + 10):
            action = get_action(obs)
            obs, rew, done, _ = env.step(action)
            if t < 30:
                continue
            im = Image.fromarray(obs)
            im.save(f"{rollout_path}/{t}.png", "PNG")
    print(f"gen_data {seed} finished")
    return


num_todo = num_episodes * max_timesteps_per_episode
simulators = [
    mp.Process(target=data_generator, args=(seed,)) for seed in range(num_processes)
]
for s in simulators:
    s.start()

for s in simulators:
    s.join()
print("done!")

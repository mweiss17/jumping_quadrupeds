import gym
import numpy as np
import torch
import os
import zipfile
from tqdm import tqdm
import pickle
import multiprocessing as mp
from multiprocessing import Pool
from PIL import Image, ImageFile
from io import BytesIO

num_episodes = 256 # 1024 # 2^10
max_timesteps_per_episode = 200
output_path = "data/random-rollouts-50k.zip"
num_processes = 32  # multiple of 2, ideally


def writer(outbox, output_path, num_todo):
    "single process that writes compressed files to disk"
    num_finished = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        while num_finished < num_todo:
            # all compression processes have finished
            if num_finished >= max_timesteps_per_episode * num_episodes: break

            tardata = outbox.get()
            print(f"{num_finished} / {num_todo}")
            if any(tardata):
                num_finished += 1
            else:
                continue

            name, data = tardata
            im = Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(96, 96, 3))
            image_file = BytesIO()
            im.save(image_file, 'PNG')

            zipf.writestr(name, image_file.getvalue())


def gen_data(seed, outbox):
    np.random.seed(seed)
    env = gym.make("CarRacing-v0")

    def get_action(obs):
        return np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), 0])

    start = seed * num_episodes // os.cpu_count()
    end = num_episodes // os.cpu_count() + seed * num_episodes // os.cpu_count()
    for i in range(start, end):
        obs = env.reset()

        for t in range(max_timesteps_per_episode+10):
            action = get_action(obs)
            obs, rew, done, _ = env.step(action)
            if t < 30:
                continue
            outbox.put([f"{i}-{t}.png", obs.tobytes()])
    print(f"gen_data {seed} finished")
    return

if __name__ == '__main__':
    num_todo = num_episodes * max_timesteps_per_episode

    outbox = mp.Queue(4*num_processes)  # limit size
    simulators = [mp.Process(target=gen_data, args=(seed, outbox)) for seed in range(num_processes)]
    for s in simulators: s.start()

    # one process to write
    writer = mp.Process(target=writer, args=(outbox, output_path, num_todo))
    writer.start()
    writer.join()  # wait for it to finish
    for s in simulators: s.join()
    print('done!')


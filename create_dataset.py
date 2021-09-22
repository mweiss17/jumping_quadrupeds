import gym
import numpy as np
import torch
import os
import zipfile
from tqdm import tqdm
from PIL import Image
import cv2
import pickle
import multiprocessing as mp

num_episodes = 64 # 1024 # 2^10
max_timesteps_per_episode = 200

def writer(outbox, output_path, num_processes):
    "single process that writes compressed files to disk"
    num_finished = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

        while True:
            # all compression processes have finished
            if num_finished >= max_timesteps_per_episode * num_episodes: break

            tardata = outbox.get()

            # a compression process has finished
            if tardata == None:
                num_finished += 1
                continue

            print(num_finished)
            fn, data = tardata
            retval, buf = cv2.imencode('.png',  data)

            name = os.path.join(output_path, fn) + '.zip'
            zipf.writestr(name, buf)

    return

def gen_data(seed, outbox):
    np.random.seed(seed)
    env = gym.make("CarRacing-v0")

    def get_action(obs):
        return np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), 0])

    start = seed * num_episodes // os.cpu_count()
    end = num_episodes // os.cpu_count() + seed * num_episodes // os.cpu_count()
    for i in tqdm(range(start, end), desc="Episode number..."):
        obs = env.reset()

        for t in range(max_timesteps_per_episode+10):
            if t < 10:
                continue
            action = get_action(obs)
            obs, rew, done, _ = env.step(action)
            outbox.put(f"{i}-{t}.png", pickle.dump(obs))

if __name__ == '__main__':
    # num_processes = os.cpu_count()
    num_processes = 1

    outbox = mp.Queue(4*num_processes)  # limit size

    # n processes to compress
    simulators = [mp.Process(target=gen_data, args=(seed, outbox)) for seed in range(num_processes)]
    for s in simulators: s.start()

    # one process to write
    writer = mp.Process(target=writer, args=(outbox, fld, num_processes))
    writer.start()
    writer.join()  # wait for it to finish
    print('done!')

    with Pool(num_processes) as p:
        p.map(gen_data, range(num_processes))

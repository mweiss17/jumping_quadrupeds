import torch
import sys
import numpy as np
import submitit
from tqdm import trange
from collections import defaultdict
from pathlib import Path
from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
from jumping_quadrupeds.buffer import ReplayBufferStorage
from jumping_quadrupeds.env import make_env
from jumping_quadrupeds.utils import DataSpec, preprocess_obs, set_seed, build_loader
from jumping_quadrupeds.ppo.agent import PPOAgent
from jumping_quadrupeds.drqv2.agent import DrQV2Agent
from jumping_quadrupeds.spr.agent import SPRAgent
from jumping_quadrupeds.mae.agent import MAEAgent


class Trainer(BaseExperiment, WandBMixin, IOMixin, submitit.helpers.Checkpointable):
    WANDB_ENTITY = "jumping_quadrupeds"
    WANDB_PROJECT = "rl-encoder-test"

    def __init__(self):
        super(Trainer, self).__init__()
        self.auto_setup()

        if self.get("use_wandb"):
            self.initialize_wandb()

        # env setup
        seed = set_seed(seed=self.get("seed"))
        self.env = make_env(seed=seed, **self.get("env/kwargs"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_returns = defaultdict(list)
        self.ep_idx = 0

        self._build_buffer()
        self._build_agent()

    def _build_buffer(self):
        data_specs = [
            DataSpec("obs", self.env.observation_space.shape, np.uint8),
            DataSpec("act", self.env.action_space.shape, np.float32),
            DataSpec("rew", (1,), np.float32),
            DataSpec("val", (1,), np.float32),
            DataSpec("discount", (1,), np.float32),
            DataSpec("logp", self.env.action_space.shape, np.float32),
        ]

        replay_dir = Path(self.experiment_directory + "/Logs/buffer")
        self.replay_storage = ReplayBufferStorage(data_specs, replay_dir)
        self.replay_loader = build_loader(replay_dir, **self.get("buffer/kwargs"))
        self._replay_iter = None

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _build_agent(self):
        if self.get("agent/name") == "ppo":
            self.agent = PPOAgent(self.env.observation_space,
                                  self.env.action_space,
                                  self.get("use_wandb"),
                                  device=self.device,
                                  **self.get("agent/kwargs"))
            if self.get("use_wandb"):
                self.wandb_watch(self.agent.ac.pi, log_freq=1)
                self.wandb_watch(self.agent.ac.v, log_freq=1)

        elif self.get("agent/name") == "drqv2":
            self.agent = DrQV2Agent(
                self.env.observation_space,
                self.env.action_space,
                self.device,
                **self.get("agent/kwargs"),
            )
        elif self.get("agent/name") == "spr":
            self.agent = SPRAgent(
                self.env.observation_space,
                self.env.action_space,
                self.device,
                self.get("buffer/kwargs/jumps"),
                **self.get("agent/kwargs"),
            )
        elif self.get("agent/name") == "mae":
            self.agent = MAEAgent(
                self.env.observation_space,
                self.env.action_space,
                self.device,
                **self.get("agent/kwargs"),
            )
        else:
            raise ValueError(f"Unknown agent {self.get('agent/name')}. Have you specified an agent to use a la ` --macro templates/agents/ppo.yml' ? ")
        if False:
            self.agent.load_checkpoint(self.experiment_directory)

    @property
    def checkpoint_now(self):
        if self.step % self.get("save_freq") == 0:
            return True
        return False

    @property
    def episode_timeout(self):
        return len(self.episode_returns[self.ep_idx]) == self.get("env/kwargs/max_ep_len")

    @property
    def update_now(self):
        update_every = self.step % self.get("agent/kwargs/update_every_steps") == 0 and self.step > 0
        gt_seed_frames = self.step > self.get("agent/kwargs/num_seed_frames")
        ep_len_gt_nstep = len(self.episode_returns[self.ep_idx]) > self.get("buffer/kwargs/nstep", 1)
        return update_every and gt_seed_frames and ep_len_gt_nstep

    def write_logs(self):
        ep_rets = list(dict(self.episode_returns).values())
        ep_rets = np.array([xi + [0] * (self.get("env/kwargs/max_ep_len") - len(xi)) for xi in ep_rets])
        full_episodic_return = ep_rets.sum(axis=1)
        if self.get("use_wandb"):
            self.wandb_log(
                **{
                    "Episode mean reward": np.mean(full_episodic_return),
                    "Episode return": full_episodic_return[-1],
                    "Episode mean length": np.mean([len(ep) for ep in self.episode_returns.values()]),
                    "Number of Episodes": len(self.episode_returns),
                }
            )
            self.env.send_wandb_video()


    @register_default_dispatch
    def __call__(self):
        obs = self.env.reset()
        for _ in trange(self.get("total_steps")):
            action, val, logp = self.agent.act(preprocess_obs(obs, self.device), self.step, eval_mode=False)
            next_obs, reward, done, misc = self.env.step(action)

            self.episode_returns[self.ep_idx].append(reward)
            self.next_step()
            self.replay_storage.add({"obs": obs, "act": action, "rew": reward, "val": val, "logp": logp})

            # Update obs (critical!)
            obs = next_obs
            if self.checkpoint_now:
                self.agent.save_checkpoint(self.experiment_directory, self.step)

            if done or self.episode_timeout:
                self.replay_storage.finish_episode()
                self.write_logs()
                self.ep_idx += 1
                obs = self.env.reset()

            if self.update_now:
                metrics = self.agent.update(self.replay_iter, self.step)
                if self.get("use_wandb"):
                    self.wandb_log(**metrics)



if __name__ == "__main__":
    # Default cmdline args Flo
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "experiments/base",
            "--inherit",
            "templates/base",
            "--macro",
            "templates/agents/ppo.yml"
        ]

    Trainer().run()

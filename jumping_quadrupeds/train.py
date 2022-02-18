import torch
import sys
import os
import numpy as np
import submitit
from tqdm import trange
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union
from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
from jumping_quadrupeds.buffer import ReplayBuffer, ReplayBufferStorage
from jumping_quadrupeds.env import make_env
from jumping_quadrupeds.utils import DataSpec, preprocess_obs, set_seed, buffer_loader_factory
from jumping_quadrupeds.ppo.agent import PPOAgent
from jumping_quadrupeds.drqv2.agent import DrQV2Agent
from jumping_quadrupeds.spr.agent import SPRAgent
from jumping_quadrupeds.mae.agent import MAEAgent


class Trainer(BaseExperiment, WandBMixin, IOMixin, submitit.helpers.Checkpointable):
    WANDB_ENTITY = "jumping_quadrupeds"
    WANDB_PROJECT = "rl-encoder-test"

    def __init__(self, skip_setup=False):
        super(Trainer, self).__init__()
        if not skip_setup:
            self.auto_setup()

    def _build(self):
        if self.get("use_wandb"):
            self.initialize_wandb()

        # env setup
        seed = set_seed(seed=self.get("seed"))
        self.env = make_env(seed=seed, **self.get("env/kwargs"))
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        replay_loader_kwargs = self.get("buffer/kwargs")
        replay_loader_kwargs.update({"replay_dir": replay_dir, "data_specs": data_specs})
        breakpoint()
        self.replay_storage = ReplayBufferStorage(data_specs, replay_dir)
        self.replay_loader = buffer_loader_factory(**replay_loader_kwargs)
        self._replay_iter = None

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _build_agent(self):
        if self.get("agent/name") == "ppo":
            self.agent = PPOAgent(
                self.env.observation_space,
                self.env.action_space,
                self.get("use_wandb"),
                device=self.device,
                **self.get("agent/kwargs"),
            )
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
            raise ValueError(
                f"Unknown agent {self.get('agent/name')}. Have you specified an agent to use a la ` --macro templates/agents/ppo.yml' ? "
            )
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
        ep_len_gt_nstep = len(self.episode_returns[self.ep_idx]) >= (self.get("buffer/kwargs/nstep", 0) + 1)
        return update_every and gt_seed_frames and ep_len_gt_nstep

    @property
    def checkpoint_best(self):
        if self.mean_return > self.read_from_cache("best_ep_mean_return", -10000.0):
            self.write_to_cache("best_ep_mean_return", self.mean_return)
            return True
        return False

    @property
    def episode_rets(self):
        ep_rets = list(dict(self.episode_returns).values())
        print(f"{self.ep_idx}, {np.array(ep_rets[-1]).mean()}")
        ep_rets = np.array([xi + [0] * (self.get("env/kwargs/max_ep_len") - len(xi)) for xi in ep_rets])
        return ep_rets

    @property
    def mean_return(self):
        return np.mean(self.episode_rets.sum(axis=1))

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

    def compute_env_specific_metrics(self, metrics):
        if "CarRacing" in self.get("env/kwargs/name"):
            metrics.update(
                {
                    "act-mean-turn": metrics["update_actor_action_mean"][0],
                    "act-mean-gas": metrics["update_actor_action_mean"][1],
                    "act-mean-brake": metrics["update_actor_action_mean"][2],
                    "act-std-turn": metrics["update_actor_action_std"][0],
                    "act-std-gas": metrics["update_actor_action_std"][1],
                    "act-std-brake": metrics["update_actor_action_std"][2],
                }
            )
            if len(metrics["update_actor_action_mean"]) > 3:
                metrics["act-mean-view"] = metrics["update_actor_action_mean"][3]

        return metrics

    def save(
        self,
        checkpoint_path: Optional[str] = None,
        is_latest: bool = False,
        is_best: bool = False,
    ):
        if checkpoint_path is None:
            if is_latest:
                checkpoint_path = os.path.join(self.checkpoint_directory, "checkpoint_latest.pt")
            elif is_best:
                checkpoint_path = os.path.join(self.checkpoint_directory, "checkpoint_best.pt")
            else:
                return
        self.agent.save_checkpoint(checkpoint_path)

    @register_default_dispatch
    def __call__(self):
        self._build()
        obs = self.env.reset()
        self.replay_storage.add({"obs": np.array(obs)})

        for _ in trange(self.get("total_steps")):
            action, val, logp = self.agent.act(preprocess_obs(obs, self.device), self.step, eval_mode=False)
            next_obs, reward, done, misc = self.env.step(action)
            self.episode_returns[self.ep_idx].append(reward)
            self.next_step()
            self.replay_storage.add({"obs": np.array(obs), "act": action, "rew": reward, "val": val, "logp": logp})

            if done or self.episode_timeout:
                print("saving episode")
                self.replay_storage.finish_episode()
                self.write_logs()
                self.ep_idx += 1
                self.save(is_best=self.checkpoint_best)
                obs = self.env.reset()
                self.replay_storage.add({"obs": np.array(obs)})

            if self.update_now:
                metrics = self.agent.update(self.replay_iter, self.step)
                metrics = self.compute_env_specific_metrics(metrics)
                if self.get("use_wandb"):
                    self.wandb_log(**metrics)

            # Update obs (critical!)
            obs = next_obs
            if self.checkpoint_now:
                self.save(checkpoint_path=self.checkpoint_path)
                self.save(is_latest=True)


if __name__ == "__main__":
    # Default cmdline args Flo
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "experiments/base",
            "--inherit",
            "templates/base",
            "--macro",
            "templates/agents/ppo.yml",
        ]

    Trainer().run()

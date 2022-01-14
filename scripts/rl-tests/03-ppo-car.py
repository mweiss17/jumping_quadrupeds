import torch
import sys
import numpy as np
import submitit
from tqdm import trange
from collections import defaultdict
from pathlib import Path
from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
from jumping_quadrupeds.rl.buffer import make_replay_loader, ReplayBufferStorage
from jumping_quadrupeds.env import make_env
from jumping_quadrupeds.utils import DataSpec



class TrainPPOConv(
    BaseExperiment, WandBMixin, IOMixin, submitit.helpers.Checkpointable
):
    WANDB_ENTITY = "jumping_quadrupeds"
    WANDB_PROJECT = "rl-encoder-test"

    def __init__(self):
        super(TrainPPOConv, self).__init__()
        self.auto_setup()

        if self.get("use_wandb"):
            self.initialize_wandb()

        seed = self.get("seed", 58235)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # env setup
        self.env = make_env(
            env_name=self.get("env_name"),
            seed=seed,
        )
        self.device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.replay_loader = make_replay_loader(
            replay_dir, self.get("buffer/on_policy"), self.get("buffer/kwargs")
        )
        self._replay_iter = None

    def _build_agent(self):
        if self.get("agent/name") == "ppo":
            from jumping_quadrupeds.rl.ppo.ppo import PPO
            self.agent = PPO(self.env.observation_space, self.env.action_space, self.get("use_wandb"), device=self.device, **self.get("agent/kwargs"))
            if self.get("use_wandb"):
                self.wandb_watch(self.agent.ac.pi, log_freq=1)
                self.wandb_watch(self.agent.ac.v, log_freq=1)

        elif self.get("agent/name") == "drqv2":
            from jumping_quadrupeds.rl.drqv2.agent import DrQV2Agent
            self.agent = DrQV2Agent(
                self.env.observation_space,
                self.env.action_space,
                self.device,
                **self.get("agent/kwargs"),
            )

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def checkpoint_now(self):
        if (self.step % self.get("save_freq") == 0) or (
            self.step == self.get("total_steps") - 1
        ):
            return True
        return False

    @property
    def episode_timeout(self):
        return len(self.episode_returns[self.ep_idx]) == self.get("max_ep_len")

    @property
    def log_now(self):
        return (len(self.episode_returns) % self.get("log_ep_freq") == 0) and self.get("use_wandb")

    @property
    def update_now(self):
        update_every = self.step % self.get("agent/kwargs/update_every_steps") == 0 and self.step > 0
        gt_seed_frames = self.step >= self.get("agent/kwargs/num_seed_frames")
        ep_len_gt_nstep = len(self.episode_returns[self.ep_idx]) > self.get("buffer/kwargs/nstep", 1)
        return update_every and gt_seed_frames and ep_len_gt_nstep

    def write_logs(self):
        if self.log_now:
            ep_rets = list(dict(self.episode_returns).values())
            ep_rets = np.array([xi + [0] * (self.get("max_ep_len") - len(xi)) for xi in ep_rets])
            full_episodic_return = ep_rets.sum(axis=1)
            self.wandb_log(
                **{
                    "Episode mean reward": np.mean(full_episodic_return),
                    "Episode return": full_episodic_return[-1],
                    "Episode mean length": np.mean([len(ep) for ep in self.episode_returns.values()]),
                    "Number of Episodes": len(self.episode_returns),
                }
            )

            # Record video of the updated policy
            self.env.send_wandb_video()


    @register_default_dispatch
    def __call__(self):
        obs = self.env.reset()

        for _ in trange(self.get("total_steps")):
            # Rollout
            action, val, logp = self.agent.act(obs, self.step, eval_mode=True)
            next_obs, reward, done, misc = self.env.step(action)

            self.episode_returns[self.ep_idx].append(reward)
            self.next_step()

            self.replay_storage.add({"obs": obs, "act": action, "rew": reward, "val": val, "logp": logp})

            # Update obs (critical!)
            obs = next_obs

            # Checkpoint
            if self.checkpoint_now:
                self.agent.save_checkpoint(self.experiment_directory, self.step)

            # Update the agent
            if self.update_now:
                metrics = self.agent.update(self.replay_iter, self.step)
                self.wandb_log(**metrics)

            # Finish the episode
            if done or self.episode_timeout:
                self.replay_storage.finish_episode()
                self.write_logs()
                self.ep_idx += 1
                obs = self.env.reset()




if __name__ == "__main__":
    # Default cmdline args Flo
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "experiments/ppo-car",
            "--inherit",
            "templates/ppo-car",
        ]

    TrainPPOConv().run()

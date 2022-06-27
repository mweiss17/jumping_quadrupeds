import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import submitit
import torch
from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
from tqdm import trange

from jumping_quadrupeds import tokenizers
from jumping_quadrupeds.buffer import ReplayBufferStorage
from jumping_quadrupeds.drqv2.agent import DrQV2Agent
from jumping_quadrupeds.env import make_env
from jumping_quadrupeds.smaq import networks
from jumping_quadrupeds.ppo.agent import PPOAgent
from jumping_quadrupeds.spr.agent import SPRAgent
from jumping_quadrupeds.utils import DataSpec, preprocess_obs, set_seed, buffer_loader_factory


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.episode_returns = []

        self._build_buffer()
        self._build_agent()

    def _build_buffer(self):
        data_specs = [
            DataSpec("observation", self.env.observation_space.shape, np.uint8),
            DataSpec("action", self.env.action_space.shape, self.env.action_space.dtype),
            DataSpec("reward", (1,), np.float32),
            DataSpec("discount", (1,), np.float32),
        ]

        replay_dir = Path(self.experiment_directory + "/Logs/buffer")
        replay_loader_kwargs = self.get("buffer/kwargs")
        replay_loader_kwargs.update({"replay_dir": replay_dir, "data_specs": data_specs})
        self.replay_loader = buffer_loader_factory(**replay_loader_kwargs)
        if self.get("buffer/kwargs/buffer_type") == "on-policy":
            self.replay_storage = self.replay_loader.dataset
        else:
            self.replay_storage = ReplayBufferStorage(data_specs, replay_dir)
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
            tokenizer_cls = getattr(tokenizers, self.get("tokenizer/cls"))
            self.agent = tokenizer_cls(
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
        elif self.get("agent/name") == "smaq":
            # Make the tokenizer
            tokenizer_cls = getattr(tokenizers, self.get("tokenizer/cls"))
            tokenizer_kwargs = dict(self.get("tokenizer/kwargs"))
            tokenizer_kwargs.update({"obs_space": self.env.observation_space, "dim": self.get("encoder/kwargs/dim")})
            tokenizer = tokenizer_cls(**tokenizer_kwargs)

            # Make the encoder
            encoder_cls = getattr(networks, self.get("encoder/cls"))
            enc_kwargs = dict(self.get("encoder/kwargs"))
            enc_kwargs.update({"dim": self.get("model/kwargs/action_encoding_dim") + enc_kwargs["dim"]})
            encoder = encoder_cls(**enc_kwargs).to(self.device)

            # Make the decoder
            decoder_cls = getattr(networks, self.get("decoder/cls"))
            decoder = decoder_cls(**self.get("decoder/kwargs")).to(self.device)

            # Make the model
            model_cls = getattr(networks, self.get("model/cls"))
            model_kwargs = dict(self.get("model/kwargs"))
            if self.env.action_space.__class__.__name__ == "Discrete":
                action_dim = self.env.action_space.n
            else:
                action_dim = self.env.action_space.shape[0]
            model_kwargs.update(
                {
                    "tokenizer": tokenizer,
                    "encoder": encoder,
                    "decoder": decoder,
                    "action_dim": action_dim,
                    "device": self.device,
                }
            )
            model = model_cls(**model_kwargs).to(self.device)
            model_ema = model_cls(**model_kwargs).to(self.device)
            model_ema.load_state_dict(model.state_dict())

            # Make the agent
            from jumping_quadrupeds.smaq import agent

            agent_cls = getattr(agent, self.get("agent/cls"))
            agent_kwargs = dict(self.get("agent/kwargs"))
            agent_kwargs.update(
                {
                    "action_space": self.env.action_space,
                    "model": model,
                    "model_ema": model_ema,
                    "device": self.device,
                },
            )

            self.agent = agent_cls(**agent_kwargs)
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
    def global_frame(self):
        return self.step * self.get("env/kwargs/action_repeat")

    @property
    def update_now(self):
        if self.step <= 0:
            return False
        update_every = (self.step % self.get("agent/kwargs/update_every_steps")) == 0
        gt_seed_frames = self.global_frame > self.get("num_seed_frames")
        ep_len_gt_nstep = self.replay_storage.cur_ep_len >= self.get("buffer/kwargs/nstep", 0)
        return update_every and gt_seed_frames and ep_len_gt_nstep

    @property
    def checkpoint_best(self):
        if self.episode_returns[-1] > self.read_from_cache("best_episode_return", -10000.0):
            self.write_to_cache("best_episode_return", self.episode_returns[-1])
            return True
        return False

    def write_logs(self, episode_reward):
        self.episode_returns.append(episode_reward)
        logs = {
            "Episode return": episode_reward,
            "Mean Episode Return": np.mean(self.episode_returns),
            "Number of Episodes": len(self.episode_returns),
        }
        if self.get("use_wandb"):
            self.wandb_log(**logs)
            self.env.send_wandb_video()
        else:
            print(logs)

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
        time_step = self.env.reset()
        self.replay_storage.add(time_step)
        episode_reward = 0
        action = torch.zeros(self.env.action_space.shape)
        for _ in trange(self.get("total_steps")):
            obs = preprocess_obs(time_step.observation, self.device)
            action, val, logp = self.agent.act(obs, action, self.step, eval_mode=False)
            time_step = self.env.step(action)

            self.next_step()
            self.replay_storage.add(time_step, val, logp)

            if self.update_now:

                metrics = self.agent.update(self.replay_iter, self.step)

                if (self.step % self.get("log_every")) == 0:
                    metrics = self.compute_env_specific_metrics(metrics)
                    if self.get("use_wandb"):
                        if self.get("agent/name") == "smaq":
                            self.wandb_log_image("gt_img_viz", metrics["gt_img"])
                            self.wandb_log_image("pred_img_viz", metrics["pred_img"])
                            self.wandb_log_image("gt_masked_img_viz", metrics["gt_masked_img"])
                            del metrics["gt_img"]
                            del metrics["pred_img"]
                            del metrics["gt_masked_img"]
                        self.wandb_log(**metrics)
                    else:
                        print(metrics)

            if self.checkpoint_now:
                self.save(checkpoint_path=self.checkpoint_path)
                self.save(is_latest=True)

            if time_step.last():
                self.write_logs(episode_reward)
                episode_reward = 0
                self.save(is_best=self.checkpoint_best)
                time_step = self.env.reset()
                self.replay_storage.add(time_step)

            episode_reward += time_step.reward


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

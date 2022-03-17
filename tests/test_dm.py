import os
import pytest
import yaml
import torch
from addict import Addict
from jumping_quadrupeds.utils import DataSpec, preprocess_obs, set_seed, buffer_loader_factory
from jumping_quadrupeds.env import make_env
from jumping_quadrupeds.train import Trainer


def _build_config(env_file, agent_file, loader):
    config_path = os.path.join(os.getcwd(), "templates/base/Configurations/train_config.yml")
    env_config_path = os.path.join(os.getcwd(), "templates/tasks", env_file)
    agent_config_path = os.path.join(os.getcwd(), "templates/agents", agent_file)
    config = Addict(yaml.load(open(config_path, "r"), Loader=loader))
    env_config = Addict(yaml.load(open(env_config_path, "r"), Loader=loader))
    agent_config = Addict(yaml.load(open(agent_config_path, "r"), Loader=loader))
    config.update(**env_config)
    config.update(**agent_config)
    return config


def run_test(env_file, agent_file, experiment_directory, total_steps):
    trainer = Trainer(skip_setup=True)
    config = _build_config(env_file=env_file, agent_file=agent_file, loader=trainer.get_loader())
    trainer._config = config
    trainer.experiment_directory = experiment_directory
    trainer.set("use_wandb", False)
    trainer.set("total_steps", total_steps)
    trainer.record_args()
    trainer._build()
    trainer()


def test_dm_mae():
    env_file = "dm-cartpole-balance.yml"
    agent_file = "mae.yml"
    experiment_directory = "tmp/test_quadrupeds/dm_mae"
    total_steps = 4000
    run_test(env_file, agent_file, experiment_directory, total_steps)


def test_dm_ppo():
    env_file = "dm-cartpole-balance.yml"
    agent_file = "ppo.yml"
    experiment_directory = "tmp/test_quadrupeds/dm_ppo"
    total_steps = 4000
    run_test(env_file, agent_file, experiment_directory, total_steps)


def test_car_ppo():
    env_file = "gym-car-racing.yml"
    agent_file = "ppo.yml"
    experiment_directory = "tmp/test_quadrupeds/car_racing_ppo"
    total_steps = 4000
    run_test(env_file, agent_file, experiment_directory, total_steps)


def test_car_mae():
    env_file = "gym-car-racing.yml"
    agent_file = "mae.yml"
    experiment_directory = "tmp/test_quadrupeds/car_racing_mae"
    total_steps = 4000
    run_test(env_file, agent_file, experiment_directory, total_steps)


def test_car_drqv2():
    env_file = "gym-car-racing.yml"
    agent_file = "drqv2.yml"
    experiment_directory = "tmp/test_quadrupeds/car_racing_drqv2"
    total_steps = 4000
    run_test(env_file, agent_file, experiment_directory, total_steps)


def test_dm_drqv2():
    env_file = "dm-cartpole-balance.yml"
    agent_file = "drqv2.yml"
    experiment_directory = "tmp/test_quadrupeds/dm_drqv2"
    total_steps = 4000
    run_test(env_file, agent_file, experiment_directory, total_steps)


if __name__ == "__main__":
    pytest.main()

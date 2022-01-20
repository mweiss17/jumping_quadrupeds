# Jumping Quadrupeds

This is a collection of deep reinforcement methods built to solve partially-observable environments.

## Installation

- `git clone git@github.com:mweiss17/jumping_quadrupeds.git`
- `cd jumping_quadrupeds` 
- `pip install -e .`

## Usage
There are several methods implemented in the jumping_quadrupeds, including PPO, DRQ-v2, SPR, and an ETH world-model method.
In order to use each, you need to specify the environment, the agent, and the training parameters.

```python
python3 scripts/rl-tests/01-train.py experiments/ppo-car --inherit templates/base --macro templates/agents/ppo.yml --config.use_wandb True
```

# Jumping Quadrupeds repository

## Installation

- `git clone git@github.com:mweiss17/jumping_quadrupeds.git`
- `cd jumping_quadrupeds` 
- `pip install -e .`


## Train the VAE
There are two ways to train the VAE -- by doing a normal exploratory run or with [WandB's hyperparameter sweep](https://docs.wandb.com/sweeps).

### Exploratory run
`python3 scripts/eth-paper/02-train-vae.py experiments/vae --inherit templates/vae`

### Hyperparameter Search
`python3 scripts/eth-paper/02-train-vae.py experiments/sweep-vae --inherit experiments/vae --dispatch setup_wandb_sweep --wandb.sweep_config templates/vae/Configurations/sweep_config.yml`

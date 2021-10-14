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
First we configure a sweep by running a command like:
`python3 scripts/eth-paper/02-train-vae.py experiments/sweep-vae --inherit templates/vae --dispatch setup_wandb_sweep --wandb.sweep_config templates/vae/Configurations/sweep_config.yml`

Here, we are reading from some folder `templates/vae` which contains a `Configurations/train_config.yaml` specifying the experimental configurations, we then construct the sweep configuration and write it into `experiments/sweep-vae`. We also need to specify the parameters of our sweep in `templates/vae/Configurations/sweep_config.yml`.

Then we can run jobs using that configuration like, where run_num is some job index: 

`python scripts/eth-paper/02-train-vae.py experiments/sweep-vae --inherit experiments/sweep-vae --wandb.sweep True -njobs 2 `

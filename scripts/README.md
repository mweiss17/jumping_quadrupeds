# A quick overview of the scripts
Currently there are two script folders (which probably should be merged).
Scripts in `eth-paper/` purport to replicate the procedure in ["Learning a State Representation and Navigation in Cluttered and Dynamic Environments"](https://arxiv.org/pdf/2103.04351.pdf). 
Scripts in `rl-tests/` are focused on the RL-portion of that work, and include some additional RL experiments.

### `eth-paper/`
- **01-gather-random-rollouts.py**: Boots several simulators with the CarRacing-v0 OpenAI Gym environment and gathers random rollouts from the environment (cropping the initial dozen frames).
- **02-train-vae.py**: Trains a VAE model on the output of `01-gather-random-rollouts.py` which are stored as a bajillion images on disk in a folder.
- **03-train-lstm.py**: Trains an LSTM on the rollouts generated by running PPO (03-ppo-car.py).

### `rl-tests/`
- **01-simple-ppo-test.py**: Trains an MLP actor-critic model on cartpole
- **02-simple-ppo-test.py**: Trains an MLP actor-critic model on cartpole with a shared encoder
- **03-base.py**: Trains a convolutional actor-critic model on CarRacing. We can parameterize this experiment with different templates.

## Running all them scripts

### Gather Random Rollouts:

`python3 scripts/eth-paper/01-gather-random-rollouts.py`

Which will generate a bunch of images in a folder called `random-rollouts-50k/`

### Train the VAE:

Next, we run `python3 scripts/eth-paper/02-train-vae.py experiments/train-vae-enc --inherit templates/vae/`

The folder `experiments/train-vae-enc` is where the experiment logs, configs, weights, etc are dumped. 

The argument `--inherit templates/vae/` states which yaml configuration file to read from.

### Train the PPO model with the VAE encoder:

N.B. There are several arguments for `03-ppo-car.py` which enable us to experiment with different settings.

- Randomly Initialized Encoder
    - `python3 scripts/rl-tests/03-ppo-car.py experiments/ppo-init-1 --inherit templates/ppo-car`
- Finetune using encoder from pre-trained VAE
    - `python3 scripts/rl-tests/03-ppo-car.py experiments/ppo-finetune-1 --inherit templates/ppo-car --config.vae_enc_checkpoint experiments/train-vae-enc/Weights/enc-40.pt`
- Freeze and finetune using encoder from pre-trained VAE
    - `python3 scripts/rl-tests/03-ppo-car.py experiments/ppo-freeze-1 --inherit templates/ppo-car  --config.freeze_encoder True --config.vae_enc_checkpoint experiments/train-vae-enc/Weights/enc-40.pt`
- Randomly Initialized Shared Encoder (Not finished)
    - `python3 scripts/rl-tests/03-ppo-car.py experiments/ppo-shared-1 --inherit templates/ppo-car  --config.shared_encoder True`


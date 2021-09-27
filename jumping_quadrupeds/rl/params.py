import attr
import torch


@attr.s
class PpoParams(object):
    # env_name (str): Environment name, used for logging
    env_name = attr.ib(default="")

    # verbose (bool): should PPO print verbose debugging messages to stdout?
    verbose = attr.ib(default=False)

    # steps_per_epoch (int): Number of steps of interaction (state-action pairs) for the agent and the environment
    #  in each epoch.
    steps_per_epoch = attr.ib(default=4000)

    # epochs (int): Number of epochs of interaction (equivalent to number of policy updates) to perform.
    epochs = attr.ib(default=50)

    # gamma (float): Discount factor. (Always between 0 and 1.)
    gamma = attr.ib(default=0.99)

    # clip_ratio (float): Hyperparameter for clipping in the policy objective.
    #  Roughly: how far can the new policy go from the old policy while
    #  still profiting (improving the objective function)? The new policy
    #  can still go farther than the clip_ratio says, but it doesn't help
    #  on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
    #  denoted by :math:`\epsilon`.
    clip_ratio = attr.ib(default=0.2)

    # pi_lr (float): Learning rate for policy optimizer.
    pi_lr = attr.ib(default=3e-4)

    # vf_lr (float): Learning rate for value function optimizer.
    vf_lr = attr.ib(default=1e-3)

    # train_pi_iters (int): Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping
    #  may cause optimizer to take fewer than this.)
    train_pi_iters = attr.ib(default=80)

    # train_v_iters (int): Number of gradient descent steps to take on value function per epoch.
    train_v_iters = attr.ib(default=80)

    # lam (float): Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
    lam = attr.ib(default=0.97)

    # max_ep_len (int): Maximum length of trajectory / episode / rollout. This is only used by some variable-length
    #  episode environments
    max_ep_len = attr.ib(default=1000)

    # target_kl (float): Roughly what KL divergence we think is appropriate between new and old policies after an
    #  update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)
    target_kl = attr.ib(default=0.01)

    # save_freq (int): save policy every n epochs
    save_freq = attr.ib(default=10)

    # log_ep_freq (int): log rewards every n episodes
    log_ep_freq = attr.ib(default=10)

    # rew_smooth_len (int): rewards are accumulated in a ring buffer and this determines the size
    rew_smooth_len = attr.ib(default=10)

    # device (str): device on which the policy network resides
    device = attr.ib(default="cuda" if torch.cuda.is_available() else "cpu")

    # seed (int): random seed
    seed = attr.ib(default=1)

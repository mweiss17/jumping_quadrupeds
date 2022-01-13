import torch
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
from rlpyt.agents.pg.gaussian import GaussianPgAgent


def to_onehot(indexes, num, dtype=None):
    """Converts integer values in multi-dimensional tensor ``indexes``
    to one-hot values of size ``num``; expanded in an additional
    trailing dimension."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot


def from_onehot(onehot, dim=-1, dtype=None):
    """Argmax over trailing dimension of tensor ``onehot``. Optional return
    dtype specification."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes



def valid_mean(tensor, valid=None, dim=None):
    """Mean of ``tensor``, accounting for optional mask ``valid``,
    optionally along a dimension."""
    dim = () if dim is None else dim
    if valid is None:
        return tensor.mean(dim=dim)
    valid = valid.type(tensor.dtype)  # Convert as needed.
    return (tensor * valid).sum(dim=dim) / valid.sum(dim=dim)


class DiscreteMixin:
    """Conversions to and from one-hot."""

    def __init__(self, dim, dtype=torch.long, onehot_dtype=torch.float):
        self._dim = dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype

    @property
    def dim(self):
        return self._dim

    def to_onehot(self, indexes, dtype=None):
        """Convert from integer indexes to one-hot, preserving leading dimensions."""
        return to_onehot(indexes, self._dim, dtype=dtype or self.onehot_dtype)

    def from_onehot(self, onehot, dtype=None):
        """Convert from one-hot to integer indexes, preserving leading dimensions."""
        return from_onehot(onehot, dtpye=dtype or self.dtype)

class EpsilonGreedy(DiscreteMixin):
    """For epsilon-greedy exploration from state-action Q-values."""

    def __init__(self, epsilon=1, **kwargs):
        super().__init__(**kwargs)
        self._epsilon = epsilon

    def sample(self, q):
        """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
        B will apply across the Batch dimension (same epsilon for all T)."""
        arg_select = torch.argmax(q, dim=-1)
        mask = torch.rand(arg_select.shape) < self._epsilon
        arg_rand = torch.randint(low=0, high=q.shape[-1], size=(mask.sum(),))
        arg_select[mask] = arg_rand
        return arg_select

    @property
    def epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon):
        """Assign value for epsilon (can be vector)."""
        self._epsilon = epsilon


    def perplexity(self, dist_info):
        """Exponential of the entropy, maybe useful for logging."""
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid=None):
        """In case some sophisticated mean is needed (e.g. internally
        ignoring select parts of action space), can override."""
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        """Exponential of the entropy, maybe useful for logging."""
        return valid_mean(self.perplexity(dist_info), valid)

class CategoricalEpsilonGreedy(EpsilonGreedy):
    """For epsilon-greedy exploration from distributional (categorical)
    representation of state-action Q-values."""

    def __init__(self, z=None, **kwargs):
        super().__init__(**kwargs)
        self.z = z

    def sample(self, p, z=None):
        """Input p to be shaped [T,B,A,P] or [B,A,P], A: number of actions, P:
        number of atoms.  Optional input z is domain of atom-values, shaped
        [P].  Vector epsilon of lenght B will apply across Batch dimension."""
        q = torch.tensordot(p, z or self.z, dims=1)
        return super().sample(q)

    def set_z(self, z):
        """Assign vector of bin locations, distributional domain."""
        self.z = z



class SPRAgent(GaussianPgAgent): # AtariCatDqnAgent
    """Agent for Categorical DQN algorithm with search."""

    def __init__(self, eval=False, **kwargs):
        """Standard init, and set the number of probability atoms (bins)."""
        super().__init__(**kwargs)
        self.eval = eval


    def __call__(self, observation, prev_action, prev_reward, train=False):
        """Returns Q-values for states/observations (with grad)."""
        if train:
            model_inputs = (observation, prev_action, prev_reward) # buffer_to
            return self.model(*model_inputs, train=train)
        else:
            prev_action = self.distribution.to_onehot(prev_action)
            model_inputs = (observation, prev_action, prev_reward) # buffer_to
            return self.model(*model_inputs).cpu()

    def initialize(self,
                   env_spaces,
                   share_memory=False,
                   global_B=1,
                   env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        self.distribution = CategoricalEpsilonGreedy(dim=env_spaces.action.n, z=torch.linspace(-1, 1, self.n_atoms))  # z placeholder for init.

        # Overwrite distribution.
        self.search = SPRActionSelection(self.model, self.distribution)

    def to_device(self, cuda_idx=None):
        """Moves the model to the specified cuda device, if not ``None``.  If
        sharing memory, instantiates a new model to preserve the shared (CPU)
        model.  Agents with additional model components (beyond
        ``self.model``) for action-selection or for use during training should
        extend this method to move those to the device, as well.
        Typically called in the runner during startup.
        """
        super().to_device(cuda_idx)
        self.search.to_device(cuda_idx)
        self.search.network = self.model

    def eval_mode(self, itr):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        super().eval_mode(itr)
        self.search.epsilon = self.distribution.epsilon
        self.search.network.head.set_sampling(False)
        self.itr = itr

    def sample_mode(self, itr):
        """Extend method to set epsilon for sampling (including annealing)."""
        super().sample_mode(itr)
        self.search.epsilon = self.distribution.epsilon
        self.search.network.head.set_sampling(True)
        self.itr = itr

    def train_mode(self, itr):
        super().train_mode(itr)
        self.search.network.head.set_sampling(True)
        self.itr = itr

    @torch.no_grad()
    def act(self, observation, step, eval_mode=None): # was step
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        if eval_mode:
            self.eval_mode(step)
        else:
            self.train_mode(step)
        action, p = self.search.run(observation.to(self.search.device))
        p = p.cpu()
        action = action.cpu()
        return (action, p, None)


class SPRActionSelection(torch.nn.Module):
    def __init__(self, network, distribution, device="cpu"):
        super().__init__()
        self.network = network
        self.epsilon = distribution._epsilon
        self.device = device
        self.first_call = True

    def to_device(self, idx):
        self.device = idx

    @torch.no_grad()
    def run(self, obs):
        while len(obs.shape) <= 4:
            obs.unsqueeze_(0)
        obs = obs.to(self.device).float() / 255.

        value = self.network.select_action(obs)
        action = self.select_action(value)
        # Stupid, stupid hack because rlpyt does _not_ handle batch_b=1 well.
        if self.first_call:
            action = action.squeeze()
            self.first_call = False
        return action, value.squeeze()

    def select_action(self, value):
        """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
        B will apply across the Batch dimension (same epsilon for all T)."""
        arg_select = torch.argmax(value, dim=-1)
        mask = torch.rand(arg_select.shape, device=value.device) < self.epsilon
        arg_rand = torch.randint(low=0, high=value.shape[-1], size=(mask.sum(),), device=value.device)
        arg_select[mask] = arg_rand
        return arg_select

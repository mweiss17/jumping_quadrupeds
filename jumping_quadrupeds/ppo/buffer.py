from jumping_quadrupeds.buffer import ReplayBuffer

class OnPolicyReplayBuffer(ReplayBuffer):
    def __init__(self, replay_dir=None, **kwargs):
        super().__init__(replay_dir=replay_dir, **kwargs)
        self._gae_lambda = kwargs.get("gae_lambda", 0.97)
        self.eps_idx = 0
        self.step_idx = 0
        self.total_idx = 0

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        if self.total_idx >= self._max_size:
            self.eps_idx = 0
            self.total_idx = 0
        self._samples_since_last_fetch += 1
        try:
            episode = self._episodes[self._episode_fns[self.eps_idx]]
        except Exception:
            breakpoint()
        if self.step_idx == 0:
            episode = self.compute_rets_and_advs(episode)

        sample = {
            "obs": episode["obs"][self.step_idx],
            "act": episode["act"][self.step_idx],
            "rew": episode["rew"][self.step_idx],
            "adv": episode["adv"][self.step_idx],
            "ret": episode["ret"][self.step_idx],
            "logp": episode["logp"][self.step_idx],
        }

        self.step_idx += 1
        self.total_idx += 1
        if self.step_idx == episode_len(episode) + 1:
            self.eps_idx += 1
            self.step_idx = 0
        return sample

    def compute_rets_and_advs(self, episode):
        # add zeros to the end
        rew = np.pad(episode["rew"], ((0, 1), (0, 0)), "constant")
        val = np.pad(episode["val"], ((0, 1), (0, 0)), "constant")

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rew[:-1] + self._discount * val[1:] - val[:-1]
        adv_buf = discount_cumsum(deltas, self._discount * self._gae_lambda)

        # # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        episode["adv"] = adv_buf

        # the next line computes rewards-to-go, to be targets for the value function
        ret_buf = discount_cumsum(rew, self._discount)[:-1]

        episode["ret"] = ret_buf
        return episode
from jumping_quadrupeds.buffer import ReplayBuffer

class OffPolicySequentialReplayBuffer(ReplayBuffer):
    def __init__(self, replay_dir=None, **kwargs):
        super().__init__(replay_dir=replay_dir, **kwargs)
        self.jumps = kwargs.get("jumps")

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()

        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self.jumps + 1) + 1

        obs = episode["obs"][idx - 1: idx + self.jumps - 1]
        action = episode["act"][idx: idx + self.jumps]
        reward = np.zeros_like(episode["rew"][idx: idx + self.jumps])
        discount = np.ones_like(episode["discount"][idx: idx + self.jumps])

        sample = {
            "obs": obs,
            "act": action,
            "rew": reward,
            "discount": discount,
        }

        return sample

import numpy as np
from jumping_quadrupeds.buffer import ReplayBuffer

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


class OffPolicyReplayBuffer(ReplayBuffer):
    def __init__(self, replay_dir=None, **kwargs):
        super().__init__(replay_dir=replay_dir, **kwargs)
        self._nstep = kwargs.get("nstep")

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()

        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

        obs = episode["obs"][idx - 1]
        action = episode["act"][idx]
        next_obs = episode["obs"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["rew"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["rew"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        sample = {
            "obs": obs,
            "act": action,
            "rew": reward,
            "discount": discount,
            "next_obs": next_obs,
        }

        return sample

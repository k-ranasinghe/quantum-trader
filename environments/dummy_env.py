import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DummySignalEnv(gym.Env):
    """Minimal no-op environment for loading PPO models."""
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # Action space: 8 floats between 0 and 1
        self.action_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        # Observation space: e.g. 15 features * 60 timesteps
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(60, 15), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

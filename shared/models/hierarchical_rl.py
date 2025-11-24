from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Callable


class HierarchicalRLAgent:
    """
    Hierarchical RL with meta-policy and sub-policies
    Inspired by EarnHFT (SOTA for HFT)
    """

    def __init__(
            self,
            env_fn: Callable,
            n_sub_policies: int = 3,
            meta_timesteps: int = 50000,
            sub_timesteps: int = 30000
    ):
        self.env_fn = env_fn
        self.n_sub_policies = n_sub_policies

        # Meta-policy selects which sub-policy to use
        self.meta_policy = PPO(
            "MlpPolicy",
            DummyVecEnv([env_fn]),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=0
        )

        # Sub-policies for different strategies
        self.sub_policies = [
            SAC("MlpPolicy", DummyVecEnv([env_fn]), verbose=0)
            for _ in range(n_sub_policies)
        ]

        self.meta_timesteps = meta_timesteps
        self.sub_timesteps = sub_timesteps

    def train(self):
        # Train meta-policy
        self.meta_policy.learn(total_timesteps=self.meta_timesteps)

        # Train sub-policies
        for sub_policy in self.sub_policies:
            sub_policy.learn(total_timesteps=self.sub_timesteps)

    def predict(self, observation, deterministic=True):
        # Meta-policy selects sub-policy
        meta_action, _ = self.meta_policy.predict(observation, deterministic=deterministic)
        sub_policy_idx = int(meta_action) % self.n_sub_policies

        # Selected sub-policy makes decision
        action, _ = self.sub_policies[sub_policy_idx].predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        self.meta_policy.save(f"{path}_meta")
        for i, sub_policy in enumerate(self.sub_policies):
            sub_policy.save(f"{path}_sub_{i}")

    def load(self, path: str):
        self.meta_policy = PPO.load(f"{path}_meta")
        self.sub_policies = [
            SAC.load(f"{path}_sub_{i}")
            for i in range(self.n_sub_policies)
        ]
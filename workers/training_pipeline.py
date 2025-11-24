from typing import Dict, Optional
import mlflow
import numpy as np
import optuna
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch


class TrainingPipeline:
    """
    Comprehensive training pipeline with hyperparameter optimization
    """

    def __init__(
            self,
            env_fn,
            mlflow_experiment: str = "quantum-trading"
    ):
        self.env_fn = env_fn
        mlflow.set_experiment(mlflow_experiment)

    def optimize_hyperparameters(
            self,
            n_trials: int = 50,
            timeout: int = 7200  # 2 hours
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna
        """

        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048])
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

            # Create model
            model = PPO(
                "MlpPolicy",
                self.env_fn(),
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                batch_size=batch_size,
                verbose=0
            )

            # Validate observations
            obs, _ = self.env_fn().reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            if torch.isnan(obs).any():
                print(f"NaN values detected in observations: {obs}")
                return -float('inf')  # Return a very low score (or you can return 0 or negative value)

            # Train
            model.learn(total_timesteps=50000)

            # Evaluate
            env = self.env_fn()
            obs, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward

            return env.get_portfolio_stats()['sharpe_ratio']

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return study.best_params

    def train_model(
            self,
            model_type: str = "ppo",
            hyperparameters: Optional[Dict] = None,
            total_timesteps: int = 200000,
            eval_freq: int = 10000
    ):
        """
        Train model with MLflow logging
        """
        with mlflow.start_run():
            # Log parameters
            params = hyperparameters or self._get_default_params(model_type)
            mlflow.log_params(params)

            # Create model
            if model_type == "ppo":
                model = PPO("MlpPolicy", self.env_fn(), **params, verbose=1)
            elif model_type == "sac":
                # We must use SAC only for continuous action spaces.
                if isinstance(self.env_fn().action_space, spaces.Discrete):
                    raise ValueError("SAC cannot be used with a discrete action space.")
                model = SAC("MlpPolicy", self.env_fn(), **params, verbose=1)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Check for NaN values in the environment
            env = self.env_fn()
            obs, _ = env.reset()
            if np.isnan(obs).any():
                print(f"Warning: NaN values detected in environment reset: {obs}")
                return None, None

            # Callbacks
            eval_env = self.env_fn()
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path='./checkpoints/',
                name_prefix=f'{model_type}_model'
            )

            # Train
            model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, checkpoint_callback]
            )

            # Final evaluation
            stats = self._evaluate_model(model)

            # Log metrics
            mlflow.log_metrics(stats)

            # Log model
            mlflow.pytorch.log_model(model.policy, "model")

            return model, stats

    def _get_default_params(self, model_type: str) -> Dict:
        """Get default hyperparameters"""
        if model_type == "ppo":
            return {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10
            }
        elif model_type == "sac":
            return {
                'learning_rate': 3e-4,
                'buffer_size': 100000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99
            }

    def _evaluate_model(self, model) -> Dict:
        """Comprehensive model evaluation"""
        env = self.env_fn()
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        return env.get_portfolio_stats()
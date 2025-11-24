import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import xgboost as xgb

from shared.environments.trading_env import Actions


@dataclass
class ModelPrediction:
    action: int
    confidence: float
    probabilities: np.ndarray
    model_name: str


class EnsembleController:
    """
    Advanced ensemble controller for multi-model trading
    Implements meta-learning and adaptive weighting
    """

    def __init__(
            self,
            models: Dict,
            regime_detector,
            meta_labeler: xgb.XGBClassifier,
            lookback_window: int = 100
    ):
        self.models = models
        self.regime_detector = regime_detector
        self.meta_labeler = meta_labeler
        self.lookback_window = lookback_window

        # Track model performance
        self.model_performance = {name: [] for name in models.keys()}
        self.adaptive_weights = {name: 1.0 for name in models.keys()}

    def predict(
            self,
            observation: np.ndarray,
            market_data: pd.DataFrame,
            return_metadata: bool = False
    ) -> Tuple[int, float, Dict]:
        """
        Generate ensemble prediction

        Returns:
            action: Trading action (0=SELL, 1=HOLD, 2=BUY)
            confidence: Confidence score [0, 1]
            metadata: Additional information
        """
        # Detect market regime
        regime_features = self._extract_regime_features(market_data)
        current_regime = self.regime_detector.predict(regime_features.reshape(1, -1))[0]

        # Get predictions from all models
        predictions = []
        for model_name, model in self.models.items():
            try:
                pred = self._get_model_prediction(model, observation, model_name)
                predictions.append(pred)
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                continue

        if len(predictions) == 0:
            return 1, 0.0, {'error': 'No valid predictions'}

        # Weighted voting
        action, base_confidence = self._weighted_vote(predictions)

        # Meta-labeling for confidence refinement
        meta_features = self._prepare_meta_features(predictions, regime_features)
        meta_confidence = self.meta_labeler.predict_proba(meta_features.reshape(1, -1))[0][1]

        # Final confidence
        final_confidence = (base_confidence + meta_confidence) / 2

        # Regime adjustment
        if current_regime == 2 and action == Actions.BUY:  # High volatility regime
            final_confidence *= 0.8  # Reduce confidence

        metadata = {
            'regime': int(current_regime),
            'base_confidence': float(base_confidence),
            'meta_confidence': float(meta_confidence),
            'models_agreement': self._calculate_agreement(predictions),
            'individual_predictions': [
                {
                    'model': p.model_name,
                    'action': int(p.action),
                    'confidence': float(p.confidence)
                }
                for p in predictions
            ]
        }

        return action, final_confidence, metadata

    def _get_model_prediction(
            self,
            model,
            observation: np.ndarray,
            model_name: str
    ) -> ModelPrediction:
        """Get prediction from individual model"""
        if hasattr(model, 'predict'):
            # RL models (Stable-Baselines3)
            action, _states = model.predict(observation, deterministic=True)

            # Get action probabilities if available
            if hasattr(model.policy, 'get_distribution'):
                dist = model.policy.get_distribution(torch.tensor(observation).unsqueeze(0))
                probs = dist.distribution.probs.detach().numpy()[0]
            else:
                probs = np.zeros(3)
                probs[action] = 1.0

            confidence = probs[action]

        elif isinstance(model, torch.nn.Module):
            # PyTorch models
            with torch.no_grad():
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                output = model(obs_tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]
                action = np.argmax(probs)
                confidence = probs[action]
        else:
            raise ValueError(f"Unknown model type for {model_name}")

        return ModelPrediction(
            action=int(action),
            confidence=float(confidence),
            probabilities=probs,
            model_name=model_name
        )

    def _weighted_vote(self, predictions: List[ModelPrediction]) -> Tuple[int, float]:
        """Aggregate predictions using adaptive weights"""
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        total_weight = 0.0

        for pred in predictions:
            weight = self.adaptive_weights[pred.model_name] * pred.confidence
            action_scores[pred.action] += weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            for action in action_scores:
                action_scores[action] /= total_weight

        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]

        return final_action, final_confidence

    def _prepare_meta_features(
            self,
            predictions: List[ModelPrediction],
            regime_features: np.ndarray
    ) -> np.ndarray:
        """Prepare features for meta-labeler"""
        # Model agreement
        actions = [p.action for p in predictions]
        agreement = len([a for a in actions if a == max(set(actions), key=actions.count)]) / len(actions)

        # Average confidence
        avg_confidence = np.mean([p.confidence for p in predictions])

        # Variance in predictions
        confidence_var = np.var([p.confidence for p in predictions])

        # Combine with regime features
        meta_features = np.concatenate([
            [agreement, avg_confidence, confidence_var],
            regime_features
        ])

        return meta_features

    def _extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        recent_data = market_data.tail(self.lookback_window)

        returns = recent_data['returns'].values
        volatility = recent_data['volatility'].values

        features = np.array([
            np.mean(returns),
            np.std(returns),
            np.mean(volatility),
            np.max(volatility),
            np.percentile(returns, 25),
            np.percentile(returns, 75)
        ])

        return features

    def _calculate_agreement(self, predictions: List[ModelPrediction]) -> float:
        """Calculate agreement among models"""
        actions = [p.action for p in predictions]
        most_common_action = max(set(actions), key=actions.count)
        agreement = sum(1 for a in actions if a == most_common_action) / len(actions)
        return agreement

    def update_performance(self, model_name: str, performance_score: float):
        """Update model performance history and adaptive weights"""
        self.model_performance[model_name].append(performance_score)

        # Keep only recent history
        if len(self.model_performance[model_name]) > self.lookback_window:
            self.model_performance[model_name] = self.model_performance[model_name][-self.lookback_window:]

        # Update adaptive weights (exponential moving average)
        if len(self.model_performance[model_name]) > 10:
            recent_performance = np.mean(self.model_performance[model_name][-10:])
            self.adaptive_weights[model_name] = 0.9 * self.adaptive_weights[model_name] + 0.1 * recent_performance

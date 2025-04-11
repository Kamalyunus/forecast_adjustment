"""
Enhanced Linear Agent Module - Q-learning agent with linear function approximation for
forecast adjustment with support for calendar effects, holidays, and promotions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os
import logging


class ForecastAgent:
    """
    Q-Learning agent with linear function approximation for forecast adjustment.
    This enhanced version supports handling calendar effects, holidays, and promotions.
    """
    
    def __init__(self, 
                feature_dim: int,
                action_size: int = 11,
                learning_rate: float = 0.005,  # Reduced from 0.01
                gamma: float = 0.95,  # Reduced from 0.99 to focus more on immediate rewards
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.05,  # Increased from 0.01 to ensure more exploration
                epsilon_decay: float = 0.998,  # Slower decay from 0.995
                adjustment_factors: Optional[List[float]] = None,
                buffer_size: int = 20000,  # Increased from 10000
                context_learning: bool = True,  # New parameter to enable context-specific learning
                logger: Optional[logging.Logger] = None):
        """
        Initialize Enhanced Linear Function Approximation Agent.
        
        Args:
            feature_dim: Dimension of state features
            action_size: Number of possible adjustment actions
            learning_rate: Learning rate for weight updates
            gamma: Discount factor for future rewards
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            adjustment_factors: Optional list of adjustment factors to use
            buffer_size: Size of experience replay buffer
            context_learning: Whether to use separate weights for different contexts
            logger: Optional logger instance
        """
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.context_learning = context_learning
        
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger("ForecastAgent")
        else:
            self.logger = logger
        
        # Set up adjustment factors with wider range to handle promotions and holidays
        if adjustment_factors is None:
            # Default factors include more extreme values for promotions and holidays
            self.adjustment_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
            # Make sure action_size matches the number of adjustment factors
            self.action_size = len(self.adjustment_factors)
        else:
            self.adjustment_factors = adjustment_factors
            # Ensure action_size is consistent with provided factors
            if action_size != len(adjustment_factors):
                self.logger.warning(f"Action size ({action_size}) doesn't match the length of adjustment_factors ({len(adjustment_factors)})")
                self.action_size = len(adjustment_factors)
        
        # Initialize weights for each action
        # We'll have a separate weight vector for each action
        self.weights = np.zeros((self.action_size, feature_dim))
        
        # Context-specific weights (holiday, promotion, weekend, weekday)
        if self.context_learning:
            self.holiday_weights = np.zeros((self.action_size, feature_dim))
            self.promo_weights = np.zeros((self.action_size, feature_dim))
            self.weekend_weights = np.zeros((self.action_size, feature_dim))
            self.weekday_weights = np.zeros((self.action_size, feature_dim))
        
        # Simple buffer for experience replay
        self.buffer = []
        self.buffer_idx = 0
        
        # Feature normalization values
        self.feature_means = np.zeros(feature_dim)
        self.feature_stds = np.ones(feature_dim)
        self.feature_count = 0
        
        # Action statistics
        self.action_counts = np.zeros(self.action_size)
        
        # Context-specific statistics
        self.holiday_action_counts = np.zeros(self.action_size)
        self.promo_action_counts = np.zeros(self.action_size)
        self.weekend_action_counts = np.zeros(self.action_size)
        self.weekday_action_counts = np.zeros(self.action_size)
        
        # Success tracking (new)
        self.positive_rewards = 0
        self.total_actions = 0
        
        self.logger.info(f"Initialized agent with {feature_dim} features and {self.action_size} actions")
        self.logger.info(f"Using adjustment factors: {self.adjustment_factors}")
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to improve learning stability."""
        # Replace any NaN or Inf values with zeros
        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update feature statistics (during training)
        if self.feature_count < 10000:  # Only update for the first N observations
            self.feature_count += 1
            delta = features - self.feature_means
            self.feature_means += delta / self.feature_count
            
            if self.feature_count > 1:
                # Update running variance
                delta2 = features - self.feature_means
                var_sum = np.sum(delta * delta2, axis=0)
                self.feature_stds = np.sqrt(var_sum / self.feature_count + 1e-8)
        
        # Ensure feature_stds is positive to avoid division by zero
        self.feature_stds = np.maximum(self.feature_stds, 1e-8)
        
        # Normalize features
        normalized = (features - self.feature_means) / self.feature_stds
        
        # Clip to prevent extreme values
        normalized = np.clip(normalized, -10.0, 10.0)
        
        # Final check for NaN or Inf
        if np.isnan(normalized).any() or np.isinf(normalized).any():
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized
    
    def get_q_values(self, features: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate Q-values for all actions given the state features.
        
        Args:
            features: State features
            context: Dictionary with context flags
            
        Returns:
            Array of Q-values for each action
        """
        # Normalize features
        norm_features = self.normalize_features(features)
        
        # Use context-specific weights if available
        if self.context_learning and context is not None:
            if context.get('is_holiday', False):
                q_values = np.dot(self.holiday_weights, norm_features)
            elif context.get('is_promotion', False):
                q_values = np.dot(self.promo_weights, norm_features)
            elif context.get('is_weekend', False):
                q_values = np.dot(self.weekend_weights, norm_features)
            else:
                q_values = np.dot(self.weekday_weights, norm_features)
                
            # Blend with general weights to avoid overfitting
            general_q = np.dot(self.weights, norm_features)
            q_values = 0.7 * q_values + 0.3 * general_q
        else:
            # Calculate Q-values using linear function: Q(s,a) = w_a^T * features
            q_values = np.dot(self.weights, norm_features)
        
        # Check for NaN values
        if np.isnan(q_values).any():
            q_values = np.nan_to_num(q_values, nan=0.0)
            
        return q_values
    
    def act(self, state: np.ndarray, explore: bool = True, context: Optional[Dict] = None) -> int:
        """
        Select an adjustment action based on the current state.
        
        Args:
            state: Current state features
            explore: Whether to use epsilon-greedy exploration
            context: Optional dictionary with context flags (is_holiday, is_promotion, is_weekend)
            
        Returns:
            Selected action index
        """
        q_values = self.get_q_values(state, context)
        
        # Enhanced exploration strategy with pattern-specific guidance
        if explore:
            if np.random.random() < self.epsilon:
                # Biased exploration for specific contexts
                if context is not None:
                    if context.get('is_holiday', False) or context.get('is_promotion', False):
                        # For holidays/promos, try higher factors more often (boost forecast)
                        bias_point = self.action_size * 0.7  # 70% through the adjustment factors
                        p = np.ones(self.action_size)
                        p[int(bias_point):] *= 3  # Increase probability for higher adjustment factors
                        p = p / p.sum()  # Normalize to create valid probability distribution
                        action = np.random.choice(self.action_size, p=p)
                    elif context.get('is_weekend', False):
                        # For weekends, boost higher factors
                        bias_point = self.action_size * 0.6  # 60% through the adjustment factors
                        p = np.ones(self.action_size)
                        p[int(bias_point):] *= 2  # Increase probability for higher adjustment factors
                        p = p / p.sum()
                        action = np.random.choice(self.action_size, p=p)
                    else:
                        # No bias for regular days
                        action = np.random.randint(self.action_size)
                else:
                    action = np.random.randint(self.action_size)
            else:
                action = np.argmax(q_values)
        else:
            action = np.argmax(q_values)
        
        # Track action statistics
        self.action_counts[action] += 1
        self.total_actions += 1
        
        # Track context-specific action statistics if context is provided
        if context is not None:
            if context.get('is_holiday', False):
                self.holiday_action_counts[action] += 1
            if context.get('is_promotion', False):
                self.promo_action_counts[action] += 1
            if context.get('is_weekend', False):
                self.weekend_action_counts[action] += 1
            else:
                self.weekday_action_counts[action] += 1
        
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, 
              context: Optional[Dict] = None) -> float:
        """
        Update weights based on a single experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            context: Context flags for specialized learning
        
        Returns:
            TD error magnitude
        """
        # Check for NaN/Inf values in inputs
        if (np.isnan(state).any() or np.isinf(state).any() or 
            np.isnan(next_state).any() or np.isinf(next_state).any() or 
            np.isnan(reward) or np.isinf(reward)):
            self.logger.warning("NaN or Inf detected in update inputs, skipping update")
            return 0.0
        
        # Track positive rewards for monitoring
        if reward > 0:
            self.positive_rewards += 1
        
        # Store experience in buffer for replay
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done, context))
        else:
            # Replace old experiences using a circular buffer
            self.buffer[self.buffer_idx] = (state, action, reward, next_state, done, context)
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        
        # Normalize states
        norm_state = self.normalize_features(state)
        norm_next_state = self.normalize_features(next_state)
        
        # Check for NaN/Inf after normalization
        if np.isnan(norm_state).any() or np.isinf(norm_state).any():
            self.logger.warning("NaN or Inf detected in normalized state, skipping update")
            return 0.0
        
        # Determine which weight matrix to use based on context
        if self.context_learning and context is not None:
            if context.get('is_holiday', False):
                weights = self.holiday_weights
            elif context.get('is_promotion', False):
                weights = self.promo_weights
            elif context.get('is_weekend', False):
                weights = self.weekend_weights
            else:
                weights = self.weekday_weights
        else:
            weights = self.weights
        
        # Current Q-value
        current_q = np.dot(weights[action], norm_state)
        
        # Next state's best Q-value (using appropriate context weights)
        if done:
            next_q = 0
        else:
            next_q_values = self.get_q_values(next_state, context)
            # Check for NaN/Inf in Q-values
            if np.isnan(next_q_values).any() or np.isinf(next_q_values).any():
                self.logger.warning("NaN or Inf detected in next Q-values, skipping update")
                return 0.0
            next_q = np.max(next_q_values)
        
        # Calculate target and TD error
        target = reward + self.gamma * next_q
        td_error = target - current_q
        
        # Clip TD error to prevent extreme updates
        td_error_clipped = np.clip(td_error, -50.0, 50.0)  # Reduced from -100/100
        
        # Check for NaN/Inf before weight update
        if np.isnan(td_error_clipped) or np.isinf(td_error_clipped):
            self.logger.warning("NaN or Inf TD error detected, skipping update")
            return 0.0
        
        # Update weights with a learning rate that decreases for larger errors
        effective_lr = self.learning_rate / (1.0 + 0.1 * abs(td_error_clipped))
        weights[action] += effective_lr * td_error_clipped * norm_state
        
        # Also update general weights (if using context-specific)
        if self.context_learning and weights is not self.weights:
            self.weights[action] += 0.3 * effective_lr * td_error_clipped * norm_state
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return abs(td_error)
    
    def batch_update(self, batch_size: int = 32) -> float:
        """
        Perform a batch update using experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Average TD error magnitude
        """
        if len(self.buffer) < batch_size:
            return 0.0
        
        # Sample batch of experiences with priority for context samples
        # This ensures we learn more from holiday, promotion, and weekend samples
        context_indices = []
        regular_indices = []
        
        for i, (_, _, _, _, _, context) in enumerate(self.buffer):
            if context is not None and (context.get('is_holiday', False) or 
                                       context.get('is_promotion', False) or
                                       context.get('is_weekend', False)):
                context_indices.append(i)
            else:
                regular_indices.append(i)
        
        # Ensure at least some context samples if available
        context_count = min(batch_size // 2, len(context_indices))
        regular_count = batch_size - context_count
        
        # Sample from each group
        if context_count > 0:
            context_batch = np.random.choice(context_indices, context_count, replace=False)
        else:
            context_batch = []
            
        regular_batch = np.random.choice(regular_indices, regular_count, replace=False)
        
        # Combine indices
        indices = np.concatenate([context_batch, regular_batch])
        batch = [self.buffer[int(i)] for i in indices]
        
        total_error = 0.0
        valid_samples = 0
        
        for state, action, reward, next_state, done, context in batch:
            # Skip samples with NaN values
            if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(reward):
                continue
                
            # Update using the sample
            td_error = self.update(state, action, reward, next_state, done, context)
            
            total_error += td_error
            valid_samples += 1
        
        # Return average TD error over valid samples
        if valid_samples > 0:
            return total_error / valid_samples
        else:
            return 0.0
    
    def calculate_adjusted_forecast(self, action_idx: int, forecast: float, 
                                   context: Optional[Dict] = None) -> float:
        """
        Apply adjustment to forecast based on the selected action and context.
        
        Args:
            action_idx: Action index
            forecast: Original forecast value
            context: Optional context information
            
        Returns:
            Adjusted forecast
        """
        # Ensure action index is valid
        action_idx = max(0, min(action_idx, len(self.adjustment_factors) - 1))
        
        # Check for NaN forecast
        if np.isnan(forecast):
            return 0.0
            
        # Apply factor to forecast
        factor = self.adjustment_factors[action_idx]
        
        # Apply context-specific adjustments
        if context is not None:
            # Apply stronger adjustments for holidays and promotions
            if context.get('is_holiday', False) and factor > 1.0:
                # Boost holiday upward adjustments
                factor = factor * 1.1
            elif context.get('is_promotion', False) and factor > 1.0:
                # Boost promotion upward adjustments
                factor = factor * 1.2
            elif context.get('is_weekend', False) and factor > 1.0:
                # Boost weekend upward adjustments slightly
                factor = factor * 1.05
        
        adjusted_forecast = forecast * factor
        
        return max(0, adjusted_forecast)  # Ensure non-negative
    
    def get_action_statistics(self) -> Dict:
        """
        Get statistics on action selection in different contexts.
        
        Returns:
            Dictionary of action statistics
        """
        # Convert counts to distributions
        total_actions = np.sum(self.action_counts)
        total_holiday = np.sum(self.holiday_action_counts)
        total_promo = np.sum(self.promo_action_counts)
        total_weekend = np.sum(self.weekend_action_counts)
        total_weekday = np.sum(self.weekday_action_counts)
        
        # Calculate distributions (handle zero counts)
        all_dist = self.action_counts / total_actions if total_actions > 0 else np.zeros_like(self.action_counts)
        holiday_dist = self.holiday_action_counts / total_holiday if total_holiday > 0 else np.zeros_like(self.holiday_action_counts)
        promo_dist = self.promo_action_counts / total_promo if total_promo > 0 else np.zeros_like(self.promo_action_counts)
        weekend_dist = self.weekend_action_counts / total_weekend if total_weekend > 0 else np.zeros_like(self.weekend_action_counts)
        weekday_dist = self.weekday_action_counts / total_weekday if total_weekday > 0 else np.zeros_like(self.weekday_action_counts)
        
        # Calculate success rate
        success_rate = self.positive_rewards / max(1, self.total_actions)
        
        return {
            'overall': {
                'counts': self.action_counts.tolist(),
                'distribution': all_dist.tolist(),
                'factors': self.adjustment_factors,
                'success_rate': success_rate
            },
            'holiday': {
                'counts': self.holiday_action_counts.tolist(),
                'distribution': holiday_dist.tolist()
            },
            'promotion': {
                'counts': self.promo_action_counts.tolist(),
                'distribution': promo_dist.tolist()
            },
            'weekend': {
                'counts': self.weekend_action_counts.tolist(),
                'distribution': weekend_dist.tolist()
            },
            'weekday': {
                'counts': self.weekday_action_counts.tolist(),
                'distribution': weekday_dist.tolist()
            }
        }
    
    def save(self, filepath: str) -> None:
        """
        Save agent to file.
        
        Args:
            filepath: Path to save the agent
        """
        data = {
            'weights': self.weights,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'feature_count': self.feature_count,
            'feature_dim': self.feature_dim,
            'action_size': self.action_size,
            'action_counts': self.action_counts,
            'holiday_action_counts': self.holiday_action_counts,
            'promo_action_counts': self.promo_action_counts,
            'weekend_action_counts': self.weekend_action_counts,
            'weekday_action_counts': self.weekday_action_counts,
            'adjustment_factors': self.adjustment_factors,
            'params': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'context_learning': self.context_learning
            },
            'positive_rewards': self.positive_rewards,
            'total_actions': self.total_actions
        }
        
        # Save context-specific weights if using them
        if self.context_learning:
            data['holiday_weights'] = self.holiday_weights
            data['promo_weights'] = self.promo_weights
            data['weekend_weights'] = self.weekend_weights
            data['weekday_weights'] = self.weekday_weights
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, logger: Optional[logging.Logger] = None) -> 'ForecastAgent':
        """
        Load agent from file.
        
        Args:
            filepath: Path to load the agent from
            logger: Optional logger instance
            
        Returns:
            Loaded agent
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Get context learning parameter or default to False for backward compatibility
        context_learning = data['params'].get('context_learning', False)
        
        # Create new agent with loaded adjustment factors
        agent = cls(
            feature_dim=data['feature_dim'],
            action_size=data['action_size'],
            learning_rate=data['params']['learning_rate'],
            gamma=data['params']['gamma'],
            epsilon_start=data['params']['epsilon'],
            epsilon_end=data['params']['epsilon_end'],
            epsilon_decay=data['params']['epsilon_decay'],
            adjustment_factors=data.get('adjustment_factors'),
            context_learning=context_learning,
            logger=logger
        )
        
        # Load saved state
        agent.weights = data['weights']
        agent.feature_means = data['feature_means']
        agent.feature_stds = data['feature_stds']
        agent.feature_count = data['feature_count']
        agent.action_counts = data['action_counts']
        
        # Load context-specific action counts if available
        if 'holiday_action_counts' in data:
            agent.holiday_action_counts = data['holiday_action_counts']
        if 'promo_action_counts' in data:
            agent.promo_action_counts = data['promo_action_counts']
        if 'weekend_action_counts' in data:
            agent.weekend_action_counts = data['weekend_action_counts']
        if 'weekday_action_counts' in data:
            agent.weekday_action_counts = data['weekday_action_counts']
        
        # Load context-specific weights if available and context learning is enabled
        if context_learning:
            if 'holiday_weights' in data:
                agent.holiday_weights = data['holiday_weights']
            if 'promo_weights' in data:
                agent.promo_weights = data['promo_weights']
            if 'weekend_weights' in data:
                agent.weekend_weights = data['weekend_weights']
            if 'weekday_weights' in data:
                agent.weekday_weights = data['weekday_weights']
        
        # Load success metrics if available
        if 'positive_rewards' in data:
            agent.positive_rewards = data['positive_rewards']
        if 'total_actions' in data:
            agent.total_actions = data['total_actions']
        
        if logger:
            logger.info(f"Agent loaded from {filepath}")
        
        return agent
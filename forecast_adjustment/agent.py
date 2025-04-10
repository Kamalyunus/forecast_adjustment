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
                learning_rate: float = 0.01,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                adjustment_factors: Optional[List[float]] = None,
                buffer_size: int = 10000,
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
        
        self.logger.info(f"Initialized agent with {feature_dim} features and {self.action_size} actions")
        self.logger.info(f"Using adjustment factors: {self.adjustment_factors}")
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to improve learning stability.
        Uses running mean and standard deviation.
        
        Args:
            features: Raw state features
            
        Returns:
            Normalized features
        """
        # Check for NaN values and replace with zeros
        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)
            
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
        
        # Normalize features
        normalized = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Additional check for NaN after normalization
        if np.isnan(normalized).any():
            normalized = np.nan_to_num(normalized, nan=0.0)
            
        return normalized
    
    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate Q-values for all actions given the state features.
        
        Args:
            features: State features
            
        Returns:
            Array of Q-values for each action
        """
        # Normalize features
        norm_features = self.normalize_features(features)
        
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
        q_values = self.get_q_values(state)
        
        # Epsilon-greedy policy
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(q_values)
        
        # Track action statistics
        self.action_counts[action] += 1
        
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
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> float:
        """
        Update the weights based on a single experience.
        
        Args:
            state: Current state features
            action: Action taken
            reward: Reward received
            next_state: Next state features
            done: Whether the episode is done
            
        Returns:
            TD error magnitude
        """
        # Check for NaN values
        if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(reward):
            # Return zero TD error if there are NaN values
            return 0.0
            
        # Store experience in buffer
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.buffer_idx] = (state, action, reward, next_state, done)
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        
        # Normalize states
        norm_state = self.normalize_features(state)
        norm_next_state = self.normalize_features(next_state)
        
        # Current Q-value: Q(s,a) = w_a^T * state
        current_q = np.dot(self.weights[action], norm_state)
        
        # Next state's best Q-value
        if done:
            next_q = 0
        else:
            next_q_values = np.dot(self.weights, norm_next_state)
            next_q = np.max(next_q_values)
        
        # Calculate target using standard Q-learning update
        target = reward + self.gamma * next_q
        
        # Handle potential NaN values
        if np.isnan(current_q) or np.isnan(target):
            return 0.0
            
        # TD error
        td_error = target - current_q
        
        # Update weights for the selected action
        self.weights[action] += self.learning_rate * td_error * norm_state
        
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
        
        # Sample batch of experiences
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        total_error = 0.0
        valid_samples = 0
        
        for state, action, reward, next_state, done in batch:
            # Skip samples with NaN values
            if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(reward):
                continue
                
            # Normalize states
            norm_state = self.normalize_features(state)
            norm_next_state = self.normalize_features(next_state)
            
            # Current Q-value
            current_q = np.dot(self.weights[action], norm_state)
            
            # Next state's best Q-value
            if done:
                next_q = 0
            else:
                next_q_values = np.dot(self.weights, norm_next_state)
                next_q = np.max(next_q_values)
            
            # Calculate target
            target = reward + self.gamma * next_q
            
            # Skip if target or current_q is NaN
            if np.isnan(current_q) or np.isnan(target):
                continue
                
            # TD error
            td_error = target - current_q
            
            # Update weights
            self.weights[action] += self.learning_rate * td_error * norm_state
            
            total_error += abs(td_error)
            valid_samples += 1
        
        # Return average TD error over valid samples
        if valid_samples > 0:
            return total_error / valid_samples
        else:
            return 0.0
    
    def calculate_adjusted_forecast(self, action_idx: int, forecast: float) -> float:
        """
        Apply adjustment to forecast based on the selected action.
        
        Args:
            action_idx: Action index
            forecast: Original forecast value
            
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
        
        return {
            'overall': {
                'counts': self.action_counts.tolist(),
                'distribution': all_dist.tolist(),
                'factors': self.adjustment_factors
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
                'epsilon_decay': self.epsilon_decay
            }
        }
        
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
        
        if logger:
            logger.info(f"Agent loaded from {filepath}")
        
        return agent
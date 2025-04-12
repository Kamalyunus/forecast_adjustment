"""
Q-learning agent with linear function approximation for forecast adjustment with 
support for context-specific learning, SKU banding, UCB exploration, and "First, Do No Harm" safeguards.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import os
import logging


class ForecastAgent:
    """
    Q-Learning agent with linear function approximation for forecast adjustment.
    This agent learns to select the best adjustment factor based on the forecast context.
    
    Features:
    - Context-specific learning (holidays, promotions, weekends)
    - Support for SKU banding (A-E bands based on sales volume)
    - Upper Confidence Bound (UCB) exploration for more efficient learning
    - "First, Do No Harm" mechanism to prevent harmful adjustments
    - Experience replay for improved sample efficiency
    - Linear function approximation for generalization
    """
    
    def __init__(self, 
                feature_dim: int,
                action_size: int = 11,
                learning_rate: float = 0.005,
                gamma: float = 0.95,
                ucb_constant: float = 2.0,  # UCB exploration parameter
                conservative_factor: float = 0.7,  # How conservative to be with adjustments (0-1)
                adjustment_factors: Optional[List[float]] = None,
                buffer_size: int = 20000,
                context_learning: bool = True,
                logger: Optional[logging.Logger] = None):
        """
        Initialize Q-learning agent with linear function approximation.
        
        Args:
            feature_dim: Dimension of state features
            action_size: Number of possible adjustment actions
            learning_rate: Learning rate for weight updates
            gamma: Discount factor for future rewards
            ucb_constant: Exploration parameter for UCB algorithm
            conservative_factor: How conservative to be with adjustment risk assessment
            adjustment_factors: Optional list of adjustment factors to use
            buffer_size: Size of experience replay buffer
            context_learning: Whether to use separate weights for different contexts
            logger: Optional logger instance
        """
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.conservative_factor = conservative_factor
        self.buffer_size = buffer_size
        self.context_learning = context_learning
        
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger("ForecastAgent")
        else:
            self.logger = logger
        
        # Set up adjustment factors
        if adjustment_factors is None:
            # Default factors with wider range for promotions/holidays
            self.adjustment_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
            self.action_size = len(self.adjustment_factors)
        else:
            self.adjustment_factors = adjustment_factors
            if action_size != len(adjustment_factors):
                self.logger.warning(f"Action size ({action_size}) doesn't match adjustment_factors length ({len(adjustment_factors)})")
                self.action_size = len(adjustment_factors)
        
        # Initialize weights for each action
        self.weights = np.zeros((self.action_size, feature_dim))
        
        # Context-specific weights
        if self.context_learning:
            self.holiday_weights = np.zeros((self.action_size, feature_dim))
            self.promo_weights = np.zeros((self.action_size, feature_dim))
            self.weekend_weights = np.zeros((self.action_size, feature_dim))
            self.weekday_weights = np.zeros((self.action_size, feature_dim))
            
            # Band-specific weights (A-E)
            self.band_weights = {
                'A': np.zeros((self.action_size, feature_dim)),
                'B': np.zeros((self.action_size, feature_dim)),
                'C': np.zeros((self.action_size, feature_dim)),
                'D': np.zeros((self.action_size, feature_dim)),
                'E': np.zeros((self.action_size, feature_dim)),
            }
        
        # Simple buffer for experience replay
        self.buffer = []
        self.buffer_idx = 0
        
        # Feature normalization values
        self.feature_means = np.zeros(feature_dim)
        self.feature_stds = np.ones(feature_dim)
        self.feature_count = 0
        
        # UCB tracking variables
        self.action_counts = np.ones((self.action_size,))  # Start at 1 to avoid division by zero
        self.action_value_sums = np.zeros((self.action_size,))
        
        # Context-specific UCB tracking
        self.context_action_counts = {
            'holiday': np.ones((self.action_size,)),
            'promotion': np.ones((self.action_size,)),
            'weekend': np.ones((self.action_size,)),
            'weekday': np.ones((self.action_size,))
        }
        self.context_value_sums = {
            'holiday': np.zeros((self.action_size,)),
            'promotion': np.zeros((self.action_size,)),
            'weekend': np.zeros((self.action_size,)),
            'weekday': np.zeros((self.action_size,))
        }
        
        # Band-specific UCB tracking
        self.band_action_counts = {
            band: np.ones((self.action_size,)) for band in ['A', 'B', 'C', 'D', 'E']
        }
        self.band_value_sums = {
            band: np.zeros((self.action_size,)) for band in ['A', 'B', 'C', 'D', 'E']
        }
        
        # Track adjustment success rate for each action
        self.successful_adjustments = np.zeros(self.action_size)
        self.total_adjustments = np.ones(self.action_size)  # Start at 1 to avoid division by zero
        
        # Success rate tracking for contexts
        self.context_successful_adjustments = {
            'holiday': np.zeros(self.action_size),
            'promotion': np.zeros(self.action_size),
            'weekend': np.zeros(self.action_size),
            'weekday': np.zeros(self.action_size)
        }
        self.context_total_adjustments = {
            'holiday': np.ones(self.action_size),
            'promotion': np.ones(self.action_size),
            'weekend': np.ones(self.action_size),
            'weekday': np.ones(self.action_size)
        }
        
        # Success rate tracking for bands
        self.band_successful_adjustments = {
            band: np.zeros(self.action_size) for band in ['A', 'B', 'C', 'D', 'E']
        }
        self.band_total_adjustments = {
            band: np.ones(self.action_size) for band in ['A', 'B', 'C', 'D', 'E']
        }
        
        # Track performance history
        self.positive_rewards = 0
        self.total_actions = 0
        self.mape_improvements = []
        self.bias_improvements = []
        
        self.logger.info(f"Initialized agent with {feature_dim} features and {self.action_size} actions")
        self.logger.info(f"Using adjustment factors: {self.adjustment_factors}")
        self.logger.info(f"UCB exploration with constant: {self.ucb_constant}")
        self.logger.info(f"Conservative factor for 'First, Do No Harm': {self.conservative_factor}")
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to improve learning stability.
        
        Args:
            features: Raw state features
            
        Returns:
            Normalized features
        """
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
            context: Dictionary with context flags (is_holiday, is_promotion, is_weekend, sku_band)
            
        Returns:
            Array of Q-values for each action
        """
        # Normalize features
        norm_features = self.normalize_features(features)
        
        # Use context-specific weights if available
        if self.context_learning and context is not None:
            # First, handle standard contexts
            if context.get('is_holiday', False):
                q_values = np.dot(self.holiday_weights, norm_features)
            elif context.get('is_promotion', False):
                q_values = np.dot(self.promo_weights, norm_features)
            elif context.get('is_weekend', False):
                q_values = np.dot(self.weekend_weights, norm_features)
            else:
                q_values = np.dot(self.weekday_weights, norm_features)
            
            # If SKU band is available, blend with band-specific weights
            if 'sku_band' in context:
                band = context['sku_band']
                if band in self.band_weights:
                    band_q = np.dot(self.band_weights[band], norm_features)
                    # Blend with more weight to band-specific values
                    q_values = 0.3 * q_values + 0.7 * band_q
                
            # Blend with general weights to avoid overfitting
            general_q = np.dot(self.weights, norm_features)
            q_values = 0.7 * q_values + 0.3 * general_q
        else:
            # Use general weights
            q_values = np.dot(self.weights, norm_features)
        
        # Check for NaN values
        if np.isnan(q_values).any():
            q_values = np.nan_to_num(q_values, nan=0.0)
            
        return q_values
    
    def act(self, state: np.ndarray, explore: bool = True, context: Optional[Dict] = None) -> int:
        """
        Select an adjustment action using UCB exploration strategy.
        
        Args:
            state: Current state features
            explore: Whether to use UCB exploration
            context: Optional dictionary with context flags
            
        Returns:
            Selected action index
        """
        # Get Q-values from function approximation
        q_values = self.get_q_values(state, context)
        
        if explore:
            # Determine which action count and value tracking to use based on context
            if self.context_learning and context is not None:
                if context.get('is_holiday', False):
                    action_counts = self.context_action_counts['holiday']
                    value_sums = self.context_value_sums['holiday']
                    success_rate = self.context_successful_adjustments['holiday'] / self.context_total_adjustments['holiday']
                elif context.get('is_promotion', False):
                    action_counts = self.context_action_counts['promotion']
                    value_sums = self.context_value_sums['promotion']
                    success_rate = self.context_successful_adjustments['promotion'] / self.context_total_adjustments['promotion']
                elif context.get('is_weekend', False):
                    action_counts = self.context_action_counts['weekend']
                    value_sums = self.context_value_sums['weekend']
                    success_rate = self.context_successful_adjustments['weekend'] / self.context_total_adjustments['weekend']
                elif 'sku_band' in context and context['sku_band'] in self.band_action_counts:
                    action_counts = self.band_action_counts[context['sku_band']]
                    value_sums = self.band_value_sums[context['sku_band']]
                    success_rate = self.band_successful_adjustments[context['sku_band']] / self.band_total_adjustments[context['sku_band']]
                else:
                    action_counts = self.context_action_counts['weekday']
                    value_sums = self.context_value_sums['weekday']
                    success_rate = self.context_successful_adjustments['weekday'] / self.context_total_adjustments['weekday']
            else:
                action_counts = self.action_counts
                value_sums = self.action_value_sums
                success_rate = self.successful_adjustments / self.total_adjustments
            
            # Calculate UCB values
            total_count = np.sum(action_counts)
            average_values = value_sums / action_counts
            
            # Combine UCB with success rate for safer exploration
            # Higher weight on actions that have been more successful
            exploration_term = self.ucb_constant * np.sqrt(np.log(total_count) / action_counts)
            
            # Adjust exploration term based on success rate
            # Reduce exploration for actions with poor success history
            adjusted_exploration = exploration_term * (0.5 + 0.5 * success_rate)
            
            # Give a bonus to the no-adjustment action (1.0x factor) for safety
            no_adjustment_idx = self.adjustment_factors.index(1.0) if 1.0 in self.adjustment_factors else -1
            if no_adjustment_idx >= 0:
                safety_bonus = (1.0 - success_rate) * 0.5  # More bonus when success is low
                adjusted_exploration[no_adjustment_idx] += np.mean(safety_bonus)
            
            # Calculate final UCB values
            ucb_values = average_values + adjusted_exploration
            
            # Bias towards safer adjustments early in training
            if self.total_actions < 1000:
                # Early in training, bias more towards no or small adjustments
                for i, factor in enumerate(self.adjustment_factors):
                    # Penalize adjustments far from 1.0
                    distance_from_neutral = abs(factor - 1.0)
                    penalty = distance_from_neutral * (1.0 - min(self.total_actions / 1000, 1.0))
                    ucb_values[i] -= penalty
            
            # Select action with highest UCB value
            action = np.argmax(ucb_values)
        else:
            # Without exploration, just take the best action according to learned Q-values
            action = np.argmax(q_values)
        
        # Track action selection
        self.total_actions += 1
        
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, 
              done: bool, context: Optional[Dict] = None, mape_improvement: Optional[float] = None) -> float:
        """
        Update weights based on a single experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            context: Context flags for specialized learning
            mape_improvement: Optional MAPE improvement from this action
        
        Returns:
            TD error magnitude
        """
        # Check for NaN/Inf values in inputs
        if (np.isnan(state).any() or np.isinf(state).any() or 
            np.isnan(next_state).any() or np.isinf(next_state).any() or 
            np.isnan(reward) or np.isinf(reward)):
            self.logger.warning("NaN or Inf detected in update inputs, skipping update")
            return 0.0
        
        # Update UCB tracking for this action
        self.action_counts[action] += 1
        self.action_value_sums[action] += reward
        
        # Update context-specific UCB tracking
        if self.context_learning and context is not None:
            if context.get('is_holiday', False):
                self.context_action_counts['holiday'][action] += 1
                self.context_value_sums['holiday'][action] += reward
            elif context.get('is_promotion', False):
                self.context_action_counts['promotion'][action] += 1
                self.context_value_sums['promotion'][action] += reward
            elif context.get('is_weekend', False):
                self.context_action_counts['weekend'][action] += 1
                self.context_value_sums['weekend'][action] += reward
            else:
                self.context_action_counts['weekday'][action] += 1
                self.context_value_sums['weekday'][action] += reward
                
            # Update band-specific UCB tracking
            if 'sku_band' in context and context['sku_band'] in self.band_action_counts:
                band = context['sku_band']
                self.band_action_counts[band][action] += 1
                self.band_value_sums[band][action] += reward
        
        # Track if this action was successful (led to MAPE improvement)
        if mape_improvement is not None:
            successful = mape_improvement > 0
            self.total_adjustments[action] += 1
            if successful:
                self.successful_adjustments[action] += 1
            
            # Update context-specific success rate
            if self.context_learning and context is not None:
                if context.get('is_holiday', False):
                    self.context_total_adjustments['holiday'][action] += 1
                    if successful:
                        self.context_successful_adjustments['holiday'][action] += 1
                elif context.get('is_promotion', False):
                    self.context_total_adjustments['promotion'][action] += 1
                    if successful:
                        self.context_successful_adjustments['promotion'][action] += 1
                elif context.get('is_weekend', False):
                    self.context_total_adjustments['weekend'][action] += 1
                    if successful:
                        self.context_successful_adjustments['weekend'][action] += 1
                else:
                    self.context_total_adjustments['weekday'][action] += 1
                    if successful:
                        self.context_successful_adjustments['weekday'][action] += 1
                
                # Update band-specific success rate
                if 'sku_band' in context and context['sku_band'] in self.band_total_adjustments:
                    band = context['sku_band']
                    self.band_total_adjustments[band][action] += 1
                    if successful:
                        self.band_successful_adjustments[band][action] += 1
            
            # Store improvement metrics
            self.mape_improvements.append(mape_improvement)
        
        # Track positive rewards for monitoring
        if reward > 0:
            self.positive_rewards += 1
        
        # Store experience in buffer for replay
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done, context, mape_improvement))
        else:
            # Replace old experiences using a circular buffer
            self.buffer[self.buffer_idx] = (state, action, reward, next_state, done, context, mape_improvement)
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
            # First check for band-specific weights (highest priority)
            if 'sku_band' in context:
                band = context['sku_band']
                if band in self.band_weights:
                    weights = self.band_weights[band]
                    # Also update general weights for this band with a smaller learning rate
                    self._update_weights(self.weights, action, norm_state, reward, next_state, done, context, 0.3)
                else:
                    weights = self.weights
            # Then check for other context-specific weights
            elif context.get('is_holiday', False):
                weights = self.holiday_weights
            elif context.get('is_promotion', False):
                weights = self.promo_weights
            elif context.get('is_weekend', False):
                weights = self.weekend_weights
            else:
                weights = self.weekday_weights
        else:
            weights = self.weights
        
        # Perform the update on the selected weight matrix
        td_error = self._update_weights(weights, action, norm_state, reward, next_state, done, context)
        
        # If using context-specific weights but not band weights, update general weights with a smaller learning rate
        if self.context_learning and weights is not self.weights and 'sku_band' not in context:
            self._update_weights(self.weights, action, norm_state, reward, next_state, done, context, 0.3)
        
        return abs(td_error)
    
    def _update_weights(self, weights: np.ndarray, action: int, norm_state: np.ndarray, 
                      reward: float, next_state: np.ndarray, done: bool, 
                      context: Optional[Dict] = None, learning_rate_factor: float = 1.0) -> float:
        """
        Helper method to update a specific weight matrix.
        
        Args:
            weights: Weight matrix to update
            action: Action taken
            norm_state: Normalized state features
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            context: Context flags
            learning_rate_factor: Factor to adjust learning rate (for secondary updates)
            
        Returns:
            TD error
        """
        # Current Q-value
        current_q = np.dot(weights[action], norm_state)
        
        # Next state's best Q-value
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
        td_error_clipped = np.clip(td_error, -50.0, 50.0)
        
        # Check for NaN/Inf before weight update
        if np.isnan(td_error_clipped) or np.isinf(td_error_clipped):
            self.logger.warning("NaN or Inf TD error detected, skipping update")
            return 0.0
        
        # Apply learning rate factor and update weights
        effective_lr = (self.learning_rate * learning_rate_factor) / (1.0 + 0.1 * abs(td_error_clipped))
        weights[action] += effective_lr * td_error_clipped * norm_state
        
        return td_error
    
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
        
        # Sample batch of experiences with priority for band and context samples
        band_indices = []
        context_indices = []
        successful_indices = []
        regular_indices = []
        
        for i, (_, _, _, _, _, context, mape_improvement) in enumerate(self.buffer):
            # Prioritize samples with MAPE improvement
            if mape_improvement is not None and mape_improvement > 0:
                successful_indices.append(i)
                
            if context is not None and 'sku_band' in context:
                band_indices.append(i)
            elif context is not None and (context.get('is_holiday', False) or 
                                       context.get('is_promotion', False) or
                                       context.get('is_weekend', False)):
                context_indices.append(i)
            else:
                regular_indices.append(i)
        
        # Prioritize sampling successful experiences
        successful_count = min(batch_size // 2, len(successful_indices))
        
        # Sample from band experiences
        band_count = min(batch_size // 4, len(band_indices))
        
        # Sample from context experiences
        context_count = min(batch_size // 4, len(context_indices))
        
        # Regular samples for the remainder
        regular_count = batch_size - successful_count - band_count - context_count
        
        # Sample from each group
        sampled_indices = []
        
        if successful_count > 0 and len(successful_indices) > 0:
            sampled_indices.extend(np.random.choice(successful_indices, successful_count, replace=False))
        
        if band_count > 0 and len(band_indices) > 0:
            remaining_band_indices = [i for i in band_indices if i not in sampled_indices]
            if remaining_band_indices:
                sampled_indices.extend(np.random.choice(remaining_band_indices, min(band_count, len(remaining_band_indices)), replace=False))
    
        # Sample from context indices if available    
        if context_count > 0 and len(context_indices) > 0:
            remaining_context_indices = [i for i in context_indices if i not in sampled_indices]
            if remaining_context_indices:
                sampled_indices.extend(np.random.choice(remaining_context_indices, min(context_count, len(remaining_context_indices)), replace=False))
        
        # Sample from regular indices if available
        if regular_count > 0 and len(regular_indices) > 0:
            remaining_regular_indices = [i for i in regular_indices if i not in sampled_indices]
            if remaining_regular_indices:
                sampled_indices.extend(np.random.choice(remaining_regular_indices, min(regular_count, len(remaining_regular_indices)), replace=False))
        
        # If we still need more samples, sample from all indices
        remaining_count = batch_size - len(sampled_indices)
        if remaining_count > 0:
            all_indices = list(range(len(self.buffer)))
            remaining_indices = [i for i in all_indices if i not in sampled_indices]
            if remaining_indices:
                additional_samples = np.random.choice(
                    remaining_indices, 
                    min(remaining_count, len(remaining_indices)), 
                    replace=False
                )
                sampled_indices.extend(additional_samples)

        # Get batch
        batch = [self.buffer[int(i)] for i in sampled_indices]
        
        total_error = 0.0
        valid_samples = 0
        
        for state, action, reward, next_state, done, context, mape_improvement in batch:
            # Skip samples with NaN values
            if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(reward):
                continue
                
            # Update using the sample
            td_error = self.update(state, action, reward, next_state, done, context, mape_improvement)
            
            total_error += td_error
            valid_samples += 1
        
        # Return average TD error over valid samples
        if valid_samples > 0:
            return total_error / valid_samples
        else:
            return 0.0
    
    def _estimate_adjustment_risk(self, 
                                original_forecast: float, 
                                adjusted_forecast: float, 
                                factor: float, 
                                historical_mape: float, 
                                historical_bias: float,
                                context: Optional[Dict]) -> float:
        """
        Estimate the risk of a forecast adjustment making MAPE worse.
        
        Args:
            original_forecast: Original forecast value
            adjusted_forecast: Adjusted forecast value
            factor: Adjustment factor
            historical_mape: Historical MAPE for this forecast scenario
            historical_bias: Historical bias for this forecast scenario
            context: Optional context dictionary
            
        Returns:
            Risk score between 0 (safe) and 1 (high risk)
        """
        # Base risk assessment on adjustment magnitude
        max_adjustment = max(abs(f - 1.0) for f in self.adjustment_factors)
        base_risk = abs(factor - 1.0) / max_adjustment
        
        # Scale by conservative factor (higher = more risk averse)
        base_risk *= self.conservative_factor
        
        # Check if adjustment direction matches historical bias direction
        if historical_bias != 0:
            # If adjusting in the same direction as the bias, it's riskier
            # We want to adjust in the opposite direction of the bias
            if (factor > 1.0 and historical_bias > 0) or (factor < 1.0 and historical_bias < 0):
                # Adjustment is in wrong direction based on historical bias
                base_risk *= 1.5
            else:
                # Adjustment is in right direction
                base_risk *= 0.7
        
        # Consider historical MAPE - higher historical error means more risk
        if historical_mape > 0:
            if historical_mape > 0.5:  # Very high historical error
                base_risk *= 1.3
            elif historical_mape > 0.3:  # Moderate historical error
                base_risk *= 1.1
            elif historical_mape < 0.1:  # Low historical error
                base_risk *= 0.8
        
        # Consider context factors
        if context:
            # Special events have different risk profiles
            if context.get('is_holiday', False) or context.get('is_promotion', False):
                if factor > 1.0:
                    # Upward adjustments for special events are generally safer
                    base_risk *= 0.8
                else:
                    # Downward adjustments for special events are riskier
                    base_risk *= 1.2
            
            # Consider SKU band
            if 'sku_band' in context:
                if context['sku_band'] in ['A', 'B']:  # High-volume SKUs
                    base_risk *= 0.8  # Less risk due to more data/stability
                elif context['sku_band'] in ['D', 'E']:  # Low-volume SKUs
                    base_risk *= 1.2  # More risk due to less data/higher volatility
            
            # Consider pattern type if available
            if 'pattern_type' in context:
                pattern = context.get('pattern_type')
                if pattern == 'underbias' and factor > 1.0:
                    # Upward adjustments for underbias patterns are safer
                    base_risk *= 0.7
                elif pattern == 'promo_holiday' and context.get('is_holiday', False) and factor > 1.0:
                    # Upward adjustments for promo_holiday patterns during holidays are safer
                    base_risk *= 0.7
                elif pattern == 'day_pattern' and context.get('is_weekend', False) and factor > 1.0:
                    # Upward adjustments for day_pattern on weekends are safer
                    base_risk *= 0.8
        
        # Consider success history for this action
        action_idx = self.adjustment_factors.index(factor) if factor in self.adjustment_factors else -1
        if action_idx >= 0:
            success_rate = self.successful_adjustments[action_idx] / self.total_adjustments[action_idx]
            
            # Lower risk for actions with high success rate
            if success_rate > 0.7:  # High success rate
                base_risk *= 0.7
            elif success_rate < 0.3:  # Low success rate
                base_risk *= 1.3
            
            # Consider context-specific success rates
            if context:
                if context.get('is_holiday', False) and self.context_total_adjustments['holiday'][action_idx] > 5:
                    holiday_success = self.context_successful_adjustments['holiday'][action_idx] / self.context_total_adjustments['holiday'][action_idx]
                    if holiday_success > 0.6:
                        base_risk *= 0.8
                    elif holiday_success < 0.3:
                        base_risk *= 1.2
                
                elif context.get('is_promotion', False) and self.context_total_adjustments['promotion'][action_idx] > 5:
                    promo_success = self.context_successful_adjustments['promotion'][action_idx] / self.context_total_adjustments['promotion'][action_idx]
                    if promo_success > 0.6:
                        base_risk *= 0.8
                    elif promo_success < 0.3:
                        base_risk *= 1.2
                
                elif 'sku_band' in context and context['sku_band'] in self.band_total_adjustments:
                    band = context['sku_band']
                    if self.band_total_adjustments[band][action_idx] > 5:
                        band_success = self.band_successful_adjustments[band][action_idx] / self.band_total_adjustments[band][action_idx]
                        if band_success > 0.6:
                            base_risk *= 0.8
                        elif band_success < 0.3:
                            base_risk *= 1.2
        
        # Cap risk between 0 and 1
        return min(1.0, max(0.0, base_risk))
        
    def calculate_adjusted_forecast(self, 
                              action_idx: int, 
                              forecast: float, 
                              context: Optional[Dict] = None,
                              historical_mape: Optional[float] = None,
                              historical_bias: Optional[float] = None) -> float:
        """
        Apply adjustment to forecast with "First, Do No Harm" safety mechanism.
        
        Args:
            action_idx: Action index selected
            forecast: Original forecast value to adjust
            context: Optional context dictionary
            historical_mape: Historical MAPE for this forecast
            historical_bias: Historical bias for this forecast
            
        Returns:
            Adjusted forecast value
        """
        # Ensure action index is valid
        action_idx = max(0, min(action_idx, len(self.adjustment_factors) - 1))
        
        # Check for NaN forecast
        if np.isnan(forecast):
            return 0.0
            
        # Get adjustment factor
        factor = self.adjustment_factors[action_idx]
        
        # Apply band-specific adjustment constraints (these still apply but may be overridden by safety mechanism)
        factor = self._apply_band_constraints(factor, context)
        
        # Calculate the potential adjusted forecast
        potential_adjusted_forecast = forecast * factor
        
        # Initialize risk_score (to avoid UnboundLocalError)
        risk_score = 0.0
        
        # Apply "First, Do No Harm" safety mechanism when historical metrics are available
        if historical_mape is not None or historical_bias is not None:
            # Default values if not provided
            if historical_mape is None:
                historical_mape = 0.2  # Default assumption
            if historical_bias is None:
                historical_bias = 0.0  # Default assumption
                
            # Assess risk of this adjustment
            risk_score = self._estimate_adjustment_risk(
                forecast, 
                potential_adjusted_forecast, 
                factor, 
                historical_mape, 
                historical_bias,
                context
            )
            
            # Get success rate for this action if available
            action_success_rate = 0.5  # Default
            if action_idx >= 0:
                action_success_rate = self.successful_adjustments[action_idx] / self.total_adjustments[action_idx]
            
            # Set thresholds based on training progress and success rate
            # More conservative early in training
            high_risk_threshold = 0.7 if self.total_actions < 1000 else 0.8
            medium_risk_threshold = 0.3 if self.total_actions < 1000 else 0.5
            
            # Adjust thresholds based on success rate 
            high_risk_threshold -= (action_success_rate - 0.5) * 0.2
            medium_risk_threshold -= (action_success_rate - 0.5) * 0.2
            
            # High risk - revert to original forecast
            if risk_score > high_risk_threshold:
                # No adjustment - safety takes priority
                return forecast
                
            # Medium risk - blend adjustment with original
            elif risk_score > medium_risk_threshold:
                # Calculate blend weight
                blend_weight = (high_risk_threshold - risk_score) / (high_risk_threshold - medium_risk_threshold)
                # Blend between original and adjusted
                return forecast * (1 - blend_weight) + potential_adjusted_forecast * blend_weight
        
        # Log unusually large adjustments
        if abs(factor - 1.0) > 0.5:
            self.logger.debug(f"Large adjustment applied: {factor:.2f}x, risk={risk_score:.2f}")
            
        return max(0, potential_adjusted_forecast)  # Ensure non-negative
    
    def _apply_band_constraints(self, factor: float, context: Optional[Dict]) -> float:
        """
        Apply band-specific constraints to adjustment factors.
        
        Args:
            factor: Original adjustment factor
            context: Context information
            
        Returns:
            Adjusted factor
        """
        if context is None:
            return factor
            
        # Apply band-specific adjustment constraints
        if 'sku_band' in context:
            band = context['sku_band']
            
            # Constraints based on SKU band
            if band in ['A', 'B']:  # Fast-selling items
                # Allow more aggressive upward adjustments for fast-selling items
                if factor > 1.0:
                    factor = factor * 1.1  # Boost upward adjustments
                
            elif band in ['D', 'E']:  # Slow-moving items
                # Constrain upward adjustments to prevent overstock for slow-moving items
                if factor > 1.0:
                    # Dampen upward adjustments for slow-moving items
                    factor = 1.0 + (factor - 1.0) * 0.7
                    
                    # If it's also a promotion or holiday, allow more upward adjustment
                    if context.get('is_promotion', False) or context.get('is_holiday', False):
                        factor = factor * 1.1  # But still boost a bit for special events
        
        # Apply context-specific adjustments
        if context.get('is_holiday', False) and factor > 1.0:
            # Boost holiday upward adjustments
            factor = factor * 1.1
        elif context.get('is_promotion', False) and factor > 1.0:
            # Boost promotion upward adjustments
            factor = factor * 1.2
        elif context.get('is_weekend', False) and factor > 1.0:
            # Boost weekend upward adjustments slightly
            factor = factor * 1.05
            
        return factor
    
    def get_action_statistics(self) -> Dict:
        """
        Get statistics on action selection in different contexts and bands.
        
        Returns:
            Dictionary of action statistics
        """
        # Calculate success rates
        success_rates = self.successful_adjustments / self.total_adjustments
        
        # Convert counts to distributions
        total_actions = np.sum(self.action_counts)
        all_dist = self.action_counts / total_actions if total_actions > 0 else np.zeros_like(self.action_counts)
        
        # Context distributions
        context_distributions = {}
        for context in ['holiday', 'promotion', 'weekend', 'weekday']:
            total = np.sum(self.context_action_counts[context])
            if total > 0:
                context_distributions[context] = {
                    'counts': self.context_action_counts[context].tolist(),
                    'distribution': (self.context_action_counts[context] / total).tolist(),
                    'success_rate': (self.context_successful_adjustments[context] / self.context_total_adjustments[context]).tolist()
                }
            else:
                context_distributions[context] = {
                    'counts': np.zeros_like(self.action_counts).tolist(),
                    'distribution': np.zeros_like(self.action_counts).tolist(),
                    'success_rate': np.zeros_like(self.action_counts).tolist()
                }
        
        # Band distributions
        band_distributions = {}
        for band in self.band_action_counts:
            total_band = np.sum(self.band_action_counts[band])
            if total_band > 0:
                band_distributions[band] = {
                    'counts': self.band_action_counts[band].tolist(),
                    'distribution': (self.band_action_counts[band] / total_band).tolist(),
                    'success_rate': (self.band_successful_adjustments[band] / self.band_total_adjustments[band]).tolist()
                }
            else:
                band_distributions[band] = {
                    'counts': np.zeros_like(self.action_counts).tolist(),
                    'distribution': np.zeros_like(self.action_counts).tolist(),
                    'success_rate': np.zeros_like(self.action_counts).tolist()
                }
        
        # Overall success rate
        overall_success_rate = self.positive_rewards / max(1, self.total_actions)
        
        # Calculate MAPE improvement statistics
        mape_stats = {
            'avg_improvement': np.mean(self.mape_improvements) if self.mape_improvements else 0,
            'positive_rate': np.mean([1 if x > 0 else 0 for x in self.mape_improvements]) if self.mape_improvements else 0,
            'recent_avg': np.mean(self.mape_improvements[-100:]) if len(self.mape_improvements) >= 100 else (np.mean(self.mape_improvements) if self.mape_improvements else 0)
        }
        
        result = {
            'overall': {
                'counts': self.action_counts.tolist(),
                'distribution': all_dist.tolist(),
                'factors': self.adjustment_factors,
                'success_rates': success_rates.tolist(),
                'overall_success_rate': overall_success_rate,
                'mape_improvement': mape_stats
            },
            'contexts': context_distributions,
            'bands': band_distributions
        }
        
        # Add backward compatibility for trainer visualization
        for context in ['holiday', 'promotion', 'weekend', 'weekday']:
            if context in context_distributions:
                result[context] = context_distributions[context]
        
        for band in band_distributions:
            result[band] = band_distributions[band]

        return result
    
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
            'action_value_sums': self.action_value_sums,
            'successful_adjustments': self.successful_adjustments,
            'total_adjustments': self.total_adjustments,
            'adjustment_factors': self.adjustment_factors,
            'mape_improvements': self.mape_improvements,
            'params': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'ucb_constant': self.ucb_constant,
                'conservative_factor': self.conservative_factor,
                'context_learning': self.context_learning
            },
            'positive_rewards': self.positive_rewards,
            'total_actions': self.total_actions
        }
        
        # Save context-specific data if using context learning
        if self.context_learning:
            data['holiday_weights'] = self.holiday_weights
            data['promo_weights'] = self.promo_weights
            data['weekend_weights'] = self.weekend_weights
            data['weekday_weights'] = self.weekday_weights
            data['band_weights'] = self.band_weights
            data['context_action_counts'] = self.context_action_counts
            data['context_value_sums'] = self.context_value_sums
            data['context_successful_adjustments'] = self.context_successful_adjustments
            data['context_total_adjustments'] = self.context_total_adjustments
            data['band_action_counts'] = self.band_action_counts
            data['band_value_sums'] = self.band_value_sums
            data['band_successful_adjustments'] = self.band_successful_adjustments
            data['band_total_adjustments'] = self.band_total_adjustments
        
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
        
        # Get parameters or use defaults for backward compatibility
        context_learning = data['params'].get('context_learning', False)
        ucb_constant = data['params'].get('ucb_constant', 2.0)
        conservative_factor = data['params'].get('conservative_factor', 0.7)
        
        # Create new agent with loaded adjustment factors
        agent = cls(
            feature_dim=data['feature_dim'],
            action_size=data['action_size'],
            learning_rate=data['params']['learning_rate'],
            gamma=data['params']['gamma'],
            ucb_constant=ucb_constant,
            conservative_factor=conservative_factor,
            adjustment_factors=data.get('adjustment_factors'),
            context_learning=context_learning,
            logger=logger
        )
        
        # Load saved state
        agent.weights = data['weights']
        agent.feature_means = data['feature_means']
        agent.feature_stds = data['feature_stds']
        agent.feature_count = data['feature_count']
        
        # Load UCB-specific data
        if 'action_counts' in data:
            agent.action_counts = data['action_counts']
        if 'action_value_sums' in data:
            agent.action_value_sums = data['action_value_sums']
            
        # Load success tracking
        if 'successful_adjustments' in data:
            agent.successful_adjustments = data['successful_adjustments']
        if 'total_adjustments' in data:
            agent.total_adjustments = data['total_adjustments']
            
        # Load MAPE improvements history
        if 'mape_improvements' in data:
            agent.mape_improvements = data['mape_improvements']
        
        # Load context-specific data if available and context learning is enabled
        if context_learning:
            if 'holiday_weights' in data:
                agent.holiday_weights = data['holiday_weights']
            if 'promo_weights' in data:
                agent.promo_weights = data['promo_weights']
            if 'weekend_weights' in data:
                agent.weekend_weights = data['weekend_weights']
            if 'weekday_weights' in data:
                agent.weekday_weights = data['weekday_weights']
            if 'band_weights' in data:
                agent.band_weights = data['band_weights']
                
            # Load context-specific UCB data
            if 'context_action_counts' in data:
                agent.context_action_counts = data['context_action_counts']
            if 'context_value_sums' in data:
                agent.context_value_sums = data['context_value_sums']
            if 'context_successful_adjustments' in data:
                agent.context_successful_adjustments = data['context_successful_adjustments']
            if 'context_total_adjustments' in data:
                agent.context_total_adjustments = data['context_total_adjustments']
            
            # Load band-specific UCB data
            if 'band_action_counts' in data:
                agent.band_action_counts = data['band_action_counts']
            if 'band_value_sums' in data:
                agent.band_value_sums = data['band_value_sums']
            if 'band_successful_adjustments' in data:
                agent.band_successful_adjustments = data['band_successful_adjustments']
            if 'band_total_adjustments' in data:
                agent.band_total_adjustments = data['band_total_adjustments']
        
        # Load success metrics if available
        if 'positive_rewards' in data:
            agent.positive_rewards = data['positive_rewards']
        if 'total_actions' in data:
            agent.total_actions = data['total_actions']
        
        if logger:
            logger.info(f"Agent loaded from {filepath}")
        
        return agent
"""
Streamlined Q-learning agent with linear function approximation for forecast adjustment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import os
import logging


class ForecastAgent:
    """
    Q-Learning agent with linear function approximation for forecast adjustment.
    """
    
    def __init__(self, 
                feature_dim: int,
                action_size: int = 11,
                learning_rate: float = 0.005,
                gamma: float = 0.95,
                ucb_constant: float = 2.0,
                conservative_factor: float = 0.7,
                adjustment_factors: Optional[List[float]] = None,
                buffer_size: int = 20000,
                context_learning: bool = True,
                logger: Optional[logging.Logger] = None):
        """Initialize Q-learning agent with linear function approximation."""
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.conservative_factor = conservative_factor
        self.buffer_size = buffer_size
        self.context_learning = context_learning
        
        # Set up logger
        self.logger = logger or logging.getLogger("ForecastAgent")
        
        # Set up adjustment factors
        if adjustment_factors is None:
            # Default factors with wider range for promotions/holidays
            self.adjustment_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
            self.action_size = len(self.adjustment_factors)
        else:
            self.adjustment_factors = adjustment_factors
            self.action_size = len(adjustment_factors)
        
        # Initialize weights using dictionaries for contexts
        self.weights = {'general': np.zeros((self.action_size, feature_dim))}
        
        # Context-specific weights (only if context_learning is enabled)
        if self.context_learning:
            contexts = ['holiday', 'promotion', 'weekend', 'weekday']
            bands = ['A', 'B', 'C', 'D', 'E']
            
            # Initialize context weights
            for context in contexts:
                self.weights[context] = np.zeros((self.action_size, feature_dim))
            
            # Initialize band weights
            for band in bands:
                self.weights[f'band_{band}'] = np.zeros((self.action_size, feature_dim))
        
        # Experience replay buffer
        self.buffer = []
        self.buffer_idx = 0
        
        # Feature normalization
        self.feature_means = np.zeros(feature_dim)
        self.feature_stds = np.ones(feature_dim)
        self.feature_count = 0
        
        # Action statistics tracking with unified structure
        self.statistics = {
            'general': {
                'counts': np.ones(self.action_size),
                'value_sums': np.zeros(self.action_size),
                'successes': np.zeros(self.action_size),
                'total': np.ones(self.action_size)
            }
        }
        
        # Initialize context-specific statistics
        if self.context_learning:
            for context in contexts + [f'band_{band}' for band in bands]:
                self.statistics[context] = {
                    'counts': np.ones(self.action_size),
                    'value_sums': np.zeros(self.action_size),
                    'successes': np.zeros(self.action_size),
                    'total': np.ones(self.action_size)
                }
        
        # Performance tracking
        self.positive_rewards = 0
        self.total_actions = 0
        self.mape_improvements = []
        
        self.logger.info(f"Initialized agent with {feature_dim} features and {self.action_size} actions")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to improve learning stability."""
        # Replace NaN/Inf with zeros
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update feature statistics (during initial training)
        if self.feature_count < 10000:
            self.feature_count += 1
            delta = features - self.feature_means
            self.feature_means += delta / self.feature_count
            
            if self.feature_count > 1:
                delta2 = features - self.feature_means
                var_sum = np.sum(delta * delta2, axis=0)
                self.feature_stds = np.sqrt(var_sum / self.feature_count + 1e-8)
        
        # Ensure feature_stds is positive
        self.feature_stds = np.maximum(self.feature_stds, 1e-8)
        
        # Normalize and clip
        normalized = np.clip((features - self.feature_means) / self.feature_stds, -10.0, 10.0)
        
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _get_context_key(self, context: Dict) -> str:
        """Determine the appropriate context key based on context dictionary."""
        if not context:
            return 'general'
            
        if context.get('is_holiday', False):
            return 'holiday'
        elif context.get('is_promotion', False):
            return 'promotion'
        elif context.get('is_weekend', False):
            return 'weekend'
        elif 'sku_band' in context:
            return f"band_{context['sku_band']}"
        else:
            return 'weekday'
    
    def get_q_values(self, features: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Calculate Q-values for all actions given the state features."""
        # Normalize features
        norm_features = self.normalize_features(features)
        
        # Use general weights as default
        q_values = np.dot(self.weights['general'], norm_features)
        
        # Apply context-specific weights if available
        if self.context_learning and context:
            context_key = self._get_context_key(context)
            
            # If we have weights for this context
            if context_key in self.weights:
                context_q = np.dot(self.weights[context_key], norm_features)
                
                # Blend with general weights
                blend_weight = 0.7  # Higher weight to context-specific values
                q_values = blend_weight * context_q + (1 - blend_weight) * q_values
        
        return np.nan_to_num(q_values, nan=0.0)
    
    def act(self, state: np.ndarray, explore: bool = True, context: Optional[Dict] = None) -> int:
        """Select an adjustment action using UCB exploration."""
        # Get Q-values
        q_values = self.get_q_values(state, context)
        
        if not explore:
            return np.argmax(q_values)
            
        # Get statistics for the appropriate context
        context_key = self._get_context_key(context) if context and self.context_learning else 'general'
        stats = self.statistics.get(context_key, self.statistics['general'])
        
        # Calculate success rates
        success_rate = stats['successes'] / stats['total']
        
        # Calculate UCB values
        total_count = np.sum(stats['counts'])
        average_values = stats['value_sums'] / stats['counts']
        
        # UCB formula with success rate modulation
        exploration = self.ucb_constant * np.sqrt(np.log(total_count) / stats['counts'])
        adjusted_exploration = exploration * (0.5 + 0.5 * success_rate)
        
        # Safety bonus for no-adjustment action
        no_adj_idx = self.adjustment_factors.index(1.0) if 1.0 in self.adjustment_factors else -1
        if no_adj_idx >= 0:
            safety_bonus = (1.0 - success_rate) * 0.5
            adjusted_exploration[no_adj_idx] += np.mean(safety_bonus)
        
        # Calculate final UCB values
        ucb_values = average_values + adjusted_exploration
        
        # Early training bias toward safer adjustments
        if self.total_actions < 1000:
            for i, factor in enumerate(self.adjustment_factors):
                distance = abs(factor - 1.0)
                ucb_values[i] -= distance * (1.0 - min(self.total_actions / 1000, 1.0))
        
        # Track action selection
        self.total_actions += 1
        
        return np.argmax(ucb_values)
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, 
              done: bool, context: Optional[Dict] = None, mape_improvement: Optional[float] = None) -> float:
        """Update weights based on experience."""
        # Validate inputs
        if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(reward):
            return 0.0
        
        # Update statistics
        self._update_statistics(action, reward, context, mape_improvement)
        
        # Store experience
        self._store_experience(state, action, reward, next_state, done, context, mape_improvement)
        
        # Update weights
        return self._update_weights(state, action, reward, next_state, done, context)
    
    def _update_statistics(self, action: int, reward: float, context: Optional[Dict], mape_improvement: Optional[float]):
        """Update action statistics and success tracking."""
        # Update general statistics
        self.statistics['general']['counts'][action] += 1
        self.statistics['general']['value_sums'][action] += reward
        
        # Track successful adjustments if MAPE improvement is provided
        if mape_improvement is not None:
            successful = mape_improvement > 0
            self.statistics['general']['total'][action] += 1
            if successful:
                self.statistics['general']['successes'][action] += 1
            
            # Store MAPE improvement
            self.mape_improvements.append(mape_improvement)
        
        # Update context-specific statistics
        if self.context_learning and context:
            context_key = self._get_context_key(context)
            if context_key in self.statistics:
                self.statistics[context_key]['counts'][action] += 1
                self.statistics[context_key]['value_sums'][action] += reward
                
                if mape_improvement is not None:
                    self.statistics[context_key]['total'][action] += 1
                    if mape_improvement > 0:
                        self.statistics[context_key]['successes'][action] += 1
        
        # Track positive rewards
        if reward > 0:
            self.positive_rewards += 1
    
    def _store_experience(self, state, action, reward, next_state, done, context, mape_improvement):
        """Store experience in replay buffer."""
        experience = (state, action, reward, next_state, done, context, mape_improvement)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            # Replace old experiences using circular buffer
            self.buffer[self.buffer_idx] = experience
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
    
    def _update_weights(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, context: Optional[Dict]) -> float:
        """Update weights based on experience."""
        # Normalize states
        norm_state = self.normalize_features(state)
        norm_next_state = self.normalize_features(next_state)
        
        # Handle NaN values
        if np.isnan(norm_state).any() or np.isinf(norm_state).any():
            return 0.0
        
        # Determine which weight matrices to update
        weight_updates = [('general', 1.0)]  # Always update general weights
        
        if self.context_learning and context:
            context_key = self._get_context_key(context)
            if context_key in self.weights:
                # Primary update for context-specific weights
                weight_updates.append((context_key, 1.0))
                
        # Apply updates to all relevant weight matrices
        td_error = 0
        for weight_key, lr_factor in weight_updates:
            weights = self.weights[weight_key]
            
            # Current Q-value
            current_q = np.dot(weights[action], norm_state)
            
            # Next state's best Q-value
            if done:
                next_q = 0
            else:
                next_q_values = self.get_q_values(next_state, context)
                next_q = np.max(next_q_values)
            
            # Calculate target and TD error
            target = reward + self.gamma * next_q
            current_td_error = target - current_q
            td_error = current_td_error  # Store for return value
            
            # Clip TD error
            td_error_clipped = np.clip(current_td_error, -50.0, 50.0)
            
            # Check for NaN/Inf
            if np.isnan(td_error_clipped) or np.isinf(td_error_clipped):
                continue
            
            # Apply learning rate and update weights
            effective_lr = (self.learning_rate * lr_factor) / (1.0 + 0.1 * abs(td_error_clipped))
            weights[action] += effective_lr * td_error_clipped * norm_state
        
        return abs(td_error)
    
    def batch_update(self, batch_size: int = 32) -> float:
        """Perform batch update using experiences from buffer."""
        if len(self.buffer) < batch_size:
            return 0.0
        
        # Simple prioritized sampling - focus on recent experiences and successful adjustments
        indices = np.random.choice(len(self.buffer), min(len(self.buffer), batch_size*2), replace=False)
        
        # Get all candidate experiences
        candidate_experiences = [self.buffer[i] for i in indices]
        
        # Prioritize experiences with positive MAPE improvements
        successful = [i for i, exp in enumerate(candidate_experiences) 
                     if exp[6] is not None and exp[6] > 0]
        
        # Select batch (half successful if possible, half random)
        successful_count = min(batch_size // 2, len(successful))
        successful_indices = np.random.choice(successful, successful_count, replace=False) if successful_count > 0 else []
        
        # Remaining random samples
        remaining = batch_size - len(successful_indices)
        remaining_indices = [i for i in range(len(candidate_experiences)) if i not in successful_indices]
        random_indices = np.random.choice(remaining_indices, remaining, replace=False) if remaining > 0 else []
        
        # Final batch
        batch_indices = list(successful_indices) + list(random_indices)
        batch = [candidate_experiences[i] for i in batch_indices]
        
        total_error = 0.0
        valid_updates = 0
        
        for experience in batch:
            state, action, reward, next_state, done, context, mape_improvement = experience
            
            # Skip invalid samples
            if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(reward):
                continue
            
            # Update weights
            td_error = self.update(state, action, reward, next_state, done, context, mape_improvement)
            total_error += td_error
            valid_updates += 1
        
        return total_error / max(1, valid_updates)
    
    def calculate_adjusted_forecast(self, 
                                  action_idx: int, 
                                  forecast: float, 
                                  context: Optional[Dict] = None,
                                  historical_mape: Optional[float] = None,
                                  historical_bias: Optional[float] = None) -> float:
        """Apply adjustment to forecast with safety mechanism."""
        # Validate action index and forecast
        action_idx = max(0, min(action_idx, len(self.adjustment_factors) - 1))
        if np.isnan(forecast):
            return 0.0
        
        # Get adjustment factor
        factor = self.adjustment_factors[action_idx]
        
        # Apply context-specific constraints (band & context)
        factor = self._apply_context_constraints(factor, context)
        
        # Calculate adjusted forecast
        adjusted_forecast = forecast * factor
        
        # Apply safety mechanism if historical metrics available
        if historical_mape is not None or historical_bias is not None:
            risk_score = self._assess_adjustment_risk(factor, historical_mape or 0.2, 
                                                     historical_bias or 0.0, context, action_idx)
            
            # Threshold-based blending
            high_threshold = 0.75
            medium_threshold = 0.4
            
            if risk_score > high_threshold:
                return forecast  # No adjustment - too risky
            elif risk_score > medium_threshold:
                blend = (high_threshold - risk_score) / (high_threshold - medium_threshold)
                return forecast * (1 - blend) + adjusted_forecast * blend
        
        return max(0, adjusted_forecast)  # Ensure non-negative
    
    def _apply_context_constraints(self, factor: float, context: Optional[Dict]) -> float:
        """Apply context-specific constraints to adjustment factors."""
        if not context:
            return factor
        
        # Apply band-specific constraints
        if 'sku_band' in context:
            band = context['sku_band']
            
            if band in ['A', 'B']:  # Fast-selling items
                if factor > 1.0:
                    factor *= 1.1  # Boost upward for fast-sellers
            
            elif band in ['D', 'E']:  # Slow-moving items
                if factor > 1.0:
                    factor = 1.0 + (factor - 1.0) * 0.7  # Dampen upward
                    
                    # Exception for promotions/holidays
                    if context.get('is_promotion') or context.get('is_holiday'):
                        factor *= 1.1  # Slight boost for special events
        
        # Apply event-specific adjustments
        if context.get('is_holiday') and factor > 1.0:
            factor *= 1.1  # Boost holiday upward adjustments
        elif context.get('is_promotion') and factor > 1.0:
            factor *= 1.2  # Boost promotion upward adjustments
            
        return factor
    
    def _assess_adjustment_risk(self, factor: float, historical_mape: float, 
                               historical_bias: float, context: Optional[Dict], action_idx: int) -> float:
        """Simplified risk assessment using point-based system."""
        # Base risk determined by adjustment magnitude
        base_risk = abs(factor - 1.0) * self.conservative_factor
        
        # Risk modifiers (positive increases risk, negative decreases it)
        modifiers = 0.0
        
        # Check bias direction alignment
        if historical_bias != 0:
            # If adjustment is in same direction as bias, it's riskier
            wrong_direction = (factor > 1.0 and historical_bias > 0) or (factor < 1.0 and historical_bias < 0)
            modifiers += 0.3 if wrong_direction else -0.2
        
        # Consider historical accuracy
        if historical_mape > 0.3:  # High error
            modifiers += 0.2
        elif historical_mape < 0.1:  # Low error
            modifiers -= 0.2
        
        # Context-specific factors
        if context:
            # Special events
            if context.get('is_holiday') or context.get('is_promotion'):
                if factor > 1.0:  # Upward during events is safer
                    modifiers -= 0.2
                else:
                    modifiers += 0.2
            
            # SKU band
            if 'sku_band' in context:
                if context['sku_band'] in ['A', 'B']:  # High-volume
                    modifiers -= 0.2  # Lower risk
                elif context['sku_band'] in ['D', 'E']:  # Low-volume
                    modifiers += 0.2  # Higher risk
        
        # Success history
        context_key = self._get_context_key(context) if context else 'general'
        stats = self.statistics.get(context_key, self.statistics['general'])
        
        success_rate = stats['successes'][action_idx] / stats['total'][action_idx]
        if success_rate > 0.7:  # High success
            modifiers -= 0.3
        elif success_rate < 0.3:  # Low success
            modifiers += 0.3
        
        # Calculate final risk
        risk = base_risk * (1.0 + modifiers)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, risk))
    
    def get_action_statistics(self) -> Dict:
        """Get statistics on action usage and success rates."""
        result = {
            'overall': {
                'counts': self.statistics['general']['counts'].tolist(),
                'distribution': (self.statistics['general']['counts'] / max(1, np.sum(self.statistics['general']['counts']))).tolist(),
                'factors': self.adjustment_factors,
                'success_rates': (self.statistics['general']['successes'] / self.statistics['general']['total']).tolist(),
                'overall_success_rate': self.positive_rewards / max(1, self.total_actions),
                'mape_improvement': {
                    'avg_improvement': np.mean(self.mape_improvements) if self.mape_improvements else 0,
                    'positive_rate': np.mean([1 if x > 0 else 0 for x in self.mape_improvements]) if self.mape_improvements else 0,
                    'recent_avg': np.mean(self.mape_improvements[-100:]) if len(self.mape_improvements) >= 100 else 
                              (np.mean(self.mape_improvements) if self.mape_improvements else 0)
                }
            }
        }
        
        # Add context-specific stats
        contexts = {}
        for key, stats in self.statistics.items():
            if key != 'general':
                total = np.sum(stats['counts'])
                if total > 0:
                    contexts[key] = {
                        'counts': stats['counts'].tolist(),
                        'distribution': (stats['counts'] / total).tolist(),
                        'success_rate': (stats['successes'] / stats['total']).tolist()
                    }
        
        if contexts:
            result['contexts'] = contexts
            
            # Add backward compatibility
            for context, stats in contexts.items():
                if context.startswith('band_'):
                    result[context[5:]] = stats  # Convert 'band_A' to 'A'
                else:
                    result[context] = stats
        
        return result
    
    def save(self, filepath: str) -> None:
        """Save agent to file."""
        data = {
            'weights': self.weights,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'feature_count': self.feature_count,
            'feature_dim': self.feature_dim,
            'action_size': self.action_size,
            'statistics': self.statistics,
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
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, logger: Optional[logging.Logger] = None) -> 'ForecastAgent':
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Get parameters
        context_learning = data['params'].get('context_learning', False)
        ucb_constant = data['params'].get('ucb_constant', 2.0)
        conservative_factor = data['params'].get('conservative_factor', 0.7)
        
        # Create new agent
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
        
        # Load statistics
        if 'statistics' in data:
            agent.statistics = data['statistics']
        elif 'action_counts' in data:
            # Handle legacy data
            agent.statistics['general']['counts'] = data['action_counts']
            agent.statistics['general']['value_sums'] = data.get('action_value_sums', np.zeros_like(data['action_counts']))
        
        # Load tracking data
        if 'mape_improvements' in data:
            agent.mape_improvements = data['mape_improvements']
        if 'positive_rewards' in data:
            agent.positive_rewards = data['positive_rewards']
        if 'total_actions' in data:
            agent.total_actions = data['total_actions']
        
        if logger:
            logger.info(f"Agent loaded from {filepath}")
        
        return agent
"""
Reward calculation for the Forecast Adjustment RL environment.
Calculates immediate and delayed rewards based on adjustment outcomes.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class RewardCalculator:
    """Calculates rewards for forecast adjustments."""
    
    def __init__(self, config):
        """
        Initialize the reward calculator.
        
        Args:
            config: Configuration dictionary with reward settings
        """
        self.config = config
        self.reward_config = config['REWARD_CONFIG']
        
        # Weights for different reward components
        self.bias_weight = self.reward_config['bias_weight']
        self.mape_weight = self.reward_config['mape_weight']
        self.consistency_penalty = self.reward_config['consistency_penalty']
        self.flip_flop_penalty = self.reward_config['flip_flop_penalty']
        self.extreme_adjustment_penalty = self.reward_config['extreme_adjustment_penalty']
    
    def calculate_immediate_reward(self, data_provider, category, band, date, 
                                  adjustment_factor, previous_adjustment):
        """
        Calculate immediate reward components (consistency-related).
        
        Args:
            data_provider: Object providing access to metrics and data
            category: Category identifier
            band: Band identifier (A, B, C)
            date: Date for which adjustment was applied
            adjustment_factor: The applied adjustment factor
            previous_adjustment: Previous adjustment factor (if any)
            
        Returns:
            Immediate reward value
        """
        immediate_reward = 0.0
        
        # If there was a previous adjustment
        if previous_adjustment is not None:
            # Get ML forecast change
            ml_change = data_provider.get_ml_forecast_change_percent(category, band)
            
            # Check if adjustment changed significantly without ML forecast changing
            adjustment_change = abs(adjustment_factor - previous_adjustment)
            if abs(ml_change) < 0.01 and adjustment_change > 0.05:
                immediate_reward -= self.consistency_penalty
                logger.debug(f"Applied consistency penalty: ML changed {ml_change:.4f}, " +
                            f"but adjustment changed {adjustment_change:.4f}")
            
            # Check for flip-flopping (reversing a recent adjustment)
            if self._is_reversal(adjustment_factor, previous_adjustment):
                immediate_reward -= self.flip_flop_penalty
                logger.debug(f"Applied flip-flop penalty: Current {adjustment_factor:.2f}, " +
                            f"Previous {previous_adjustment:.2f}")
        
        return immediate_reward
    
    def calculate_delayed_reward(self, data_provider, category, band, 
                               adjustment_date, evaluation_date, adjustment_factor):
        """
        Calculate delayed reward components when actuals become available.
        
        Args:
            data_provider: Object providing access to metrics and data
            category: Category identifier
            band: Band identifier (A, B, C)
            adjustment_date: Date when adjustment was applied
            evaluation_date: Date when actuals became available for evaluation
            adjustment_factor: The applied adjustment factor
            
        Returns:
            Delayed reward value
        """
        # Get metrics before adjustment (using original ML forecast)
        previous_bias = data_provider.get_historical_bias(
            category, band, adjustment_date, before_adjustment=True)
        previous_mape = data_provider.get_historical_mape(
            category, band, adjustment_date, before_adjustment=True)
        
        # Get metrics after adjustment (with actuals)
        current_bias = data_provider.get_bias_with_actuals(
            category, band, adjustment_date, evaluation_date)
        current_mape = data_provider.get_mape_with_actuals(
            category, band, adjustment_date, evaluation_date)
        
        # Calculate bias improvement
        if previous_bias < 0:  # Underforecasting
            # Less negative bias is an improvement
            bias_improvement = previous_bias - current_bias
        else:  # Overforecasting
            # Less positive bias is an improvement
            bias_improvement = current_bias - previous_bias
        
        # Calculate MAPE improvement (lower is better)
        mape_improvement = previous_mape - current_mape
        
        # Weighted reward calculation
        delayed_reward = (self.bias_weight * bias_improvement + 
                        self.mape_weight * mape_improvement)
        
        # Penalty for extreme adjustments that worsen metrics
        if (abs(adjustment_factor - 1.0) > 0.05 and 
            bias_improvement < 0 and mape_improvement < 0):
            delayed_reward -= self.extreme_adjustment_penalty
            logger.debug(f"Applied extreme adjustment penalty: Factor {adjustment_factor:.2f}, " +
                        f"Bias change {bias_improvement:.4f}, MAPE change {mape_improvement:.4f}")
        
        logger.debug(f"Calculated delayed reward: {delayed_reward:.4f} " +
                   f"for {category}-{band} " +
                   f"(Bias improvement: {bias_improvement:.4f}, " +
                   f"MAPE improvement: {mape_improvement:.4f})")
        
        return delayed_reward
    
    def _is_reversal(self, current_adjustment, previous_adjustment):
        """
        Check if current adjustment reverses the previous adjustment direction.
        
        Args:
            current_adjustment: Current adjustment factor
            previous_adjustment: Previous adjustment factor
            
        Returns:
            True if the current adjustment reverses the previous one
        """
        # If previous adjustment increased forecasts but current decreases them
        if previous_adjustment > 1.0 and current_adjustment < 1.0:
            return True
        
        # If previous adjustment decreased forecasts but current increases them
        if previous_adjustment < 1.0 and current_adjustment > 1.0:
            return True
        
        return False
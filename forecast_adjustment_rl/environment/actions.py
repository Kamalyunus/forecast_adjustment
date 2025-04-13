"""
Action handling for the Forecast Adjustment RL environment.
Applies adjustment factors to forecasts and manages action selection.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class ActionHandler:
    """Handles actions for the forecast adjustment RL environment."""
    
    def __init__(self, config):
        """
        Initialize the action handler.
        
        Args:
            config: Configuration dictionary with action settings
        """
        self.config = config
        self.action_config = config['ACTION_CONFIG']
        self.adjustment_factors = self.action_config['adjustment_factors']
        self.action_dim = len(self.adjustment_factors)
    
    def get_action_dim(self):
        """Return the dimension of the action space."""
        return self.action_dim
    
    def get_adjustment_factor(self, action_idx):
        """
        Get the adjustment factor for a given action index.
        
        Args:
            action_idx: Index into the adjustment_factors list
            
        Returns:
            The adjustment factor as a float
        """
        return self.adjustment_factors[action_idx]
    
    def get_action_idx(self, adjustment_factor):
        """
        Get the action index for a given adjustment factor.
        
        Args:
            adjustment_factor: Adjustment factor value
            
        Returns:
            Index of the closest adjustment factor
        """
        # Find the closest adjustment factor
        idx = min(range(len(self.adjustment_factors)), 
                  key=lambda i: abs(self.adjustment_factors[i] - adjustment_factor))
        return idx
    
    def apply_adjustment(self, data_provider, category, band, date, action_idx):
        """
        Apply an adjustment to forecasts for a category-band combination.
        
        Args:
            data_provider: Object providing access to forecasts and adjustment storage
            category: Category identifier
            band: Band identifier (A, B, C)
            date: Date for which to apply adjustment
            action_idx: Index of the selected adjustment factor
            
        Returns:
            Dictionary with information about the applied adjustment
        """
        adjustment_factor = self.get_adjustment_factor(action_idx)
        
        # Get all SKUs in this category-band
        skus = data_provider.get_skus_for_category_band(category, band)
        
        if not skus:
            logger.warning(f"No SKUs found for {category}-{band}")
            return {
                'category': category,
                'band': band,
                'date': date,
                'adjustment_factor': adjustment_factor,
                'action_idx': action_idx,
                'num_skus': 0
            }
        
        # Get ML forecasts for these SKUs
        ml_forecasts = data_provider.get_ml_forecasts(skus, date)
        
        # Calculate adjusted forecasts
        adjusted_forecasts = {sku: ml_forecasts[sku] * adjustment_factor for sku in skus}
        
        # Store the adjustments
        data_provider.save_adjustments(skus, adjusted_forecasts, date, adjustment_factor)
        
        # Log the adjustment
        logger.info(f"Applied adjustment factor {adjustment_factor} to {category}-{band} " +
                    f"on {date.strftime('%Y-%m-%d')} ({len(skus)} SKUs)")
        
        return {
            'category': category,
            'band': band,
            'date': date,
            'adjustment_factor': adjustment_factor,
            'action_idx': action_idx,
            'num_skus': len(skus)
        }
    
    def get_max_action(self, action_probs):
        """
        Get the action with highest probability.
        
        Args:
            action_probs: Action probability distribution from policy network
            
        Returns:
            Index of the action with highest probability
        """
        return np.argmax(action_probs)
    
    def sample_action(self, action_probs):
        """
        Sample an action from the probability distribution.
        
        Args:
            action_probs: Action probability distribution from policy network
            
        Returns:
            Sampled action index
        """
        return np.random.choice(self.action_dim, p=action_probs)
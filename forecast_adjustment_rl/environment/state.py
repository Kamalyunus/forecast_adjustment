"""
State representation for the Forecast Adjustment RL environment.
Builds state vectors for the RL agent based on forecast and sales data.
"""

import numpy as np
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class StateBuilder:
    """Builds state representations for the RL agent."""
    
    def __init__(self, config):
        """
        Initialize the state builder.
        
        Args:
            config: Configuration dictionary with state settings
        """
        self.config = config
        self.feature_config = config['FEATURE_CONFIG']
        self.state_config = config['STATE_CONFIG']
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(config)
        
        # Track which features are enabled
        self.enabled_features = self.feature_config['features']
        
        # Calculate state dimensionality
        self.state_dim = self._calculate_state_dim()
        
        logger.info(f"Initialized StateBuilder with {self.state_dim} dimensional state space")
    
    def _calculate_state_dim(self):
        """Calculate the dimension of the state vector."""
        dim = 0
        
        # Count all enabled features
        if 'mape_short' in self.enabled_features:
            dim += 1
        if 'mape_long' in self.enabled_features:
            dim += 1
        if 'bias_short' in self.enabled_features:
            dim += 1
        if 'bias_long' in self.enabled_features:
            dim += 1
        if 'mape_trend' in self.enabled_features:
            dim += 1
        if 'bias_trend' in self.enabled_features:
            dim += 1
        
        # Week of month indicators (WoM1-4)
        if 'week_of_month' in self.enabled_features:
            dim += 4  # One-hot encoded
        
        # Month of year
        if 'month_of_year' in self.enabled_features:
            dim += 12  # One-hot encoded
        
        # Sales metrics
        if 'sales_volume' in self.enabled_features:
            dim += 1
        if 'sales_volatility' in self.enabled_features:
            dim += 1
        
        # Band indicators
        if 'band' in self.enabled_features:
            dim += 3  # One-hot encoded (A, B, C)
        
        # Forecast behavior
        if 'forecast_momentum' in self.enabled_features:
            dim += 1
        if 'forecast_revision_rate' in self.enabled_features:
            dim += 1
        
        # Adjustment history
        if 'previous_adjustment' in self.enabled_features:
            dim += 1
        if 'adjustment_age' in self.enabled_features:
            dim += 1
        if 'adjustment_success_rate' in self.enabled_features:
            dim += 1
        
        return dim
    
    def build_state(self, data_provider, category, band, date):
        """
        Build state vector for a specific category-band on a specific date.
        
        Args:
            data_provider: Object providing access to metrics and data
            category: Category identifier
            band: Band identifier (A, B, C)
            date: Date for which to build the state
            
        Returns:
            numpy array with the state representation
        """
        # Use the feature engineer to build state vector
        return self.feature_engineer.build_state_vector(data_provider, category, band, date)
    
    def get_state_dim(self):
        """Return the dimension of the state vector."""
        return self.state_dim
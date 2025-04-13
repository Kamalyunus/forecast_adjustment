"""
Feature engineering for the Forecast Adjustment RL system.
Transforms raw data into features for state representation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_week_of_month, min_max_normalize, z_score_normalize

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Transforms raw data into features for state representation.
    """
    
    def __init__(self, config):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config['FEATURE_CONFIG']
        
        # Feature normalization parameters (will be updated during fit)
        self.normalization_params = {}
    
    def extract_temporal_features(self, date):
        """
        Extract temporal features from a date.
        
        Args:
            date: Datetime object
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Week of Month (WoM)
        wom = get_week_of_month(date)
        features['wom'] = wom
        
        # One-hot encoded WoM
        for i in range(1, 5):
            features[f'wom_{i}'] = 1 if wom == i else 0
        
        # Month features
        features['month'] = date.month
        
        # One-hot encoded month
        for i in range(1, 13):
            features[f'month_{i}'] = 1 if date.month == i else 0
        
        # Day of week
        dow = date.weekday()
        features['dow'] = dow
        
        # One-hot encoded day of week
        for i in range(7):
            features[f'dow_{i}'] = 1 if dow == i else 0
        
        # Quarter
        quarter = (date.month - 1) // 3 + 1
        features['quarter'] = quarter
        
        # One-hot encoded quarter
        for i in range(1, 5):
            features[f'quarter_{i}'] = 1 if quarter == i else 0
        
        # Days until end of month
        if date.month == 12:
            next_month = datetime(date.year + 1, 1, 1)
        else:
            next_month = datetime(date.year, date.month + 1, 1)
        
        days_in_month = (next_month - datetime(date.year, date.month, 1)).days
        days_until_eom = days_in_month - date.day + 1
        features['days_until_eom'] = days_until_eom
        
        return features
    
    def extract_forecast_features(self, data_provider, category, band, date, lookback_weeks=4):
        """
        Extract forecast performance features.
        
        Args:
            data_provider: DataProvider object
            category: Category identifier
            band: Band identifier
            date: Date for feature extraction
            lookback_weeks: Number of weeks to look back
            
        Returns:
            Dictionary of forecast features
        """
        features = {}
        
        # Get MAPE for different lookback periods
        features['mape_1w'] = data_provider.get_mape(category, band, lookback_weeks=1)
        features['mape_4w'] = data_provider.get_mape(category, band, lookback_weeks=4)
        
        # Get bias for different lookback periods
        features['bias_1w'] = data_provider.get_bias(category, band, lookback_weeks=1)
        features['bias_4w'] = data_provider.get_bias(category, band, lookback_weeks=4)
        
        # Calculate trends
        features['mape_trend'] = features['mape_4w'] - features['mape_1w']
        features['bias_trend'] = self._calculate_bias_trend(features['bias_1w'], features['bias_4w'])
        
        # ML forecast momentum
        features['forecast_momentum'] = data_provider.get_forecast_momentum(category, band)
        
        # ML forecast revision rate
        features['forecast_revision'] = data_provider.get_forecast_revision_rate(category, band)
        
        return features
    
    def extract_sales_features(self, data_provider, category, band):
        """
        Extract sales pattern features.
        
        Args:
            data_provider: DataProvider object
            category: Category identifier
            band: Band identifier
            
        Returns:
            Dictionary of sales features
        """
        features = {}
        
        # Volume and volatility
        features['sales_volume'] = data_provider.get_sales_volume(category, band)
        features['sales_volatility'] = data_provider.get_sales_volatility(category, band)
        
        return features
    
    def extract_adjustment_features(self, data_provider, category, band):
        """
        Extract adjustment history features.
        
        Args:
            data_provider: DataProvider object
            category: Category identifier
            band: Band identifier
            
        Returns:
            Dictionary of adjustment features
        """
        features = {}
        
        # Previous adjustment
        features['prev_adjustment'] = data_provider.get_previous_adjustment(category, band)
        
        # Age of adjustment in days
        features['adjustment_age'] = data_provider.get_adjustment_age(category, band)
        
        # Success rate of previous adjustments
        features['adjustment_success'] = data_provider.get_adjustment_success_rate(category, band)
        
        return features
    
    def extract_category_band_features(self, category, band):
        """
        Extract category and band features.
        
        Args:
            category: Category identifier
            band: Band identifier
            
        Returns:
            Dictionary of category-band features
        """
        features = {}
        
        # One-hot encoded band
        features['band_A'] = 1 if band == 'A' else 0
        features['band_B'] = 1 if band == 'B' else 0
        features['band_C'] = 1 if band == 'C' else 0
        
        # Category features
        # In a real implementation, would use category embeddings or other representations
        # For POC, just use a simplified approach
        
        # Extract a number from the category name if possible
        if isinstance(category, str) and '_' in category:
            category_parts = category.split('_')
            if len(category_parts) > 1 and category_parts[-1].isdigit():
                features['category_id'] = int(category_parts[-1])
            else:
                features['category_id'] = hash(category) % 100  # Simple hash
        else:
            features['category_id'] = hash(str(category)) % 100  # Simple hash
        
        return features
    
    def build_feature_vector(self, data_provider, category, band, date, normalize=True):
        """
        Build a complete feature vector for state representation.
        
        Args:
            data_provider: DataProvider object
            category: Category identifier
            band: Band identifier
            date: Date for feature extraction
            normalize: Whether to normalize features
            
        Returns:
            Dictionary with all features
        """
        all_features = {}
        
        # Temporal features
        temporal_features = self.extract_temporal_features(date)
        all_features.update(temporal_features)
        
        # Forecast features
        forecast_features = self.extract_forecast_features(data_provider, category, band, date)
        all_features.update(forecast_features)
        
        # Sales features
        sales_features = self.extract_sales_features(data_provider, category, band)
        all_features.update(sales_features)
        
        # Adjustment features
        adjustment_features = self.extract_adjustment_features(data_provider, category, band)
        all_features.update(adjustment_features)
        
        # Category-band features
        category_band_features = self.extract_category_band_features(category, band)
        all_features.update(category_band_features)
        
        # Normalize features if requested
        if normalize:
            all_features = self._normalize_features(all_features)
        
        return all_features
    
    def build_state_vector(self, data_provider, category, band, date):
        """
        Build state vector for RL agent.
        
        Args:
            data_provider: DataProvider object
            category: Category identifier
            band: Band identifier
            date: Date for state construction
            
        Returns:
            Numpy array with state representation
        """
        # Get all features
        features = self.build_feature_vector(data_provider, category, band, date)
        
        # Select and order features based on configuration
        enabled_features = self.feature_config['features']
        
        # Map config feature names to actual feature keys
        feature_mapping = {
            'mape_short': 'mape_1w',
            'mape_long': 'mape_4w',
            'bias_short': 'bias_1w',
            'bias_long': 'bias_4w',
            'mape_trend': 'mape_trend',
            'bias_trend': 'bias_trend',
            'week_of_month': ['wom_1', 'wom_2', 'wom_3', 'wom_4'],
            'month_of_year': [f'month_{i}' for i in range(1, 13)],
            'sales_volume': 'sales_volume',
            'sales_volatility': 'sales_volatility',
            'band': ['band_A', 'band_B', 'band_C'],
            'forecast_momentum': 'forecast_momentum',
            'forecast_revision_rate': 'forecast_revision',
            'previous_adjustment': 'prev_adjustment',
            'adjustment_age': 'adjustment_age',
            'adjustment_success_rate': 'adjustment_success'
        }
        
        # Build ordered state vector
        state = []
        for feature_name in enabled_features:
            if feature_name in feature_mapping:
                mapping = feature_mapping[feature_name]
                if isinstance(mapping, list):
                    # Multiple features (one-hot encoded)
                    for key in mapping:
                        state.append(features.get(key, 0.0))
                else:
                    # Single feature
                    state.append(features.get(mapping, 0.0))
        
        return np.array(state, dtype=np.float32)
    
    def _normalize_features(self, features):
        """
        Normalize feature values using appropriate methods.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Dictionary of normalized features
        """
        normalized = {}
        
        # Features that need min-max normalization (0-1 range)
        min_max_features = [
            'sales_volume', 'prev_adjustment', 'category_id'
        ]
        
        # Features that need z-score normalization (mean 0, std 1)
        z_score_features = [
            'mape_1w', 'mape_4w', 'bias_1w', 'bias_4w', 
            'mape_trend', 'bias_trend', 'forecast_momentum',
            'forecast_revision', 'sales_volatility', 'adjustment_success'
        ]
        
        # Features to cap at specific ranges
        capped_features = {
            'adjustment_age': (0, 30)  # Cap at 30 days
        }
        
        # Binary features (no normalization needed)
        binary_features = [
            'wom_1', 'wom_2', 'wom_3', 'wom_4',
            'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
            'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
            'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
            'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4',
            'band_A', 'band_B', 'band_C'
        ]
        
        # Apply normalization for each feature
        for key, value in features.items():
            if key in binary_features:
                # No normalization for binary features
                normalized[key] = value
            elif key in min_max_features:
                # Min-max normalization
                if key in self.normalization_params:
                    min_val, max_val = self.normalization_params[key]
                    normalized[key] = min_max_normalize([value], min_val, max_val)[0]
                else:
                    # If no params yet, use default range based on feature
                    if key == 'sales_volume':
                        normalized[key] = min_max_normalize([value], 0, 1000)[0]
                    elif key == 'prev_adjustment':
                        normalized[key] = min_max_normalize([value], 0.8, 1.2)[0]
                    else:
                        normalized[key] = min_max_normalize([value], 0, 100)[0]
            elif key in z_score_features:
                # Z-score normalization
                if key in self.normalization_params:
                    mean, std = self.normalization_params[key]
                    normalized[key] = z_score_normalize([value], mean, std)[0]
                else:
                    # If no params yet, use default parameters based on feature
                    if key.startswith('mape'):
                        normalized[key] = z_score_normalize([value], 0.15, 0.1)[0]
                    elif key.startswith('bias'):
                        normalized[key] = z_score_normalize([value], 0, 0.1)[0]
                    else:
                        normalized[key] = z_score_normalize([value])[0]
            elif key in capped_features:
                # Cap features at specific range
                min_val, max_val = capped_features[key]
                capped_value = max(min_val, min(value, max_val))
                normalized[key] = (capped_value - min_val) / (max_val - min_val)
            else:
                # Pass through unchanged
                normalized[key] = value
        
        return normalized
    
    def fit_normalizers(self, data_provider, categories, bands, dates):
        """
        Fit normalizers to data.
        
        Args:
            data_provider: DataProvider object
            categories: List of categories
            bands: List of bands
            dates: List of dates
            
        Returns:
            Self with fitted normalization parameters
        """
        # Collect all feature values
        feature_values = {}
        
        for category in categories:
            for band in bands:
                for date in dates:
                    features = self.build_feature_vector(
                        data_provider, category, band, date, normalize=False
                    )
                    
                    for key, value in features.items():
                        if key not in feature_values:
                            feature_values[key] = []
                        feature_values[key].append(value)
        
        # Calculate normalization parameters
        for key, values in feature_values.items():
            if len(values) > 0:
                values = np.array(values)
                
                if key in ['sales_volume', 'prev_adjustment', 'category_id']:
                    # Min-max normalization parameters
                    self.normalization_params[key] = (np.min(values), np.max(values))
                elif key in ['mape_1w', 'mape_4w', 'bias_1w', 'bias_4w', 
                            'mape_trend', 'bias_trend', 'forecast_momentum',
                            'forecast_revision', 'sales_volatility', 'adjustment_success']:
                    # Z-score normalization parameters
                    self.normalization_params[key] = (np.mean(values), np.std(values))
        
        logger.info(f"Fitted normalizers for {len(self.normalization_params)} features")
        return self
    
    def _calculate_bias_trend(self, short_bias, long_bias):
        """
        Calculate bias trend (positive means improving).
        
        Args:
            short_bias: Short-term bias
            long_bias: Long-term bias
            
        Returns:
            Bias trend value
        """
        # If both are negative (underforecasting), improvement is when short is less negative
        if short_bias < 0 and long_bias < 0:
            return short_bias - long_bias
        
        # If both are positive (overforecasting), improvement is when short is less positive
        elif short_bias > 0 and long_bias > 0:
            return long_bias - short_bias
        
        # If one is positive and one is negative, improvement is moving toward zero
        else:
            return -abs(short_bias) + abs(long_bias)
"""
Inference module for the Forecast Adjustment RL system.
Provides functions for making forecast adjustments using trained models.
"""

import logging
from datetime import datetime
import pandas as pd
import os

from models.agent import ForecastAdjustmentAgent
from environment.state import StateBuilder
from environment.actions import ActionHandler
from data.data_loader import DataProvider

logger = logging.getLogger(__name__)

class ForecastAdjuster:
    """
    Makes forecast adjustments using a trained RL agent.
    """
    
    def __init__(self, config, model_path=None):
        """
        Initialize the forecast adjuster.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model (if None, uses latest model)
        """
        self.config = config
        self.system_config = config['SYSTEM_CONFIG']
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.system_config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure directories exist
        import os
        os.makedirs('data', exist_ok=True)
        os.makedirs(self.config['TRAINING_CONFIG']['model_dir'], exist_ok=True)
        
        # Initialize components
        self.data_provider = DataProvider(config)
        self.state_builder = StateBuilder(config)
        self.action_handler = ActionHandler(config)
        
        # Initialize RL agent
        self.agent = ForecastAdjustmentAgent(
            self.state_builder.get_state_dim(),
            self.action_handler.get_action_dim(),
            config
        )
        
        # Load model
        if model_path is None:
            model_path = self._find_latest_model()
        
        if model_path:
            success = self.agent.load_model(model_path)
            if not success:
                logger.warning(f"Could not load model from {model_path}. Using untrained model.")
            else:
                logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No pre-trained model found. Using untrained model.")
            # Since we're starting with an untrained model, let's save it
            model_dir = self.config['TRAINING_CONFIG']['model_dir']
            model_path = os.path.join(model_dir, 'forecast_adjustment_model_initial.pt')
            self.agent.save_model(model_path)
            logger.info(f"Saved initial model to {model_path}")
    
    def _find_latest_model(self):
        """
        Find the latest trained model.
        
        Returns:
            Path to the latest model or None if no models found
        """
        model_dir = self.config['TRAINING_CONFIG']['model_dir']
        
        if not os.path.exists(model_dir):
            # Create the directory since we're starting fresh
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")
            return None
        
        # Find all model files
        model_files = [
            os.path.join(model_dir, f) for f in os.listdir(model_dir)
            if f.startswith('forecast_adjustment_model_') and f.endswith('.pt')
        ]
        
        if not model_files:
            logger.warning(f"No model files found in {model_dir}")
            return None
        
        # Return the most recent model
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"Found latest model: {latest_model}")
        return latest_model
    
    def adjust_forecasts(self, date=None, categories=None, bands=None, top_n=None):
        """
        Apply forecast adjustments using the trained model.
        
        Args:
            date: Date for which to adjust forecasts (defaults to today)
            categories: List of categories to adjust (defaults to all)
            bands: List of bands to adjust (defaults to A, B, C)
            top_n: Only adjust top N categories by volume (if provided)
            
        Returns:
            DataFrame with adjustment details
        """
        if date is None:
            date = datetime.now()
        
        if categories is None:
            # Get all categories
            categories = sorted(self.data_provider.ml_forecasts['category'].unique())
        
        if bands is None:
            bands = ['A', 'B', 'C']
        
        # If top_n is provided, filter to top categories by volume
        if top_n is not None and top_n < len(categories):
            category_volumes = []
            
            for category in categories:
                volume = sum(
                    self.data_provider.get_sales_volume(category, band)
                    for band in bands
                )
                category_volumes.append((category, volume))
            
            # Sort by volume (descending) and take top N
            category_volumes.sort(key=lambda x: x[1], reverse=True)
            categories = [c for c, _ in category_volumes[:top_n]]
            
            logger.info(f"Selected top {top_n} categories by volume")
        
        logger.info(f"Adjusting forecasts for {len(categories)} categories " +
                  f"and {len(bands)} bands on {date.strftime('%Y-%m-%d')}")
        
        adjustments = []
        
        # Process each category and band
        for category in categories:
            for band in bands:
                # Get adjustment from the agent
                adjustment_factor, action_idx, action_probs = self.agent.get_adjustment_for_category_band(
                    self.data_provider, category, band, date, training=False
                )
                
                # Apply adjustment if not 1.0 (no adjustment)
                if adjustment_factor != 1.0:
                    # Get skus for this category-band
                    skus = self.data_provider.get_skus_for_category_band(category, band)
                    
                    # Get ML forecasts
                    ml_forecasts = self.data_provider.get_ml_forecasts(skus, date)
                    
                    # Calculate adjusted forecasts
                    adjusted_forecasts = {
                        sku: ml_forecasts[sku] * adjustment_factor for sku in skus
                    }
                    
                    # Save adjustments
                    self.data_provider.save_adjustments(
                        skus, adjusted_forecasts, date, adjustment_factor
                    )
                    
                    # Add to adjustments list
                    adjustments.append({
                        'category': category,
                        'band': band,
                        'date': date,
                        'adjustment_factor': adjustment_factor,
                        'num_skus': len(skus),
                        'total_forecast': sum(ml_forecasts.values()),
                        'total_adjusted': sum(adjusted_forecasts.values()),
                        'action_probability': action_probs[action_idx],
                        'confidence': max(action_probs)
                    })
                    
                    logger.info(f"Applied {adjustment_factor:.2f} to {category}-{band} " +
                              f"({len(skus)} SKUs, confidence: {max(action_probs):.2f})")
                else:
                    logger.debug(f"No adjustment needed for {category}-{band}")
        
        # Return as DataFrame
        return pd.DataFrame(adjustments)
    
    def explain_adjustment(self, category, band, date=None):
        """
        Explain why a specific adjustment was made.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            date: Date for which to explain adjustment (defaults to today)
            
        Returns:
            Dictionary with explanation details
        """
        if date is None:
            date = datetime.now()
        
        # Build state for this category-band
        state = self.state_builder.build_state(
            self.data_provider, category, band, date
        )
        
        # Get action probabilities
        _, action_probs = self.agent.select_action(state, training=False)
        
        # Get key metrics
        week_of_month = (date.day - 1) // 7 + 1
        short_bias = self.data_provider.get_bias(category, band, lookback_weeks=1)
        long_bias = self.data_provider.get_bias(category, band, lookback_weeks=4)
        mape = self.data_provider.get_mape(category, band, lookback_weeks=1)
        
        # Determine primary reason for adjustment
        primary_reason = "Unknown"
        if week_of_month == 1 and short_bias < -0.05:
            primary_reason = "Week of Month 1 underforecasting correction"
        elif week_of_month != 1 and short_bias < -0.05:
            primary_reason = "Consistent underforecasting correction"
        elif short_bias > 0.05:
            primary_reason = "Overforecasting correction"
        elif abs(short_bias - long_bias) > 0.1:
            primary_reason = "Correcting recent bias change"
        
        # Get most likely adjustment
        best_action_idx = max(range(len(action_probs)), key=lambda i: action_probs[i])
        best_adjustment = self.action_handler.get_adjustment_factor(best_action_idx)
        
        # Check if this matches any previous adjustments
        prev_adjustment = self.data_provider.get_previous_adjustment(category, band)
        adjustment_age = self.data_provider.get_adjustment_age(category, band)
        adjustment_changed = abs(best_adjustment - prev_adjustment) > 0.01
        
        explanation = {
            'category': category,
            'band': band,
            'date': date,
            'best_adjustment': best_adjustment,
            'previous_adjustment': prev_adjustment,
            'adjustment_age_days': adjustment_age,
            'adjustment_changed': adjustment_changed,
            'week_of_month': week_of_month,
            'short_bias': short_bias,
            'long_bias': long_bias,
            'mape': mape,
            'primary_reason': primary_reason,
            'action_probabilities': {
                str(self.action_handler.get_adjustment_factor(i)): prob
                for i, prob in enumerate(action_probs)
            },
            'confidence': max(action_probs)
        }
        
        return explanation
"""
Enhanced Forecast Environment Module - Environment for forecast adjustment with
support for calendar effects, holidays, and promotions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta


class ForecastEnvironment:
    """
    Environment for training RL agents to adjust forecasts, with support for
    calendar effects, holidays, and promotions.
    """
    
    def __init__(self, 
                forecast_data: pd.DataFrame,
                historical_data: Optional[pd.DataFrame] = None,
                holiday_data: Optional[pd.DataFrame] = None,
                promotion_data: Optional[pd.DataFrame] = None,
                validation_length: int = 30,
                forecast_horizon: int = 14,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                start_date: Optional[str] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the enhanced forecast adjustment environment.
        
        Args:
            forecast_data: DataFrame with forecast data (ml_day_X columns)
            historical_data: Optional DataFrame with actual historical values for validation
            holiday_data: Optional DataFrame with holiday information
            promotion_data: Optional DataFrame with promotion information
            validation_length: Number of days to use for validation
            forecast_horizon: Number of days in the forecast horizon
            optimize_for: Which metric to optimize for ("mape", "bias", or "both")
            start_date: Starting date for the forecast period (format: 'YYYY-MM-DD')
            logger: Optional logger instance
        """
        self.forecast_data = forecast_data
        self.historical_data = historical_data
        self.validation_length = validation_length
        self.forecast_horizon = forecast_horizon
        self.optimize_for = optimize_for
        
        # Set up logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ForecastEnvironment")
        else:
            self.logger = logger
            
        # Set up calendar information
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            # Default to current date if not provided
            self.start_date = pd.to_datetime(datetime.now())
        
        # Extract SKUs
        self.skus = self.forecast_data['sku_id'].unique().tolist()
        self.logger.info(f"Environment initialized with {len(self.skus)} SKUs")
        
        # Extract ML forecast columns
        self.ml_cols = [col for col in self.forecast_data.columns if col.startswith('ml_day_')]
        
        # Extract accuracy metrics
        self.accuracy_cols = [
            'ml_mape_7d', 'ml_mape_30d', 'ml_bias_7d', 'ml_bias_30d'
        ]
        
        # Process holiday data if provided
        self.holiday_calendar = None
        if holiday_data is not None:
            self._setup_holiday_calendar(holiday_data)
        
        # Process promotion data if provided
        self.promotion_schedule = None
        if promotion_data is not None:
            self._setup_promotion_schedule(promotion_data)
        
        # Create validation data if historical data is provided
        self.has_validation = False
        if historical_data is not None:
            self._setup_validation_data()
            self.has_validation = True
            
        # Current state of the environment
        self.current_step = 0
        self.current_state = None
        self.done = False
        
        # History tracking
        self.adjustment_history = []
    
    def _setup_holiday_calendar(self, holiday_data: pd.DataFrame):
        """
        Process holiday data into a usable format.
        
        Args:
            holiday_data: DataFrame with columns 'date' and 'holiday_name'
        """
        self.logger.info("Setting up holiday calendar")
        
        # Initialize holiday calendar
        max_days = max(len(self.ml_cols), self.validation_length) + self.forecast_horizon
        self.holiday_calendar = {i: False for i in range(max_days)}
        self.holiday_names = {i: "" for i in range(max_days)}
        
        # Process holiday data
        try:
            for _, row in holiday_data.iterrows():
                holiday_date = pd.to_datetime(row['date'])
                holiday_name = row['holiday_name']
                
                # Calculate day index relative to start date
                delta = (holiday_date - self.start_date).days
                
                if 0 <= delta < max_days:
                    self.holiday_calendar[delta] = True
                    self.holiday_names[delta] = holiday_name
            
            self.logger.info(f"Processed {sum(self.holiday_calendar.values())} holidays")
        except Exception as e:
            self.logger.warning(f"Error processing holiday data: {str(e)}")
            # Continue with empty holiday calendar
    
    def _setup_promotion_schedule(self, promotion_data: pd.DataFrame):
        """
        Process promotion data into a usable format.
        
        Args:
            promotion_data: DataFrame with columns 'sku_id', 'start_date', 'end_date', 'promo_type'
        """
        self.logger.info("Setting up promotion schedule")
        
        # Initialize promotion schedule
        self.promotion_schedule = {}
        self.promotion_types = {}
        
        # Process promotion data
        try:
            for _, row in promotion_data.iterrows():
                sku = row['sku_id']
                
                # Skip if SKU not in our dataset
                if sku not in self.skus:
                    continue
                
                # Convert dates
                start_date = pd.to_datetime(row['start_date'])
                end_date = pd.to_datetime(row['end_date'])
                promo_type = row.get('promo_type', 'generic')
                
                # Calculate day indices relative to start date
                start_idx = (start_date - self.start_date).days
                end_idx = (end_date - self.start_date).days
                
                # Store promotion information
                for day_idx in range(max(0, start_idx), min(end_idx + 1, len(self.ml_cols) + self.forecast_horizon)):
                    key = (sku, day_idx)
                    self.promotion_schedule[key] = True
                    self.promotion_types[key] = promo_type
            
            self.logger.info(f"Processed {len(self.promotion_schedule)} promotion days")
        except Exception as e:
            self.logger.warning(f"Error processing promotion data: {str(e)}")
            # Continue with empty promotion schedule
    
    def _generate_calendar_features(self, step: int) -> Dict[int, Dict[str, float]]:
        """
        Generate calendar features (day of week, day of month) for the forecast horizon.
        
        Args:
            step: Current step in the episode
            
        Returns:
            Dictionary mapping day indices to calendar features
        """
        calendar_features = {}
        
        for i in range(self.forecast_horizon):
            day_idx = step + i
            current_date = self.start_date + pd.Timedelta(days=day_idx)
            
            # Extract calendar features
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            day_of_month = current_date.day
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
            
            # Store features
            calendar_features[day_idx] = {
                'day_of_week': day_of_week / 6.0,  # Normalize to [0,1]
                'day_of_month': day_of_month / 31.0,  # Normalize to [0,1]
                'is_weekend': is_weekend
            }
        
        return calendar_features
    
    def _check_if_holiday(self, day_idx: int) -> bool:
        """
        Check if a given day is a holiday.
        
        Args:
            day_idx: Day index relative to start date
            
        Returns:
            True if the day is a holiday
        """
        if self.holiday_calendar and day_idx in self.holiday_calendar:
            return self.holiday_calendar[day_idx]
        return False
    
    def _check_if_promotion(self, sku: str, day_idx: int) -> bool:
        """
        Check if a given SKU is on promotion for a specific day.
        
        Args:
            sku: SKU identifier
            day_idx: Day index relative to start date
            
        Returns:
            True if the SKU is on promotion
        """
        if self.promotion_schedule:
            key = (sku, day_idx)
            return key in self.promotion_schedule
        return False
    
    def _get_promotion_type(self, sku: str, day_idx: int) -> str:
        """
        Get the promotion type for a given SKU and day.
        
        Args:
            sku: SKU identifier
            day_idx: Day index relative to start date
            
        Returns:
            Promotion type or empty string if no promotion
        """
        if self.promotion_types:
            key = (sku, day_idx)
            return self.promotion_types.get(key, "")
        return ""
    
    def _setup_validation_data(self):
        """
        Set up validation data from historical data.
        """
        self.validation_data = {}
        
        # Organize historical data by SKU and date
        if 'sale_date' in self.historical_data.columns and 'quantity' in self.historical_data.columns:
            for sku in self.skus:
                sku_history = self.historical_data[self.historical_data['sku_id'] == sku].copy()
                if not sku_history.empty:
                    try:
                        # Try to parse dates with flexible format
                        sku_history['sale_date'] = pd.to_datetime(sku_history['sale_date'])
                        sku_history = sku_history.sort_values('sale_date')
                        
                        # Use the most recent data for validation
                        if len(sku_history) >= self.validation_length:
                            self.validation_data[sku] = sku_history.tail(self.validation_length)['quantity'].values
                        else:
                            # Pad with zeros if not enough data
                            actual_values = sku_history['quantity'].values
                            self.validation_data[sku] = np.pad(
                                actual_values, 
                                (0, self.validation_length - len(actual_values)),
                                'constant'
                            )
                    except Exception as e:
                        self.logger.warning(f"Error processing dates for SKU {sku}: {str(e)}")
                        # If date processing fails, sort by index as fallback
                        if len(sku_history) >= self.validation_length:
                            self.validation_data[sku] = sku_history.tail(self.validation_length)['quantity'].values
                        else:
                            actual_values = sku_history['quantity'].values
                            self.validation_data[sku] = np.pad(
                                actual_values, 
                                (0, self.validation_length - len(actual_values)),
                                'constant'
                            )
        
        # If not enough validation data, create synthetic data
        missing_skus = set(self.skus) - set(self.validation_data.keys())
        if missing_skus:
            self.logger.warning(f"Creating synthetic validation data for {len(missing_skus)} SKUs")
            
            for sku in missing_skus:
                # Get forecast data for this SKU
                sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
                
                if not sku_forecast.empty:
                    # Extract ML forecasts for the first few days
                    ml_forecasts = []
                    for col in self.ml_cols[:self.validation_length]:
                        if col in sku_forecast.columns:
                            ml_forecasts.append(sku_forecast[col].iloc[0])
                        else:
                            ml_forecasts.append(0)
                    
                    # Create synthetic actuals by adding noise to forecasts
                    if 'ml_mape_7d' in sku_forecast.columns:
                        mape = sku_forecast['ml_mape_7d'].iloc[0]
                    else:
                        mape = 0.2  # Default 20% MAPE
                        
                    if 'ml_bias_7d' in sku_forecast.columns:
                        bias = sku_forecast['ml_bias_7d'].iloc[0]
                    else:
                        bias = 0.0  # Default no bias
                    
                    # Apply bias and random error based on MAPE
                    synthetic_actuals = []
                    for forecast in ml_forecasts[:self.validation_length]:
                        # Apply bias
                        biased_forecast = forecast * (1 + bias)
                        
                        # Apply random error based on MAPE
                        error_range = biased_forecast * mape
                        random_error = np.random.uniform(-error_range, error_range)
                        
                        # Final synthetic actual
                        synthetic_actual = max(0, biased_forecast + random_error)
                        synthetic_actuals.append(synthetic_actual)
                    
                    self.validation_data[sku] = np.array(synthetic_actuals)
        
        self.logger.info(f"Validation data prepared for {len(self.validation_data)} SKUs")
    
    def get_feature_dims(self) -> Tuple[int, int, int, int, int, int]:
        """
        Get dimensions of the state features.
        
        Returns:
            Tuple of (forecast_dim, error_metrics_dim, calendar_dim, holiday_dim, promo_dim, total_feature_dim)
        """
        forecast_dim = min(len(self.ml_cols), self.forecast_horizon)
        error_metrics_dim = len(self.accuracy_cols)
        
        # Calendar features: day of week (7), day of month (31), is weekend (1)
        calendar_dim = 3 * self.forecast_horizon
        
        # Holiday features: binary indicator for each day in horizon
        holiday_dim = self.forecast_horizon
        
        # Promotion features: binary indicator for each day in horizon
        promo_dim = self.forecast_horizon
        
        # Total dimensions
        total_dim = forecast_dim + error_metrics_dim + calendar_dim + holiday_dim + promo_dim
        
        return forecast_dim, error_metrics_dim, calendar_dim, holiday_dim, promo_dim, total_dim
    
    def _get_sku_state(self, sku: str, step: int) -> np.ndarray:
        """
        Get enhanced state representation for a specific SKU.
        
        Args:
            sku: SKU identifier
            step: Current step in the episode
            
        Returns:
            State features for the SKU
        """
        sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
        
        if sku_forecast.empty:
            # Default state if SKU not found
            forecast_dim, error_dim, calendar_dim, holiday_dim, promo_dim, total_dim = self.get_feature_dims()
            return np.zeros(total_dim)
        
        # 1. Extract forecasts for the next N days
        forecasts = []
        for i in range(self.forecast_horizon):
            day_idx = step + i
            if day_idx < len(self.ml_cols):
                ml_col = self.ml_cols[day_idx]
                
                if ml_col in sku_forecast.columns:
                    ml_value = sku_forecast[ml_col].iloc[0]
                    forecasts.append(ml_value)
                else:
                    forecasts.append(0.0)
            else:
                forecasts.append(0.0)
        
        # 2. Extract forecast accuracy metrics
        error_metrics = []
        for col in self.accuracy_cols:
            if col in sku_forecast.columns:
                error_metrics.append(sku_forecast[col].iloc[0])
            else:
                error_metrics.append(0.0)
        
        # 3. Generate calendar features
        calendar_features = self._generate_calendar_features(step)
        
        # Flatten calendar features for the horizon
        calendar_flat = []
        for i in range(self.forecast_horizon):
            day_idx = step + i
            if day_idx in calendar_features:
                features = calendar_features[day_idx]
                calendar_flat.extend([
                    features['day_of_week'],  # Normalized to [0,1]
                    features['day_of_month'],  # Normalized to [0,1]
                    features['is_weekend']
                ])
            else:
                # Default values if not available
                calendar_flat.extend([0.0, 0.0, 0.0])
        
        # 4. Generate holiday indicators
        holiday_indicators = []
        for i in range(self.forecast_horizon):
            day_idx = step + i
            is_holiday = float(self._check_if_holiday(day_idx))
            holiday_indicators.append(is_holiday)
        
        # 5. Generate promotion indicators
        promo_indicators = []
        for i in range(self.forecast_horizon):
            day_idx = step + i
            is_promo = float(self._check_if_promotion(sku, day_idx))
            promo_indicators.append(is_promo)
        
        # Combine all features
        state = np.concatenate([
            forecasts, 
            error_metrics, 
            calendar_flat,
            holiday_indicators,
            promo_indicators
        ]).astype(np.float32)
        
        return state
    
    def reset(self) -> List[np.ndarray]:
        """
        Reset the environment to the initial state.
        
        Returns:
            Initial state for all SKUs
        """
        self.current_step = 0
        self.done = False
        self.adjustment_history = []
        
        # Get initial state for all SKUs
        self.current_state = [self._get_sku_state(sku, self.current_step) for sku in self.skus]
        
        return self.current_state
    
    def step(self, actions: Dict[str, Tuple[int, float]]) -> Tuple[List[np.ndarray], Dict[str, float], bool, Dict]:
        """
        Take a step in the environment by applying forecast adjustments.
        
        Args:
            actions: Dictionary mapping SKU to (action_idx, adjusted_forecast)
            
        Returns:
            Tuple of (next_state, rewards, done, info)
        """
        rewards = {}
        info = {
            'original_mape': {},
            'adjusted_mape': {},
            'original_bias': {},
            'adjusted_bias': {},
            'is_holiday': {},
            'is_promotion': {}  # Add promotion info to output
        }
        
        # Process actions for each SKU
        for i, sku in enumerate(self.skus):
            if sku not in actions:
                rewards[sku] = 0.0
                continue
                
            action_idx, adjusted_forecast = actions[sku]
            
            # Get original forecast
            sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
            
            if not sku_forecast.empty and self.current_step < len(self.ml_cols):
                ml_col = self.ml_cols[self.current_step]
                
                if ml_col in sku_forecast.columns:
                    original_forecast = sku_forecast[ml_col].iloc[0]
                else:
                    original_forecast = 0.0
            else:
                original_forecast = 0.0
            
            # Add holiday and promotion information to info
            info['is_holiday'][sku] = self._check_if_holiday(self.current_step)
            info['is_promotion'][sku] = self._check_if_promotion(sku, self.current_step)
            
            # Calculate reward based on validation data (if available)
            if self.has_validation and sku in self.validation_data and self.current_step < len(self.validation_data[sku]):
                actual = self.validation_data[sku][self.current_step]
                
                # Calculate original error metrics
                if actual > 0:
                    original_mape = abs(original_forecast - actual) / actual
                    original_bias = (original_forecast - actual) / actual
                else:
                    original_mape = 1.0 if original_forecast > 0 else 0.0
                    original_bias = 1.0 if original_forecast > 0 else 0.0
                
                # Calculate adjusted error metrics
                if actual > 0:
                    adjusted_mape = abs(adjusted_forecast - actual) / actual
                    adjusted_bias = (adjusted_forecast - actual) / actual
                else:
                    adjusted_mape = 1.0 if adjusted_forecast > 0 else 0.0
                    adjusted_bias = 1.0 if adjusted_forecast > 0 else 0.0
                
                # Calculate reward based on improvement
                mape_improvement = original_mape - adjusted_mape
                bias_improvement = abs(original_bias) - abs(adjusted_bias)
                
                # Reward based on optimization target
                if self.optimize_for == "mape":
                    reward = mape_improvement * 10  # Scale for better learning
                elif self.optimize_for == "bias":
                    reward = bias_improvement * 10
                else:  # "both"
                    reward = (mape_improvement + bias_improvement) * 5
                
                # Store metrics in info
                info['original_mape'][sku] = float(original_mape)
                info['adjusted_mape'][sku] = float(adjusted_mape)
                info['original_bias'][sku] = float(original_bias)
                info['adjusted_bias'][sku] = float(adjusted_bias)
            else:
                # No validation data, use forecast accuracy metrics as proxy
                sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
                
                if not sku_forecast.empty and 'ml_mape_7d' in sku_forecast.columns:
                    mape = sku_forecast['ml_mape_7d'].iloc[0]
                    bias = sku_forecast['ml_bias_7d'].iloc[0] if 'ml_bias_7d' in sku_forecast.columns else 0.0
                    
                    # Use historical accuracy to estimate current error
                    est_original_mape = mape
                    est_original_bias = bias
                    
                    # Estimate adjustment impact based on action
                    adjustment_factor = adjusted_forecast / original_forecast if original_forecast > 0 else 1.0
                    
                    # Simple heuristic: if historical bias is positive (overforecast), 
                    # reducing forecast (factor < 1) might help
                    if bias > 0 and adjustment_factor < 1.0:
                        est_bias_improvement = bias * (1.0 - adjustment_factor)
                    # If historical bias is negative (underforecast), 
                    # increasing forecast (factor > 1) might help
                    elif bias < 0 and adjustment_factor > 1.0:
                        est_bias_improvement = abs(bias) * (adjustment_factor - 1.0)
                    else:
                        # Going in the wrong direction
                        est_bias_improvement = -abs(bias) * abs(adjustment_factor - 1.0)
                    
                    # For MAPE, smaller adjustments tend to be safer
                    est_mape_factor = abs(adjustment_factor - 1.0)
                    est_mape_improvement = -est_mape_factor * mape
                    
                    # Create a heuristic reward
                    if self.optimize_for == "mape":
                        reward = est_mape_improvement * 5
                    elif self.optimize_for == "bias":
                        reward = est_bias_improvement * 5
                    else:  # "both"
                        reward = (est_mape_improvement + est_bias_improvement) * 2.5
                    
                    # Populate info with estimates
                    info['original_mape'][sku] = float(est_original_mape)
                    info['adjusted_mape'][sku] = float(est_original_mape + est_mape_improvement)
                    info['original_bias'][sku] = float(est_original_bias)
                    info['adjusted_bias'][sku] = float(est_original_bias - est_bias_improvement)
                else:
                    # No forecast accuracy data
                    reward = 0.0
                    info['original_mape'][sku] = 0.0
                    info['adjusted_mape'][sku] = 0.0
                    info['original_bias'][sku] = 0.0
                    info['adjusted_bias'][sku] = 0.0
            
            rewards[sku] = reward
            
            # Track adjustment
            self.adjustment_history.append({
                'step': self.current_step,
                'sku': sku,
                'original_forecast': original_forecast,
                'adjusted_forecast': adjusted_forecast,
                'action_idx': action_idx,
                'reward': reward,
                'is_holiday': info['is_holiday'][sku],
                'is_promotion': info['is_promotion'][sku],
                'day_of_week': (self.start_date + pd.Timedelta(days=self.current_step)).weekday(),
                'day_of_month': (self.start_date + pd.Timedelta(days=self.current_step)).day
            })
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= min(len(self.ml_cols), self.validation_length):
            self.done = True
            
        # Get next state
        next_state = [self._get_sku_state(sku, self.current_step) for sku in self.skus]
        self.current_state = next_state
        
        return next_state, rewards, self.done, info
    
    def get_adjustment_history(self) -> pd.DataFrame:
        """
        Get the history of forecast adjustments.
        
        Returns:
            DataFrame of adjustment history
        """
        if not self.adjustment_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.adjustment_history)
    
    def calculate_accuracy_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate overall forecast accuracy metrics for original and adjusted forecasts.
        
        Returns:
            Dictionary of metrics
        """
        if not self.adjustment_history:
            return {}
        
        history_df = self.get_adjustment_history()
        metrics = {}
        
        # Calculate metrics per SKU
        for sku in self.skus:
            sku_history = history_df[history_df['sku'] == sku]
            if sku_history.empty:
                continue
            
            # Calculate MAPE and Bias if validation data exists
            if self.has_validation and sku in self.validation_data:
                sku_actuals = self.validation_data[sku][:len(sku_history)]
                
                # Filter out zero actuals for MAPE calculation
                non_zero_idx = np.where(sku_actuals > 0)[0]
                
                if len(non_zero_idx) > 0:
                    # Original forecasts
                    orig_forecasts = sku_history['original_forecast'].values
                    orig_non_zero = orig_forecasts[non_zero_idx]
                    actuals_non_zero = sku_actuals[non_zero_idx]
                    
                    orig_mape = np.mean(np.abs(orig_non_zero - actuals_non_zero) / actuals_non_zero)
                    orig_bias = np.mean((orig_non_zero - actuals_non_zero) / actuals_non_zero)
                    
                    # Adjusted forecasts
                    adj_forecasts = sku_history['adjusted_forecast'].values
                    adj_non_zero = adj_forecasts[non_zero_idx]
                    
                    adj_mape = np.mean(np.abs(adj_non_zero - actuals_non_zero) / actuals_non_zero)
                    adj_bias = np.mean((adj_non_zero - actuals_non_zero) / actuals_non_zero)
                    
                    # Calculate improvements
                    mape_improvement = (orig_mape - adj_mape) / orig_mape if orig_mape > 0 else 0
                    bias_improvement = (abs(orig_bias) - abs(adj_bias)) / abs(orig_bias) if orig_bias != 0 else 0
                    
                    # Split metrics by context
                    # Holiday metrics
                    holiday_idx = sku_history['is_holiday'].values.astype(bool)
                    if holiday_idx.any():
                        holiday_orig_mape = np.mean(np.abs(orig_forecasts[holiday_idx] - sku_actuals[holiday_idx]) / sku_actuals[holiday_idx])
                        holiday_adj_mape = np.mean(np.abs(adj_forecasts[holiday_idx] - sku_actuals[holiday_idx]) / sku_actuals[holiday_idx])
                        holiday_improvement = (holiday_orig_mape - holiday_adj_mape) / holiday_orig_mape if holiday_orig_mape > 0 else 0
                    else:
                        holiday_improvement = 0
                    
                    # Promotion metrics
                    promo_idx = sku_history['is_promotion'].values.astype(bool)
                    if promo_idx.any():
                        promo_orig_mape = np.mean(np.abs(orig_forecasts[promo_idx] - sku_actuals[promo_idx]) / sku_actuals[promo_idx])
                        promo_adj_mape = np.mean(np.abs(adj_forecasts[promo_idx] - sku_actuals[promo_idx]) / sku_actuals[promo_idx])
                        promo_improvement = (promo_orig_mape - promo_adj_mape) / promo_orig_mape if promo_orig_mape > 0 else 0
                    else:
                        promo_improvement = 0
                    
                    # Store all metrics
                    metrics[sku] = {
                        'original_mape': orig_mape,
                        'adjusted_mape': adj_mape,
                        'mape_improvement': mape_improvement,
                        'original_bias': orig_bias,
                        'adjusted_bias': adj_bias,
                        'bias_improvement': bias_improvement,
                        'holiday_improvement': holiday_improvement,
                        'promo_improvement': promo_improvement,
                        'sample_size': len(non_zero_idx)
                    }
        
        return metrics

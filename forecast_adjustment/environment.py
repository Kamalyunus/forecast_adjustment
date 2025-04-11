"""
Historical Forecast Training Environment - Environment for training forecast adjustment
by evaluating complete historical forecast horizons against actual data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta


class ForecastEnvironment:
    """
    Environment for training RL agents to adjust forecasts using historical forecasts
    and evaluating complete forecast horizons against actual data.
    """
    
    def __init__(self, 
                forecast_data: pd.DataFrame,
                actual_data: pd.DataFrame,  # Actual values for all dates
                holiday_data: Optional[pd.DataFrame] = None,
                promotion_data: Optional[pd.DataFrame] = None,
                forecast_horizon: int = 14,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the forecast adjustment environment.
        
        Args:
            forecast_data: DataFrame with forecast data (must have forecast_date and ml_day_X columns)
            actual_data: DataFrame with actual values (must have date and actual_value columns)
            holiday_data: Optional DataFrame with holiday information
            promotion_data: Optional DataFrame with promotion information
            forecast_horizon: Number of days in the forecast horizon
            optimize_for: Which metric to optimize for ("mape", "bias", or "both")
            start_date: Starting date for historical forecasts (format: 'YYYY-MM-DD')
            end_date: Ending date for historical forecasts (format: 'YYYY-MM-DD')
            logger: Optional logger instance
        """
        self.forecast_data = forecast_data
        self.actual_data = actual_data
        self.forecast_horizon = forecast_horizon
        self.optimize_for = optimize_for
        
        # Set up logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ForecastEnvironment")
        else:
            self.logger = logger
            
        # Verify required columns in forecast_data
        if 'forecast_date' not in self.forecast_data.columns or 'sku_id' not in self.forecast_data.columns:
            self.logger.error("Forecast data must contain 'forecast_date' and 'sku_id' columns")
            raise ValueError("Forecast data must contain 'forecast_date' and 'sku_id' columns")
        
        # Verify required columns in actual_data
        if 'date' not in self.actual_data.columns or 'sku_id' not in self.actual_data.columns or 'actual_value' not in self.actual_data.columns:
            self.logger.error("Actual data must contain 'date', 'sku_id', and 'actual_value' columns")
            raise ValueError("Actual data must contain 'date', 'sku_id', and 'actual_value' columns")
            
        # Convert date columns to datetime
        self.forecast_data['forecast_date'] = pd.to_datetime(self.forecast_data['forecast_date'])
        self.actual_data['date'] = pd.to_datetime(self.actual_data['date'])
        
        # Set date range for historical forecasts
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            # Default to earliest forecast date
            self.start_date = self.forecast_data['forecast_date'].min()
            
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            # Default to latest forecast date that has complete actuals
            max_actual_date = self.actual_data['date'].max()
            # Latest forecast date that has complete actuals for all days in horizon
            self.end_date = max_actual_date - pd.Timedelta(days=self.forecast_horizon-1)
        
        # Filter forecast_data to only include dates within range
        self.forecast_data = self.forecast_data[
            (self.forecast_data['forecast_date'] >= self.start_date) & 
            (self.forecast_data['forecast_date'] <= self.end_date)
        ]
        
        # Extract unique forecast dates in ascending order
        self.forecast_dates = sorted(self.forecast_data['forecast_date'].unique())
        
        if len(self.forecast_dates) == 0:
            self.logger.error(f"No forecasts found in date range {self.start_date} to {self.end_date}")
            raise ValueError(f"No forecasts found in date range {self.start_date} to {self.end_date}")
            
        self.logger.info(f"Environment initialized with {len(self.forecast_dates)} forecast generation dates")
        
        # Extract SKUs
        self.skus = self.forecast_data['sku_id'].unique().tolist()
        self.logger.info(f"Environment initialized with {len(self.skus)} SKUs")
        
        # Extract ML forecast columns
        self.ml_cols = [col for col in self.forecast_data.columns if col.startswith('ml_day_')]
        if len(self.ml_cols) < self.forecast_horizon:
            self.logger.warning(f"Only found {len(self.ml_cols)} forecast day columns, but horizon is {self.forecast_horizon}")
        
        # Extract accuracy metrics
        self.accuracy_cols = [col for col in self.forecast_data.columns if col.startswith('ml_mape') or col.startswith('ml_bias')]
        
        # Process holiday data if provided
        self.holiday_calendar = None
        if holiday_data is not None:
            self._setup_holiday_calendar(holiday_data)
        
        # Process promotion data if provided
        self.promotion_schedule = None
        if promotion_data is not None:
            self._setup_promotion_schedule(promotion_data)
        
        # Create actual value lookup dictionary for fast access
        self._setup_actual_values()
        
        # Current state of the environment
        self.current_forecast_idx = 0  # Index in self.forecast_dates
        self.current_horizon_day = 0  # Which day in the forecast horizon (0-based)
        self.current_state = None
        self.done = False
        
        # History tracking
        self.adjustment_history = []
    
    def _setup_actual_values(self):
        """
        Create a lookup dictionary for actual values.
        
        Structure: {sku_id: {date: actual_value}}
        """
        self.actual_values = {}
        
        for _, row in self.actual_data.iterrows():
            sku = row['sku_id']
            date = row['date']
            value = row['actual_value']
            
            if sku not in self.actual_values:
                self.actual_values[sku] = {}
                
            self.actual_values[sku][date] = value
        
        self.logger.info(f"Actual values prepared for {len(self.actual_values)} SKUs")
    
    def get_feature_dims(self) -> Tuple[int, int, int, int, int, int, int]:
        """
        Get dimensions of the state features.
        
        Returns:
            Tuple of (forecast_dim, error_metrics_dim, calendar_dim, holiday_dim, 
                     promo_dim, horizon_position_dim, total_feature_dim)
        """
        forecast_dim = min(len(self.ml_cols), self.forecast_horizon)
        error_metrics_dim = len(self.accuracy_cols)
        
        # Calendar features: day of week (7), day of month (31), is weekend (1)
        calendar_dim = 3 * self.forecast_horizon
        
        # Holiday features: binary indicator for each day in horizon
        holiday_dim = self.forecast_horizon
        
        # Promotion features: binary indicator for each day in horizon
        promo_dim = self.forecast_horizon
        
        # Horizon position indicator: one-hot encoding of which day in the horizon
        horizon_position_dim = self.forecast_horizon
        
        # Total dimensions
        total_dim = forecast_dim + error_metrics_dim + calendar_dim + holiday_dim + promo_dim + horizon_position_dim
        
        return forecast_dim, error_metrics_dim, calendar_dim, holiday_dim, promo_dim, horizon_position_dim, total_dim
    
    def _get_sku_state(self, sku: str, forecast_date: pd.Timestamp, horizon_day: int) -> np.ndarray:
        """
        Get state representation for a specific SKU, forecast date, and horizon day.
        
        Args:
            sku: SKU identifier
            forecast_date: Date when the forecast was generated
            horizon_day: Which day in the forecast horizon (0-based)
            
        Returns:
            State features for the SKU
        """
        # Get forecasts for this SKU at the given forecast date
        sku_forecasts = self.forecast_data[
            (self.forecast_data['sku_id'] == sku) & 
            (self.forecast_data['forecast_date'] == forecast_date)
        ]
        
        if sku_forecasts.empty:
            # Default state if SKU not found
            forecast_dim, error_dim, calendar_dim, holiday_dim, promo_dim, horizon_dim, total_dim = self.get_feature_dims()
            return np.zeros(total_dim)
        
        # Use the first matching row
        sku_forecast = sku_forecasts.iloc[0]
        
        # 1. Extract forecasts for the next N days (typically include all days in horizon)
        forecasts = []
        for i in range(self.forecast_horizon):
            day_idx = i + 1
            if day_idx <= len(self.ml_cols):
                ml_col = f'ml_day_{day_idx}'
                
                if ml_col in sku_forecast:
                    ml_value = sku_forecast[ml_col]
                    forecasts.append(ml_value)
                else:
                    forecasts.append(0.0)
            else:
                forecasts.append(0.0)
        
        # 2. Extract forecast accuracy metrics
        error_metrics = []
        for col in self.accuracy_cols:
            if col in sku_forecast:
                error_metrics.append(sku_forecast[col])
            else:
                error_metrics.append(0.0)
        
        # 3. Generate calendar features for the forecast horizon
        calendar_features = self._generate_calendar_features(forecast_date)
        
        # Flatten calendar features for the horizon
        calendar_flat = []
        for i in range(self.forecast_horizon):
            target_date = forecast_date + pd.Timedelta(days=i+1)
            if target_date in calendar_features:
                features = calendar_features[target_date]
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
            target_date = forecast_date + pd.Timedelta(days=i+1)
            is_holiday = float(self._check_if_holiday(target_date))
            holiday_indicators.append(is_holiday)
        
        # 5. Generate promotion indicators
        promo_indicators = []
        for i in range(self.forecast_horizon):
            target_date = forecast_date + pd.Timedelta(days=i+1)
            is_promo = float(self._check_if_promotion(sku, target_date))
            promo_indicators.append(is_promo)
        
        # 6. Add horizon position indicator (one-hot encoding)
        horizon_position = np.zeros(self.forecast_horizon)
        if 0 <= horizon_day < self.forecast_horizon:
            horizon_position[horizon_day] = 1.0
        
        # Combine all features
        state = np.concatenate([
            forecasts, 
            error_metrics, 
            calendar_flat,
            holiday_indicators,
            promo_indicators,
            horizon_position
        ]).astype(np.float32)
        
        return state
    
    def _generate_calendar_features(self, start_date: pd.Timestamp) -> Dict[pd.Timestamp, Dict[str, float]]:
        """
        Generate calendar features for the forecast horizon.
        
        Args:
            start_date: Starting date for the forecast
            
        Returns:
            Dictionary mapping dates to calendar features
        """
        calendar_features = {}
        
        for i in range(self.forecast_horizon):
            target_date = start_date + pd.Timedelta(days=i+1)
            
            # Extract calendar features
            day_of_week = target_date.weekday()  # 0=Monday, 6=Sunday
            day_of_month = target_date.day
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
            
            # Store features
            calendar_features[target_date] = {
                'day_of_week': day_of_week / 6.0,  # Normalize to [0,1]
                'day_of_month': day_of_month / 31.0,  # Normalize to [0,1]
                'is_weekend': is_weekend
            }
        
        return calendar_features
    
    def _check_if_holiday(self, date: pd.Timestamp) -> bool:
        """
        Check if a given date is a holiday.
        
        Args:
            date: Date to check
            
        Returns:
            True if the date is a holiday
        """
        if self.holiday_calendar and date in self.holiday_calendar:
            return self.holiday_calendar[date]
        return False
    
    def _check_if_promotion(self, sku: str, date: pd.Timestamp) -> bool:
        """
        Check if a given SKU is on promotion for a specific date.
        
        Args:
            sku: SKU identifier
            date: Date to check
            
        Returns:
            True if the SKU is on promotion
        """
        if self.promotion_schedule:
            key = (sku, date)
            return key in self.promotion_schedule
        return False
    
    def _setup_holiday_calendar(self, holiday_data: pd.DataFrame):
        """
        Process holiday data into a usable format.
        
        Args:
            holiday_data: DataFrame with columns 'date' and 'holiday_name'
        """
        self.logger.info("Setting up holiday calendar")
        
        # Initialize holiday calendar
        self.holiday_calendar = {}
        self.holiday_names = {}
        
        # Process holiday data
        try:
            for _, row in holiday_data.iterrows():
                holiday_date = pd.to_datetime(row['date'])
                holiday_name = row['holiday_name']
                
                self.holiday_calendar[holiday_date] = True
                self.holiday_names[holiday_date] = holiday_name
            
            self.logger.info(f"Processed {len(self.holiday_calendar)} holidays")
        except Exception as e:
            self.logger.warning(f"Error processing holiday data: {str(e)}")
    
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
                
                # Store promotion information for each date in the range
                current_date = start_date
                while current_date <= end_date:
                    key = (sku, current_date)
                    self.promotion_schedule[key] = True
                    self.promotion_types[key] = promo_type
                    current_date += pd.Timedelta(days=1)
            
            self.logger.info(f"Processed {len(self.promotion_schedule)} promotion days")
        except Exception as e:
            self.logger.warning(f"Error processing promotion data: {str(e)}")
    
    def reset(self) -> List[np.ndarray]:
        """
        Reset the environment to the initial state.
        
        Returns:
            Initial state for all SKUs
        """
        self.current_forecast_idx = 0
        self.current_horizon_day = 0
        self.done = False
        self.adjustment_history = []
        
        # Get current forecast date
        forecast_date = self.forecast_dates[self.current_forecast_idx]
        
        # Get initial state for all SKUs for day 0 in horizon
        self.current_state = [self._get_sku_state(sku, forecast_date, self.current_horizon_day) for sku in self.skus]
        
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
            'is_promotion': {},
            'forecast_date': {},
            'horizon_day': {},
            'target_date': {},
            'has_actual': {}
        }
        
        # Get current forecast date
        forecast_date = self.forecast_dates[self.current_forecast_idx]
        
        # Calculate target date (forecast date + horizon day + 1)
        # Horizon day is 0-based, but forecast is for 1+ days ahead
        target_date = forecast_date + pd.Timedelta(days=self.current_horizon_day + 1)
        
        # Process actions for each SKU
        for i, sku in enumerate(self.skus):
            if sku not in actions:
                rewards[sku] = 0.0
                continue
                
            action_idx, adjusted_forecast = actions[sku]
            
            # Get forecasts for this SKU at the current forecast date
            sku_forecasts = self.forecast_data[
                (self.forecast_data['sku_id'] == sku) & 
                (self.forecast_data['forecast_date'] == forecast_date)
            ]
            
            original_forecast = 0.0
            if not sku_forecasts.empty:
                # Get the first matching forecast
                sku_forecast = sku_forecasts.iloc[0]
                
                # Get original forecast for the current horizon day (1-based)
                ml_col = f'ml_day_{self.current_horizon_day + 1}'
                if ml_col in sku_forecast:
                    original_forecast = sku_forecast[ml_col]
            
            # Add context information to info
            info['is_holiday'][sku] = self._check_if_holiday(target_date)
            info['is_promotion'][sku] = self._check_if_promotion(sku, target_date)
            info['forecast_date'][sku] = forecast_date
            info['horizon_day'][sku] = self.current_horizon_day
            info['target_date'][sku] = target_date
            
            # Check if we have actual data for this target date and SKU
            has_actual = False
            actual_value = 0.0
            
            if sku in self.actual_values and target_date in self.actual_values[sku]:
                has_actual = True
                actual_value = self.actual_values[sku][target_date]
                
            info['has_actual'][sku] = has_actual
            
            # Calculate reward based on actual data (if available)
            if has_actual and actual_value > 0:
                # Calculate original error metrics
                original_mape = abs(original_forecast - actual_value) / actual_value
                original_bias = (original_forecast - actual_value) / actual_value
                
                # Calculate adjusted error metrics
                adjusted_mape = abs(adjusted_forecast - actual_value) / actual_value
                adjusted_bias = (adjusted_forecast - actual_value) / actual_value
                
                # Calculate reward based on improvement
                mape_improvement = original_mape - adjusted_mape
                bias_improvement = abs(original_bias) - abs(adjusted_bias)
                
                # Consider horizon day in reward calculation
                # More weight for near-term horizon days, less for further out
                horizon_factor = 1.0 - 0.5 * (self.current_horizon_day / self.forecast_horizon)
                
                # Rewards scaled by horizon factor - more important to get near-term days right
                mape_reward = mape_improvement * horizon_factor * 1.0
                bias_reward = bias_improvement * horizon_factor * 1.0
                
                # Reward based on optimization target
                if self.optimize_for == "mape":
                    reward = mape_reward
                elif self.optimize_for == "bias":
                    reward = bias_reward
                else:  # "both"
                    reward = (mape_reward + bias_reward) / 2
                
                # Store metrics in info
                info['original_mape'][sku] = float(original_mape)
                info['adjusted_mape'][sku] = float(adjusted_mape)
                info['original_bias'][sku] = float(original_bias)
                info['adjusted_bias'][sku] = float(adjusted_bias)
            else:
                # No actual data available, use forecast accuracy metrics as proxy
                if not sku_forecasts.empty:
                    sku_forecast = sku_forecasts.iloc[0]
                    
                    # Use different metrics based on horizon day
                    if self.current_horizon_day < 7:
                        mape_col = 'ml_mape_7d'
                        bias_col = 'ml_bias_7d'
                    else:
                        mape_col = 'ml_mape_30d'
                        bias_col = 'ml_bias_30d'
                    
                    mape = sku_forecast[mape_col] if mape_col in sku_forecast else 0.2
                    bias = sku_forecast[bias_col] if bias_col in sku_forecast else 0.0
                    
                    # Use historical accuracy to estimate current error
                    est_original_mape = mape
                    est_original_bias = bias
                    
                    # Estimate adjustment impact based on action
                    adjustment_factor = adjusted_forecast / original_forecast if original_forecast > 0 else 1.0
                    
                    # Consider horizon day in heuristic calculation
                    horizon_factor = self.current_horizon_day / self.forecast_horizon
                    
                    # Different adjustment strategies based on horizon
                    # Near-term forecasts (days 0-3): Usually need small adjustments
                    # Mid-term forecasts (days 4-9): May need moderate adjustments
                    # Long-term forecasts (days 10-13): May need larger adjustments
                    
                    if self.current_horizon_day < 4:
                        # Near-term: reward small adjustments, penalize large ones
                        adj_magnitude = abs(adjustment_factor - 1.0)
                        if adj_magnitude < 0.1:
                            est_mape_improvement = 0.02 * mape  # Small improvement
                        else:
                            est_mape_improvement = -0.05 * mape * adj_magnitude  # Penalty for large adjustment
                    elif self.current_horizon_day < 10:
                        # Mid-term: moderate adjustments might help
                        # For bias correction, reward adjustments in the right direction
                        if bias > 0 and adjustment_factor < 1.0:
                            # Reducing forecast when historically overforecasting
                            est_bias_improvement = bias * (1.0 - adjustment_factor) * 0.5
                        elif bias < 0 and adjustment_factor > 1.0:
                            # Increasing forecast when historically underforecasting
                            est_bias_improvement = abs(bias) * (adjustment_factor - 1.0) * 0.5
                        else:
                            # Going in wrong direction
                            est_bias_improvement = -abs(bias) * abs(adjustment_factor - 1.0) * 0.3
                        
                        est_mape_improvement = est_bias_improvement * 0.5  # MAPE improvement correlates with bias
                    else:
                        # Long-term: larger adjustments might be appropriate
                        # Reward appropriate directional adjustments based on historical bias
                        if bias > 0 and adjustment_factor < 1.0:
                            # Reducing forecast when historically overforecasting
                            est_bias_improvement = bias * (1.0 - adjustment_factor) * 0.8
                        elif bias < 0 and adjustment_factor > 1.0:
                            # Increasing forecast when historically underforecasting
                            est_bias_improvement = abs(bias) * (adjustment_factor - 1.0) * 0.8
                        else:
                            # Going in wrong direction but less penalty for long-term
                            est_bias_improvement = -abs(bias) * abs(adjustment_factor - 1.0) * 0.2
                        
                        est_mape_improvement = est_bias_improvement * 0.7  # MAPE improvement correlates with bias
                    
                    # Create a heuristic reward with horizon weighting
                    horizon_weight = 1.0 - 0.5 * horizon_factor  # More weight for near-term days
                    
                    if self.optimize_for == "mape":
                        reward = est_mape_improvement * 5 * horizon_weight
                    elif self.optimize_for == "bias":
                        reward = est_bias_improvement * 5 * horizon_weight
                    else:  # "both"
                        reward = (est_mape_improvement + est_bias_improvement) * 2.5 * horizon_weight
                    
                    # Populate info with estimates
                    info['original_mape'][sku] = float(est_original_mape)
                    info['adjusted_mape'][sku] = float(est_original_mape - est_mape_improvement)
                    info['original_bias'][sku] = float(est_original_bias)
                    info['adjusted_bias'][sku] = float(est_original_bias - est_bias_improvement)
                else:
                    # No forecast data
                    reward = 0.0
                    info['original_mape'][sku] = 0.0
                    info['adjusted_mape'][sku] = 0.0
                    info['original_bias'][sku] = 0.0
                    info['adjusted_bias'][sku] = 0.0
            
            rewards[sku] = reward
            
            # Track adjustment
            self.adjustment_history.append({
                'forecast_date': forecast_date,
                'target_date': target_date,
                'sku': sku,
                'horizon_day': self.current_horizon_day,
                'original_forecast': original_forecast,
                'adjusted_forecast': adjusted_forecast,
                'action_idx': action_idx,
                'reward': reward,
                'has_actual': has_actual,
                'actual_value': actual_value if has_actual else None,
                'is_holiday': info['is_holiday'][sku],
                'is_promotion': info['is_promotion'][sku],
                'day_of_week': target_date.weekday(),
                'day_of_month': target_date.day
            })
        
        # Increment horizon day
        self.current_horizon_day += 1
        
        # If we've processed all days in the horizon, move to next forecast date
        if self.current_horizon_day >= self.forecast_horizon:
            self.current_horizon_day = 0
            self.current_forecast_idx += 1
            
        # Check if episode is done
        if self.current_forecast_idx >= len(self.forecast_dates):
            self.done = True
            
        # Get next state
        if not self.done:
            forecast_date = self.forecast_dates[self.current_forecast_idx]
            next_state = [self._get_sku_state(sku, forecast_date, self.current_horizon_day) for sku in self.skus]
            self.current_state = next_state
        else:
            next_state = self.current_state
        
        return next_state, rewards, self.done, info
    
    def get_adjustment_history(self) -> List[Dict]:
        """Get the history of forecast adjustments."""
        return self.adjustment_history
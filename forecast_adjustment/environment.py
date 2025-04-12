"""
Enhanced Forecast Training Environment - Environment for training forecast adjustment
with SKU banding (A-E) for better inventory management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta


class ForecastEnvironment:
    """
    Environment for training RL agents to adjust forecasts using historical forecasts
    and evaluating complete forecast horizons against actual data, with support for SKU banding.
    """
    
    def __init__(self, 
                forecast_data: pd.DataFrame,
                actual_data: pd.DataFrame,
                sku_band_data: Optional[pd.DataFrame] = None,  # New: SKU banding information
                holiday_data: Optional[pd.DataFrame] = None,
                promotion_data: Optional[pd.DataFrame] = None,
                forecast_horizon: int = 14,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                reward_scaling: float = 5.0,
                pattern_emphasis: float = 1.5,
                band_emphasis: float = 1.8,  # New: Emphasis for band-specific rewards
                logger: Optional[logging.Logger] = None):
        """
        Initialize the forecast adjustment environment with SKU banding support.
        
        Args:
            forecast_data: DataFrame with forecast data (must have forecast_date and ml_day_X columns)
            actual_data: DataFrame with actual values (must have date and actual_value columns)
            sku_band_data: Optional DataFrame with SKU band information (A-E)
            holiday_data: Optional DataFrame with holiday information
            promotion_data: Optional DataFrame with promotion information
            forecast_horizon: Number of days in the forecast horizon
            optimize_for: Which metric to optimize for ("mape", "bias", or "both")
            start_date: Starting date for historical forecasts (format: 'YYYY-MM-DD')
            end_date: Ending date for historical forecasts (format: 'YYYY-MM-DD')
            reward_scaling: Factor to scale rewards (higher = stronger signal)
            pattern_emphasis: Factor to emphasize pattern-specific rewards
            band_emphasis: Factor to emphasize band-specific reward scaling
            logger: Optional logger instance
        """
        self.forecast_data = forecast_data
        self.actual_data = actual_data
        self.forecast_horizon = forecast_horizon
        self.optimize_for = optimize_for
        self.reward_scaling = reward_scaling
        self.pattern_emphasis = pattern_emphasis
        self.band_emphasis = band_emphasis
        
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
        
        # Extract SKUs
        self.skus = self.forecast_data['sku_id'].unique().tolist()
        self.logger.info(f"Environment initialized with {len(self.skus)} SKUs")

        # Process SKU band data if provided
        self.sku_bands = {}
        if sku_band_data is not None:
            self._setup_sku_bands(sku_band_data)
        
        # Check for pattern_type column in actual_data - useful for specialized learning
        self.has_pattern_types = 'pattern_type' in self.actual_data.columns
        if self.has_pattern_types:
            self.logger.info("Found pattern_type column in actual data, will use for specialized learning")
            # Extract pattern types for each SKU
            self.sku_patterns = {}
            for _, row in self.actual_data[['sku_id', 'pattern_type']].drop_duplicates().iterrows():
                self.sku_patterns[row['sku_id']] = row['pattern_type']
            
            # Print pattern type counts
            pattern_counts = self.actual_data['pattern_type'].value_counts()
            for pattern, count in pattern_counts.items():
                self.logger.info(f"Pattern '{pattern}': {count} data points")
        
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
        
        # Extract ML forecast columns
        self.ml_cols = [col for col in self.forecast_data.columns if col.startswith('ml_day_')]
        if len(self.ml_cols) < self.forecast_horizon:
            self.logger.warning(f"Only found {len(self.ml_cols)} forecast day columns, but horizon is {self.forecast_horizon}")
        
        # Extract accuracy metrics
        self.accuracy_cols = [col for col in self.forecast_data.columns if col.startswith('ml_mape') or col.startswith('ml_bias')]
        
        # Process holiday data if provided
        self.holiday_calendar = {}
        if holiday_data is not None:
            self._setup_holiday_calendar(holiday_data)
        
        # Process promotion data if provided
        self.promotion_schedule = {}
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
        
        # Performance tracking for each pattern type
        self.pattern_performance = {}
        if self.has_pattern_types:
            unique_patterns = set(self.sku_patterns.values())
            for pattern in unique_patterns:
                self.pattern_performance[pattern] = {
                    'original_mape': [],
                    'adjusted_mape': [],
                    'original_bias': [],
                    'adjusted_bias': []
                }
        
        # NEW: Performance tracking for each band
        self.band_performance = {
            'A': {'original_mape': [], 'adjusted_mape': [], 'original_bias': [], 'adjusted_bias': []},
            'B': {'original_mape': [], 'adjusted_mape': [], 'original_bias': [], 'adjusted_bias': []},
            'C': {'original_mape': [], 'adjusted_mape': [], 'original_bias': [], 'adjusted_bias': []},
            'D': {'original_mape': [], 'adjusted_mape': [], 'original_bias': [], 'adjusted_bias': []},
            'E': {'original_mape': [], 'adjusted_mape': [], 'original_bias': [], 'adjusted_bias': []}
        }
        
        # Track day-of-week performance 
        self.dow_performance = {i: {'original_mape': [], 'adjusted_mape': [], 'count': 0} for i in range(7)}
        
        # Track the performance baseline to detect improvements
        self.baseline_mape = None
        self.baseline_bias = None
        self.total_improvement = 0.0
    
    def _setup_sku_bands(self, sku_band_data: pd.DataFrame):
        """
        Process SKU band data into a usable format.
        
        Args:
            sku_band_data: DataFrame with columns 'sku_id' and 'band'
        """
        self.logger.info("Setting up SKU band information")
        
        # Extract required columns
        if 'sku_id' not in sku_band_data.columns or 'band' not in sku_band_data.columns:
            self.logger.error("SKU band data must contain 'sku_id' and 'band' columns")
            raise ValueError("SKU band data must contain 'sku_id' and 'band' columns")
        
        # Store band information for each SKU
        for _, row in sku_band_data.iterrows():
            sku_id = row['sku_id']
            band = row['band']
            
            if band not in ['A', 'B', 'C', 'D', 'E']:
                self.logger.warning(f"Unknown band '{band}' for SKU {sku_id}, defaulting to 'C'")
                band = 'C'
                
            self.sku_bands[sku_id] = band
        
        # For SKUs without band information, default to 'C'
        for sku in self.skus:
            if sku not in self.sku_bands:
                self.sku_bands[sku] = 'C'
                
        # Count SKUs in each band
        band_counts = {band: 0 for band in ['A', 'B', 'C', 'D', 'E']}
        for band in self.sku_bands.values():
            band_counts[band] += 1
            
        for band, count in band_counts.items():
            self.logger.info(f"Band {band}: {count} SKUs")
    
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
    
    def _setup_holiday_calendar(self, holiday_data: pd.DataFrame):
        """
        Process holiday data into a usable format.
        
        Args:
            holiday_data: DataFrame with columns 'date' and 'holiday_name'
        """
        self.logger.info("Setting up holiday calendar")
        
        # Process holiday data
        try:
            for _, row in holiday_data.iterrows():
                holiday_date = pd.to_datetime(row['date'])
                holiday_name = row['holiday_name']
                
                self.holiday_calendar[holiday_date] = holiday_name
            
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
                    self.promotion_schedule[key] = promo_type
                    current_date += pd.Timedelta(days=1)
            
            self.logger.info(f"Processed {len(self.promotion_schedule)} promotion days")
        except Exception as e:
            self.logger.warning(f"Error processing promotion data: {str(e)}")
    
    def get_feature_dims(self) -> Tuple[int, int, int, int, int, int, int, int]:
        """
        Get dimensions of the state features.
        
        Returns:
            Tuple of (forecast_dim, error_metrics_dim, calendar_dim, holiday_dim, 
                     promo_dim, horizon_position_dim, band_dim, total_feature_dim)
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
        
        # NEW: Band features: one-hot encoding of band (A-E)
        band_dim = 5
        
        # Total dimensions
        total_dim = forecast_dim + error_metrics_dim + calendar_dim + holiday_dim + promo_dim + horizon_position_dim + band_dim
        
        return forecast_dim, error_metrics_dim, calendar_dim, holiday_dim, promo_dim, horizon_position_dim, band_dim, total_dim
    
    def _get_pattern_type(self, sku: str) -> str:
        """Get the pattern type for a specific SKU."""
        if self.has_pattern_types and sku in self.sku_patterns:
            return self.sku_patterns[sku]
        return "unknown"
    
    def _get_sku_band(self, sku: str) -> str:
        """Get the band (A-E) for a specific SKU."""
        return self.sku_bands.get(sku, 'C')  # Default to 'C' if unknown
    
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
            forecast_dim, error_dim, calendar_dim, holiday_dim, promo_dim, horizon_dim, band_dim, total_dim = self.get_feature_dims()
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
        
        # 7. NEW: Add band indicator (one-hot encoding)
        band = self._get_sku_band(sku)
        band_indicator = np.zeros(5)  # A-E: 5 bands
        band_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}.get(band, 2)  # Default to C (index 2)
        band_indicator[band_index] = 1.0
        
        # Combine all features
        state = np.concatenate([
            forecasts, 
            error_metrics, 
            calendar_flat,
            holiday_indicators,
            promo_indicators,
            horizon_position,
            band_indicator
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
        return date in self.holiday_calendar
    
    def _check_if_promotion(self, sku: str, date: pd.Timestamp) -> bool:
        """
        Check if a given SKU is on promotion for a specific date.
        
        Args:
            sku: SKU identifier
            date: Date to check
            
        Returns:
            True if the SKU is on promotion
        """
        key = (sku, date)
        return key in self.promotion_schedule
    
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
        
        # Reset performance tracking
        if self.has_pattern_types:
            for pattern in self.pattern_performance:
                self.pattern_performance[pattern] = {
                    'original_mape': [],
                    'adjusted_mape': [],
                    'original_bias': [],
                    'adjusted_bias': []
                }
        
        # Reset band performance tracking
        for band in self.band_performance:
            self.band_performance[band] = {
                'original_mape': [],
                'adjusted_mape': [],
                'original_bias': [],
                'adjusted_bias': []
            }
        
        for i in range(7):
            self.dow_performance[i] = {'original_mape': [], 'adjusted_mape': [], 'count': 0}
        
        self.baseline_mape = None
        self.baseline_bias = None
        self.total_improvement = 0.0
        
        # Capture current step for easier access
        self.current_step = 1
        
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
            'has_actual': {},
            'pattern_type': {},
            'sku_band': {}  # NEW: Track SKU band info
        }
        
        # Get current forecast date
        forecast_date = self.forecast_dates[self.current_forecast_idx]
        
        # Calculate target date (forecast date + horizon day + 1)
        # Horizon day is 0-based, but forecast is for 1+ days ahead
        target_date = forecast_date + pd.Timedelta(days=self.current_horizon_day + 1)
        
        # Track sum of original/adjusted errors for baseline calculation
        sum_original_mape = 0.0
        sum_adjusted_mape = 0.0
        sum_original_bias = 0.0
        sum_adjusted_bias = 0.0
        count_with_actuals = 0
        
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
            
            # Get SKU band and pattern type
            sku_band = self._get_sku_band(sku)
            pattern_type = self._get_pattern_type(sku)
            
            # Add context information to info
            is_holiday = self._check_if_holiday(target_date)
            is_promotion = self._check_if_promotion(sku, target_date)
            
            info['is_holiday'][sku] = is_holiday
            info['is_promotion'][sku] = is_promotion
            info['forecast_date'][sku] = forecast_date
            info['horizon_day'][sku] = self.current_horizon_day
            info['target_date'][sku] = target_date
            info['pattern_type'][sku] = pattern_type
            info['sku_band'][sku] = sku_band
            
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
                
                # Store metrics in info
                info['original_mape'][sku] = float(original_mape)
                info['adjusted_mape'][sku] = float(adjusted_mape)
                info['original_bias'][sku] = float(original_bias)
                info['adjusted_bias'][sku] = float(adjusted_bias)
                
                # Track for baseline
                sum_original_mape += original_mape
                sum_adjusted_mape += adjusted_mape
                sum_original_bias += abs(original_bias)
                sum_adjusted_bias += abs(adjusted_bias)
                count_with_actuals += 1
                
                # Calculate improvements
                mape_improvement = original_mape - adjusted_mape
                bias_improvement = abs(original_bias) - abs(adjusted_bias)
                
                # Apply SKU band-specific emphasis factors for rewards
                band_factor = 1.0
                if sku_band in ['A', 'B']:  # Fast-selling items
                    # For high-volume SKUs, we care about both accuracy and bias
                    band_factor = self.band_emphasis
                    
                    # Even more penalty for underbias in high-volume SKUs
                    # (to avoid stockouts on popular items)
                    if original_bias < 0 and adjusted_bias < 0 and abs(adjusted_bias) > abs(original_bias):
                        band_factor *= 1.5  # Extra penalty for increasing underbias on fast-moving items
                        
                elif sku_band in ['D', 'E']:  # Slow-selling items
                    # For low-volume SKUs, we care more about avoiding overbias
                    band_factor = self.band_emphasis * 0.8
                    
                    # Stronger penalty for overbias on slow-moving items
                    # (to prevent excess inventory)
                    if original_bias > 0 and adjusted_bias > 0 and adjusted_bias > original_bias:
                        band_factor *= 1.5  # Extra penalty for increasing overbias on slow-moving items
                        
                # Apply pattern-specific emphasis factors
                pattern_factor = 1.0
                
                if pattern_type == "promo_holiday" and (is_holiday or is_promotion):
                    pattern_factor = self.pattern_emphasis
                elif pattern_type == "day_pattern" and target_date.weekday() >= 5:
                    pattern_factor = self.pattern_emphasis
                elif pattern_type == "underbias":
                    pattern_factor = self.pattern_emphasis * 0.8
                
                # Consider horizon day in reward calculation
                horizon_factor = 1.0 - 0.5 * (self.current_horizon_day / self.forecast_horizon)
                
                # Calculate rewards based on improvements
                mape_reward = mape_improvement * self.reward_scaling * pattern_factor * horizon_factor * band_factor
                bias_reward = bias_improvement * self.reward_scaling * pattern_factor * horizon_factor * band_factor
                
                # Positive reinforcement: larger rewards for improvements, smaller penalties for degradation
                if mape_improvement > 0:
                    mape_reward *= 1.5  # Boost positive rewards to encourage improvement
                
                # Reward based on optimization target
                if self.optimize_for == "mape":
                    reward = mape_reward
                elif self.optimize_for == "bias":
                    reward = bias_reward
                else:  # "both"
                    reward = (mape_reward + bias_reward) / 2
                
                # Track performance by pattern type
                if self.has_pattern_types and pattern_type in self.pattern_performance:
                    self.pattern_performance[pattern_type]['original_mape'].append(original_mape)
                    self.pattern_performance[pattern_type]['adjusted_mape'].append(adjusted_mape)
                    self.pattern_performance[pattern_type]['original_bias'].append(original_bias)
                    self.pattern_performance[pattern_type]['adjusted_bias'].append(adjusted_bias)
                
                # Track performance by band
                if sku_band in self.band_performance:
                    self.band_performance[sku_band]['original_mape'].append(original_mape)
                    self.band_performance[sku_band]['adjusted_mape'].append(adjusted_mape)
                    self.band_performance[sku_band]['original_bias'].append(original_bias)
                    self.band_performance[sku_band]['adjusted_bias'].append(adjusted_bias)
                
                # Track day-of-week performance
                dow = target_date.weekday()
                self.dow_performance[dow]['original_mape'].append(original_mape)
                self.dow_performance[dow]['adjusted_mape'].append(adjusted_mape)
                self.dow_performance[dow]['count'] += 1
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
                    horizon_factor = 1.0 - 0.5 * (self.current_horizon_day / self.forecast_horizon)
                    
                    # Different adjustment strategies based on pattern type, context, and SKU band
                    est_mape_improvement = 0.0
                    est_bias_improvement = 0.0
                    
                    # NEW: Band-specific heuristics
                    if sku_band in ['A', 'B']:  # High-volume, fast-selling items
                        if is_holiday or is_promotion:
                            # For high-volume items during events, reward upward adjustments (avoid stockouts)
                            if adjustment_factor > 1.1:
                                est_mape_improvement = 0.05 * adjustment_factor
                                est_bias_improvement = 0.08 * adjustment_factor
                            else:
                                est_mape_improvement = -0.03
                                est_bias_improvement = -0.04
                        else:
                            # For regular days, prefer moderate adjustments based on historical bias
                            if bias < -0.1:  # Historical underbias
                                # Reward upward adjustments
                                if adjustment_factor > 1.1:
                                    est_mape_improvement = 0.05 * adjustment_factor
                                    est_bias_improvement = 0.06 * adjustment_factor
                                else:
                                    est_mape_improvement = -0.02
                                    est_bias_improvement = -0.03
                            elif bias > 0.1:  # Historical overbias
                                # Reward downward adjustments
                                if adjustment_factor < 0.9:
                                    est_mape_improvement = 0.04 * (2 - adjustment_factor)
                                    est_bias_improvement = 0.05 * (2 - adjustment_factor)
                                else:
                                    est_mape_improvement = -0.01
                                    est_bias_improvement = -0.02
                            else:
                                # Small historical bias - prefer small adjustments
                                est_mape_improvement = 0.02 * (1 - abs(adjustment_factor - 1.0))
                                est_bias_improvement = 0.02 * (1 - abs(adjustment_factor - 1.0))
                                
                    elif sku_band in ['D', 'E']:  # Low-volume, slow-selling items
                        # For slow-moving items, prioritize avoiding overbias
                        if adjustment_factor > 1.0:
                            # Penalize large upward adjustments (avoid overstock)
                            est_mape_improvement = 0.02 - 0.04 * (adjustment_factor - 1.0)
                            est_bias_improvement = 0.02 - 0.06 * (adjustment_factor - 1.0)
                            
                            # Exception for promotions
                            if is_promotion:
                                est_mape_improvement += 0.04
                                est_bias_improvement += 0.04
                        elif adjustment_factor < 1.0:
                            # Reward small downward adjustments (reduce overstock risk)
                            adj_magnitude = 1.0 - adjustment_factor
                            if adj_magnitude < 0.2:  # Small adjustment
                                est_mape_improvement = 0.03 * adj_magnitude  
                                est_bias_improvement = 0.04 * adj_magnitude
                            else:  # Large adjustment
                                est_mape_improvement = 0.01  # Diminishing returns for very large adjustments
                                est_bias_improvement = 0.01
                    else:  # 'C' band - balanced approach
                        # Pattern-specific proxy rewards
                        if pattern_type == "promo_holiday":
                            if is_holiday or is_promotion:
                                # For promo/holiday SKUs during events, reward upward adjustments
                                if adjustment_factor > 1.1:  # Significant increase
                                    est_mape_improvement = 0.05 * adjustment_factor
                                    est_bias_improvement = 0.08 * adjustment_factor
                                else:
                                    # Penalize for not increasing forecast during events
                                    est_mape_improvement = -0.03
                                    est_bias_improvement = -0.04
                            else:
                                # For non-event days, smaller adjustments are better
                                adj_magnitude = abs(adjustment_factor - 1.0)
                                if adj_magnitude < 0.1:
                                    est_mape_improvement = 0.01
                                    est_bias_improvement = 0.01
                                else:
                                    est_mape_improvement = -0.01 * adj_magnitude
                                    est_bias_improvement = -0.01 * adj_magnitude
                        
                        elif pattern_type == "day_pattern":
                            # For day pattern SKUs, reward adjustments that match day-of-week patterns
                            dow = target_date.weekday()
                            
                            if dow >= 5:  # Weekend
                                # For weekends, reward upward adjustments
                                if adjustment_factor > 1.1:
                                    est_mape_improvement = 0.05 * adjustment_factor
                                    est_bias_improvement = 0.06 * adjustment_factor
                                else:
                                    est_mape_improvement = -0.02
                                    est_bias_improvement = -0.03
                            elif dow == 0:  # Monday
                                # For Mondays, reward downward adjustments
                                if adjustment_factor < 0.9:
                                    est_mape_improvement = 0.04 * (2 - adjustment_factor)
                                    est_bias_improvement = 0.05 * (2 - adjustment_factor)
                                else:
                                    est_mape_improvement = -0.01
                                    est_bias_improvement = -0.02
                            else:
                                # For other days, moderate adjustments are better
                                adj_magnitude = abs(adjustment_factor - 1.0)
                                if adj_magnitude < 0.2:
                                    est_mape_improvement = 0.01
                                    est_bias_improvement = 0.01
                                else:
                                    est_mape_improvement = -0.01 * adj_magnitude
                                    est_bias_improvement = -0.01 * adj_magnitude
                        
                        elif pattern_type == "underbias":
                            # For underbias SKUs, reward upward adjustments
                            if adjustment_factor > 1.0:
                                est_mape_improvement = 0.03 * adjustment_factor
                                est_bias_improvement = 0.04 * adjustment_factor
                            else:
                                est_mape_improvement = -0.02
                                est_bias_improvement = -0.03
                        
                        else:
                            # Generic strategy for unknown patterns
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
                    
                    # Combine all factors for a heuristic reward
                    band_factor = 1.0
                    if sku_band in ['A', 'B']:
                        band_factor = self.band_emphasis
                    elif sku_band in ['D', 'E']:
                        band_factor = self.band_emphasis * 0.8
                        
                    pattern_factor = 1.0
                    if pattern_type == "promo_holiday" and (is_holiday or is_promotion):
                        pattern_factor = self.pattern_emphasis
                    elif pattern_type == "day_pattern" and target_date.weekday() >= 5:
                        pattern_factor = self.pattern_emphasis
                    elif pattern_type == "underbias":
                        pattern_factor = self.pattern_emphasis * 0.8
                    
                    if self.optimize_for == "mape":
                        reward = est_mape_improvement * self.reward_scaling * pattern_factor * horizon_factor * band_factor
                    elif self.optimize_for == "bias":
                        reward = est_bias_improvement * self.reward_scaling * pattern_factor * horizon_factor * band_factor
                    else:  # "both"
                        reward = (est_mape_improvement + est_bias_improvement) * (self.reward_scaling / 2) * pattern_factor * horizon_factor * band_factor
                    
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
                'day_of_month': target_date.day,
                'pattern_type': pattern_type,
                'sku_band': sku_band  # NEW: Record SKU band
            })
        
        # Update baseline metrics (first time we have actuals)
        if count_with_actuals > 0 and self.baseline_mape is None:
            self.baseline_mape = sum_original_mape / count_with_actuals
            self.baseline_bias = sum_original_bias / count_with_actuals
        
        # Track overall improvement
        if count_with_actuals > 0:
            current_mape = sum_adjusted_mape / count_with_actuals
            current_bias = sum_adjusted_bias / count_with_actuals
            
            if self.baseline_mape is not None and self.baseline_mape > 0:
                mape_improvement = (self.baseline_mape - current_mape) / self.baseline_mape
                self.total_improvement += mape_improvement
        
        # Increment current step
        self.current_step += 1
        
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
    
    def get_pattern_performance(self) -> Dict:
        """Get performance metrics for each pattern type."""
        performance = {}
        
        if self.has_pattern_types:
            for pattern, metrics in self.pattern_performance.items():
                if len(metrics['original_mape']) > 0:
                    avg_original_mape = np.mean(metrics['original_mape'])
                    avg_adjusted_mape = np.mean(metrics['adjusted_mape'])
                    avg_original_bias = np.mean([abs(b) for b in metrics['original_bias']])
                    avg_adjusted_bias = np.mean([abs(b) for b in metrics['adjusted_bias']])
                    
                    mape_improvement = (avg_original_mape - avg_adjusted_mape) / avg_original_mape if avg_original_mape > 0 else 0
                    bias_improvement = (avg_original_bias - avg_adjusted_bias) / avg_original_bias if avg_original_bias > 0 else 0
                    
                    performance[pattern] = {
                        'mape_improvement': mape_improvement,
                        'bias_improvement': bias_improvement,
                        'sample_count': len(metrics['original_mape'])
                    }
        
        return performance
    
    def get_band_performance(self) -> Dict:
        """Get performance metrics for each SKU band."""
        performance = {}
        
        for band, metrics in self.band_performance.items():
            if len(metrics['original_mape']) > 0:
                avg_original_mape = np.mean(metrics['original_mape'])
                avg_adjusted_mape = np.mean(metrics['adjusted_mape'])
                avg_original_bias = np.mean([abs(b) for b in metrics['original_bias']])
                avg_adjusted_bias = np.mean([abs(b) for b in metrics['adjusted_bias']])
                
                mape_improvement = (avg_original_mape - avg_adjusted_mape) / avg_original_mape if avg_original_mape > 0 else 0
                bias_improvement = (avg_original_bias - avg_adjusted_bias) / avg_original_bias if avg_original_bias > 0 else 0
                
                # For band D and E, we're more concerned about overbias
                # Calculate additional metrics specifically for overbias
                overbias_indices = [i for i, b in enumerate(metrics['original_bias']) if b > 0]
                if overbias_indices:
                    orig_overbias = np.mean([metrics['original_bias'][i] for i in overbias_indices])
                    adj_overbias = np.mean([metrics['adjusted_bias'][i] for i in overbias_indices])
                    overbias_reduction = (orig_overbias - adj_overbias) / orig_overbias if orig_overbias > 0 else 0
                else:
                    overbias_reduction = 0
                
                performance[band] = {
                    'mape_improvement': mape_improvement,
                    'bias_improvement': bias_improvement,
                    'overbias_reduction': overbias_reduction,
                    'sample_count': len(metrics['original_mape'])
                }
        
        return performance
    
    def get_dow_performance(self) -> Dict:
        """Get performance metrics for each day of the week."""
        performance = {}
        
        for dow, metrics in self.dow_performance.items():
            if metrics['count'] > 0:
                avg_original_mape = np.mean(metrics['original_mape'])
                avg_adjusted_mape = np.mean(metrics['adjusted_mape'])
                
                mape_improvement = (avg_original_mape - avg_adjusted_mape) / avg_original_mape if avg_original_mape > 0 else 0
                
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                performance[day_names[dow]] = {
                    'mape_improvement': mape_improvement,
                    'sample_count': metrics['count']
                }
        
        return performance
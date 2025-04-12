"""
Streamlined reinforcement learning environment for forecast adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta


class ForecastEnvironment:
    """
    Environment for training RL agents to adjust forecasts with support for
    SKU banding, calendar effects, holidays, and promotions.
    """
    
    def __init__(self, 
                forecast_data: pd.DataFrame,
                actual_data: pd.DataFrame,
                sku_band_data: Optional[pd.DataFrame] = None,
                holiday_data: Optional[pd.DataFrame] = None,
                promotion_data: Optional[pd.DataFrame] = None,
                forecast_horizon: int = 14,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                reward_scaling: float = 5.0,
                logger: Optional[logging.Logger] = None):
        """Initialize the forecast adjustment environment."""
        self.forecast_data = forecast_data
        self.actual_data = actual_data
        self.forecast_horizon = forecast_horizon
        self.optimize_for = optimize_for
        self.reward_scaling = reward_scaling
        
        # Set up logger
        self.logger = logger or logging.getLogger("ForecastEnvironment")
        
        # Validate required columns
        self._validate_data()
        
        # Extract SKUs
        self.skus = self.forecast_data['sku_id'].unique().tolist()
        self.logger.info(f"Environment initialized with {len(self.skus)} SKUs")

        # Process SKU band data
        self.sku_bands = self._setup_sku_bands(sku_band_data)
        
        # Check for pattern types
        self.has_pattern_types = 'pattern_type' in self.actual_data.columns
        self.sku_patterns = self._extract_patterns() if self.has_pattern_types else {}
        
        # Convert date columns
        self.forecast_data['forecast_date'] = pd.to_datetime(self.forecast_data['forecast_date'])
        self.actual_data['date'] = pd.to_datetime(self.actual_data['date'])
        
        # Set date range
        self._set_date_range(start_date, end_date)
        
        # Extract forecast columns
        self.ml_cols = [col for col in self.forecast_data.columns if col.startswith('ml_day_')]
        self.accuracy_cols = [col for col in self.forecast_data.columns if col.startswith('ml_mape') or col.startswith('ml_bias')]
        
        # Process holiday and promotion data
        self.holiday_calendar = self._setup_holiday_calendar(holiday_data)
        self.promotion_schedule = self._setup_promotion_schedule(promotion_data)
        
        # Create actual value lookup
        self.actual_values = self._setup_actual_values()
        
        # Initialize state
        self.current_forecast_idx = 0
        self.current_horizon_day = 0
        self.current_state = None
        self.done = False
        self.adjustment_history = []
        
        # Performance tracking
        self.pattern_performance = {}
        self.band_performance = {band: {'original_mape': [], 'adjusted_mape': []} for band in ['A', 'B', 'C', 'D', 'E']}
        self.baseline_mape = None
        self.total_improvement = 0.0
    
    def _validate_data(self):
        """Validate required columns in data."""
        # Check forecast data
        for col in ['forecast_date', 'sku_id']:
            if col not in self.forecast_data.columns:
                raise ValueError(f"Forecast data must contain '{col}' column")
        
        # Check actual data
        for col in ['date', 'sku_id', 'actual_value']:
            if col not in self.actual_data.columns:
                raise ValueError(f"Actual data must contain '{col}' column")
    
    def _setup_sku_bands(self, sku_band_data: Optional[pd.DataFrame]) -> Dict[str, str]:
        """Process SKU band data."""
        bands = {}
        
        if sku_band_data is not None:
            if all(col in sku_band_data.columns for col in ['sku_id', 'band']):
                # Create band lookup
                for _, row in sku_band_data.iterrows():
                    sku_id = row['sku_id']
                    band = row['band']
                    
                    if band not in ['A', 'B', 'C', 'D', 'E']:
                        band = 'C'  # Default to C for unknown bands
                        
                    bands[sku_id] = band
        
        # Default to 'C' for any SKU without band info
        for sku in self.skus:
            if sku not in bands:
                bands[sku] = 'C'
                
        return bands
    
    def _extract_patterns(self) -> Dict[str, str]:
        """Extract pattern types for each SKU."""
        patterns = {}
        
        if self.has_pattern_types:
            # Get unique SKU-pattern pairs
            sku_patterns = self.actual_data[['sku_id', 'pattern_type']].drop_duplicates()
            
            # Create lookup dictionary
            for _, row in sku_patterns.iterrows():
                patterns[row['sku_id']] = row['pattern_type']
                
            # Log pattern distribution
            self.logger.info(f"Found patterns: {self.actual_data['pattern_type'].value_counts().to_dict()}")
                
        return patterns
    
    def _set_date_range(self, start_date: Optional[str], end_date: Optional[str]):
        """Set forecast date range."""
        # Set start date
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = self.forecast_data['forecast_date'].min()
            
        # Set end date
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            max_actual = self.actual_data['date'].max()
            self.end_date = max_actual - pd.Timedelta(days=self.forecast_horizon-1)
        
        # Filter forecast data to date range
        self.forecast_data = self.forecast_data[
            (self.forecast_data['forecast_date'] >= self.start_date) & 
            (self.forecast_data['forecast_date'] <= self.end_date)
        ]
        
        # Extract unique forecast dates
        self.forecast_dates = sorted(self.forecast_data['forecast_date'].unique())
        
        if not self.forecast_dates:
            raise ValueError(f"No forecasts found in date range {self.start_date} to {self.end_date}")
            
        self.logger.info(f"Using {len(self.forecast_dates)} forecast dates from {self.start_date} to {self.end_date}")
    
    def _setup_actual_values(self) -> Dict[str, Dict[pd.Timestamp, float]]:
        """Create lookup for actual values."""
        values = {}
        
        # Group by SKU and date for faster access
        for _, row in self.actual_data.iterrows():
            sku = row['sku_id']
            date = row['date']
            value = row['actual_value']
            
            if sku not in values:
                values[sku] = {}
                
            values[sku][date] = value
        
        return values
    
    def _setup_holiday_calendar(self, holiday_data: Optional[pd.DataFrame]) -> Dict[pd.Timestamp, str]:
        """Process holiday data."""
        calendar = {}
        
        if holiday_data is not None:
            try:
                for _, row in holiday_data.iterrows():
                    date = pd.to_datetime(row['date'])
                    name = row['holiday_name']
                    calendar[date] = name
                
                self.logger.info(f"Processed {len(calendar)} holidays")
            except Exception as e:
                self.logger.warning(f"Error processing holiday data: {str(e)}")
        
        return calendar
    
    def _setup_promotion_schedule(self, promotion_data: Optional[pd.DataFrame]) -> Dict[Tuple[str, pd.Timestamp], str]:
        """Process promotion data."""
        promos = {}
        
        if promotion_data is not None:
            try:
                for _, row in promotion_data.iterrows():
                    sku = row['sku_id']
                    
                    # Skip if SKU not in dataset
                    if sku not in self.skus:
                        continue
                    
                    # Convert dates
                    start_date = pd.to_datetime(row['start_date'])
                    end_date = pd.to_datetime(row['end_date'])
                    promo_type = row.get('promo_type', 'generic')
                    
                    # Store each day in range
                    current_date = start_date
                    while current_date <= end_date:
                        promos[(sku, current_date)] = promo_type
                        current_date += pd.Timedelta(days=1)
                
                self.logger.info(f"Processed {len(promos)} promotion days")
            except Exception as e:
                self.logger.warning(f"Error processing promotion data: {str(e)}")
        
        return promos
    
    def get_feature_dims(self) -> Tuple[int, int, int, int, int, int, int, int]:
        """Get dimensions of the state features."""
        forecast_dim = min(len(self.ml_cols), self.forecast_horizon)
        error_metrics_dim = len(self.accuracy_cols)
        
        # Calendar features: day of week, day of month, is weekend
        calendar_dim = 3 * self.forecast_horizon
        
        # Holiday and promotion indicators
        holiday_dim = self.forecast_horizon
        promo_dim = self.forecast_horizon
        
        # Horizon position
        horizon_position_dim = self.forecast_horizon
        
        # Band features (one-hot for A-E)
        band_dim = 5
        
        # Total dimensions
        total_dim = forecast_dim + error_metrics_dim + calendar_dim + holiday_dim + promo_dim + horizon_position_dim + band_dim
        
        return forecast_dim, error_metrics_dim, calendar_dim, holiday_dim, promo_dim, horizon_position_dim, band_dim, total_dim
    
    def _check_if_holiday(self, date_or_offset: Union[pd.Timestamp, int]) -> bool:
        """Check if date is a holiday."""
        if isinstance(date_or_offset, int):
            # Convert day offset to date
            date = self.start_date + pd.Timedelta(days=date_or_offset)
        else:
            date = date_or_offset
            
        return date in self.holiday_calendar
    
    def _check_if_promotion(self, sku: str, date_or_offset: Union[pd.Timestamp, int]) -> bool:
        """Check if SKU is on promotion for date."""
        if isinstance(date_or_offset, int):
            # Convert day offset to date
            date = self.start_date + pd.Timedelta(days=date_or_offset)
        else:
            date = date_or_offset
            
        return (sku, date) in self.promotion_schedule
    
    def _get_context_features(self, forecast_date: pd.Timestamp, sku: str, horizon_day: int) -> Dict:
        """Get context features for a specific forecast situation."""
        # Target date
        target_date = forecast_date + pd.Timedelta(days=horizon_day + 1)
        
        # Basic calendar features
        day_of_week = target_date.weekday()
        day_of_month = target_date.day
        is_weekend = day_of_week >= 5
        
        # Check special events
        is_holiday = self._check_if_holiday(target_date)
        is_promotion = self._check_if_promotion(sku, target_date)
        
        # Get SKU-specific info
        sku_band = self.sku_bands.get(sku, 'C')
        pattern_type = self.sku_patterns.get(sku, "unknown") if self.has_pattern_types else "unknown"
        
        return {
            'day_of_week': day_of_week / 6.0,  # Normalize to [0,1]
            'day_of_month': day_of_month / 31.0,  # Normalize to [0,1]
            'is_weekend': float(is_weekend),
            'is_holiday': float(is_holiday),
            'is_promotion': float(is_promotion),
            'sku_band': sku_band,
            'pattern_type': pattern_type
        }
        
    def _get_sku_state(self, sku: str, forecast_date: pd.Timestamp, horizon_day: int) -> np.ndarray:
        """Get state representation for a specific SKU, forecast date, and horizon day."""
        # Get feature dimensions
        feature_dims = self.get_feature_dims()
        total_dim = feature_dims[-1]
        
        # Get forecasts for this SKU and date
        sku_forecasts = self.forecast_data[
            (self.forecast_data['sku_id'] == sku) & 
            (self.forecast_data['forecast_date'] == forecast_date)
        ]
        
        if sku_forecasts.empty:
            # Default state if SKU not found
            return np.zeros(total_dim)
        
        # Use first matching row
        sku_forecast = sku_forecasts.iloc[0]
        
        # Initialize state components
        forecast_dim, error_dim, calendar_dim, holiday_dim, promo_dim, horizon_dim, band_dim, _ = feature_dims
        
        # 1. Extract forecasts
        forecasts = np.zeros(forecast_dim)
        for i in range(min(forecast_dim, self.forecast_horizon)):
            ml_col = f'ml_day_{i+1}'
            if ml_col in sku_forecast:
                forecasts[i] = sku_forecast[ml_col]
        
        # 2. Extract error metrics
        error_metrics = np.zeros(error_dim)
        for i, col in enumerate(self.accuracy_cols):
            if col in sku_forecast:
                error_metrics[i] = sku_forecast[col]
        
        # 3. Prepare calendar features
        calendar_flat = np.zeros(calendar_dim)
        for i in range(self.forecast_horizon):
            target_date = forecast_date + pd.Timedelta(days=i+1)
            day_of_week = target_date.weekday() / 6.0  # Normalize to [0,1]
            day_of_month = target_date.day / 31.0  # Normalize to [0,1]
            is_weekend = float(target_date.weekday() >= 5)
            
            # Store calendar features in groups of 3
            calendar_flat[i*3] = day_of_week
            calendar_flat[i*3+1] = day_of_month
            calendar_flat[i*3+2] = is_weekend
        
        # 4. Holiday indicators
        holiday_indicators = np.zeros(holiday_dim)
        for i in range(self.forecast_horizon):
            target_date = forecast_date + pd.Timedelta(days=i+1)
            holiday_indicators[i] = float(self._check_if_holiday(target_date))
        
        # 5. Promotion indicators
        promo_indicators = np.zeros(promo_dim)
        for i in range(self.forecast_horizon):
            target_date = forecast_date + pd.Timedelta(days=i+1)
            promo_indicators[i] = float(self._check_if_promotion(sku, target_date))
        
        # 6. Horizon position (one-hot)
        horizon_position = np.zeros(horizon_dim)
        if 0 <= horizon_day < horizon_dim:
            horizon_position[horizon_day] = 1.0
        
        # 7. Band indicator (one-hot)
        band_indicator = np.zeros(band_dim)
        band = self.sku_bands.get(sku, 'C')
        band_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}.get(band, 2)
        band_indicator[band_idx] = 1.0
        
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
    
    def reset(self) -> List[np.ndarray]:
        """Reset the environment to the initial state."""
        self.current_forecast_idx = 0
        self.current_horizon_day = 0
        self.done = False
        self.adjustment_history = []
        
        # Get current forecast date
        forecast_date = self.forecast_dates[self.current_forecast_idx]
        
        # Get initial state for all SKUs
        self.current_state = [
            self._get_sku_state(sku, forecast_date, self.current_horizon_day) 
            for sku in self.skus
        ]
        
        # Reset performance tracking
        self.pattern_performance = {}
        if self.has_pattern_types:
            for pattern in set(self.sku_patterns.values()):
                self.pattern_performance[pattern] = {
                    'original_mape': [],
                    'adjusted_mape': []
                }
        
        # Reset band performance
        for band in self.band_performance:
            self.band_performance[band] = {
                'original_mape': [],
                'adjusted_mape': []
            }
        
        self.baseline_mape = None
        self.total_improvement = 0.0
        
        # Track current step
        self.current_step = 1
        
        return self.current_state
    
    def _calculate_reward(self, 
                         sku: str, 
                         original_forecast: float, 
                         adjusted_forecast: float, 
                         actual_value: float, 
                         context: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate reward and metrics based on forecast adjustment."""
        metrics = {
            'original_mape': 0.0,
            'adjusted_mape': 0.0,
            'original_bias': 0.0,
            'adjusted_bias': 0.0
        }
        
        # Default reward
        reward = 0.0
        
        # If we have actual data and it's positive
        if actual_value > 0:
            # Calculate error metrics
            metrics['original_mape'] = abs(original_forecast - actual_value) / actual_value
            metrics['adjusted_mape'] = abs(adjusted_forecast - actual_value) / actual_value
            metrics['original_bias'] = (original_forecast - actual_value) / actual_value
            metrics['adjusted_bias'] = (adjusted_forecast - actual_value) / actual_value
            
            # Calculate improvements
            mape_improvement = metrics['original_mape'] - metrics['adjusted_mape']
            bias_improvement = abs(metrics['original_bias']) - abs(metrics['adjusted_bias'])
            
            # Context-specific scaling
            scale_factor = 1.0
            
            # Band-specific scaling
            sku_band = context.get('sku_band', 'C')
            if sku_band in ['A', 'B']:  # High-volume
                scale_factor *= 1.5
                # Penalty for increasing underbias
                if metrics['original_bias'] < 0 and metrics['adjusted_bias'] < 0 and abs(metrics['adjusted_bias']) > abs(metrics['original_bias']):
                    scale_factor *= 0.5
            elif sku_band in ['D', 'E']:  # Low-volume
                scale_factor *= 0.8
                # Penalty for increasing overbias
                if metrics['original_bias'] > 0 and metrics['adjusted_bias'] > 0 and metrics['adjusted_bias'] > metrics['original_bias']:
                    scale_factor *= 0.5
            
            # Special event scaling
            if context.get('is_holiday', False) or context.get('is_promotion', False):
                scale_factor *= 1.3
            
            # Pattern-specific scaling
            pattern_type = context.get('pattern_type', 'unknown')
            if (pattern_type == "promo_holiday" and (context.get('is_holiday', False) or context.get('is_promotion', False))) or \
               (pattern_type == "day_pattern" and context.get('is_weekend', False)):
                scale_factor *= 1.2
            
            # Calculate reward based on improvements
            mape_reward = mape_improvement * self.reward_scaling * scale_factor
            bias_reward = bias_improvement * self.reward_scaling * scale_factor
            
            # Positive reinforcement
            if mape_improvement > 0:
                mape_reward *= 1.3
            
            # Final reward based on optimization target
            if self.optimize_for == "mape":
                reward = mape_reward
            elif self.optimize_for == "bias":
                reward = bias_reward
            else:  # "both"
                reward = (mape_reward + bias_reward) / 2
            
            # Track metrics by pattern type
            if self.has_pattern_types and pattern_type in self.pattern_performance:
                self.pattern_performance[pattern_type]['original_mape'].append(metrics['original_mape'])
                self.pattern_performance[pattern_type]['adjusted_mape'].append(metrics['adjusted_mape'])
            
            # Track metrics by band
            if sku_band in self.band_performance:
                self.band_performance[sku_band]['original_mape'].append(metrics['original_mape'])
                self.band_performance[sku_band]['adjusted_mape'].append(metrics['adjusted_mape'])
                
        else:
            # No actual data - use heuristic reward
            # Get historical metrics if available
            forecast_date = self.forecast_dates[self.current_forecast_idx]
            sku_forecasts = self.forecast_data[
                (self.forecast_data['sku_id'] == sku) & 
                (self.forecast_data['forecast_date'] == forecast_date)
            ]
            
            if not sku_forecasts.empty:
                sku_forecast = sku_forecasts.iloc[0]
                
                # Use relevant historical metrics
                if self.current_horizon_day < 7:
                    mape_col = 'ml_mape_7d'
                    bias_col = 'ml_bias_7d'
                else:
                    mape_col = 'ml_mape_30d'
                    bias_col = 'ml_bias_30d'
                
                hist_mape = sku_forecast[mape_col] if mape_col in sku_forecast else 0.2
                hist_bias = sku_forecast[bias_col] if bias_col in sku_forecast else 0.0
                
                # Calculate adjustment factor
                factor = adjusted_forecast / original_forecast if original_forecast > 0 else 1.0
                
                # Simple heuristic reward
                # Direction-based: improve if adjustment counters historical bias
                if hist_bias > 0.1 and factor < 0.9:  # Overbias corrected by downward adjustment
                    reward = 0.5 * self.reward_scaling * (1.0 - factor)
                elif hist_bias < -0.1 and factor > 1.1:  # Underbias corrected by upward adjustment
                    reward = 0.5 * self.reward_scaling * (factor - 1.0)
                else:
                    # Small reward for minor adjustments when bias is small
                    reward = 0.1 * self.reward_scaling * (1.0 - abs(factor - 1.0))
                
                # Fill metrics with estimates for tracking
                metrics['original_mape'] = hist_mape
                metrics['adjusted_mape'] = hist_mape * 0.9  # Assume some improvement
                metrics['original_bias'] = hist_bias
                metrics['adjusted_bias'] = hist_bias * 0.8  # Assume some improvement
        
        return reward, metrics
    
    def step(self, actions: Dict[str, Tuple[int, float]]) -> Tuple[List[np.ndarray], Dict[str, float], bool, Dict]:
        """Take a step in the environment by applying forecast adjustments."""
        rewards = {}
        info = {
            'original_mape': {},
            'adjusted_mape': {},
            'original_bias': {},
            'adjusted_bias': {},
            'is_holiday': {},
            'is_promotion': {},
            'target_date': {},
            'has_actual': {},
            'pattern_type': {},
            'sku_band': {}
        }
        
        # Get current forecast date
        forecast_date = self.forecast_dates[self.current_forecast_idx]
        
        # Calculate target date
        target_date = forecast_date + pd.Timedelta(days=self.current_horizon_day + 1)
        
        # Initialize metrics tracking
        sum_original_mape = 0.0
        sum_adjusted_mape = 0.0
        count_with_actuals = 0
        
        # Process actions for each SKU
        for i, sku in enumerate(self.skus):
            if sku not in actions:
                rewards[sku] = 0.0
                continue
                
            action_idx, adjusted_forecast = actions[sku]
            
            # Get original forecast
            original_forecast = 0.0
            
            # Find matching forecast
            sku_forecasts = self.forecast_data[
                (self.forecast_data['sku_id'] == sku) & 
                (self.forecast_data['forecast_date'] == forecast_date)
            ]
            
            if not sku_forecasts.empty:
                sku_forecast = sku_forecasts.iloc[0]
                ml_col = f'ml_day_{self.current_horizon_day + 1}'
                if ml_col in sku_forecast:
                    original_forecast = sku_forecast[ml_col]
            
            # Get context information
            context = self._get_context_features(forecast_date, sku, self.current_horizon_day)
            
            # Add context to info
            info['is_holiday'][sku] = bool(context['is_holiday'])
            info['is_promotion'][sku] = bool(context['is_promotion'])
            info['target_date'][sku] = target_date
            info['pattern_type'][sku] = context['pattern_type']
            info['sku_band'][sku] = context['sku_band']
            
            # Check for actual data
            has_actual = False
            actual_value = 0.0
            
            if sku in self.actual_values and target_date in self.actual_values[sku]:
                has_actual = True
                actual_value = self.actual_values[sku][target_date]
                
            info['has_actual'][sku] = has_actual
            
            # Calculate reward and metrics
            reward, metrics = self._calculate_reward(
                sku, original_forecast, adjusted_forecast, actual_value, context
            )
            
            # Store metrics
            for metric, value in metrics.items():
                info[metric][sku] = float(value)
            
            rewards[sku] = reward
            
            # Track for baseline
            if has_actual and actual_value > 0:
                sum_original_mape += metrics['original_mape']
                sum_adjusted_mape += metrics['adjusted_mape']
                count_with_actuals += 1
            
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
                'is_holiday': bool(context['is_holiday']),
                'is_promotion': bool(context['is_promotion']),
                'day_of_week': target_date.weekday(),
                'pattern_type': context['pattern_type'],
                'sku_band': context['sku_band']
            })
        
        # Update baseline metrics
        if count_with_actuals > 0 and self.baseline_mape is None:
            self.baseline_mape = sum_original_mape / count_with_actuals
        
        # Track overall improvement
        if count_with_actuals > 0 and self.baseline_mape is not None and self.baseline_mape > 0:
            current_mape = sum_adjusted_mape / count_with_actuals
            mape_improvement = (self.baseline_mape - current_mape) / self.baseline_mape
            self.total_improvement += mape_improvement
        
        # Increment step counter
        self.current_step += 1
        
        # Increment horizon day
        self.current_horizon_day += 1
        
        # Check if we need to move to next forecast date
        if self.current_horizon_day >= self.forecast_horizon:
            self.current_horizon_day = 0
            self.current_forecast_idx += 1
            
        # Check if episode is done
        if self.current_forecast_idx >= len(self.forecast_dates):
            self.done = True
            
        # Get next state
        if not self.done:
            forecast_date = self.forecast_dates[self.current_forecast_idx]
            next_state = [
                self._get_sku_state(sku, forecast_date, self.current_horizon_day) 
                for sku in self.skus
            ]
            self.current_state = next_state
        else:
            next_state = self.current_state
        
        return next_state, rewards, self.done, info
    
    def get_adjustment_history(self) -> List[Dict]:
        """Get history of all forecast adjustments."""
        return self.adjustment_history
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary across patterns and bands."""
        summary = {
            'overall_improvement': self.total_improvement,
            'patterns': {},
            'bands': {}
        }
        
        # Pattern performance
        for pattern, metrics in self.pattern_performance.items():
            if metrics['original_mape']:
                avg_original = np.mean(metrics['original_mape'])
                avg_adjusted = np.mean(metrics['adjusted_mape'])
                improvement = (avg_original - avg_adjusted) / avg_original if avg_original > 0 else 0
                
                summary['patterns'][pattern] = {
                    'improvement': improvement,
                    'sample_count': len(metrics['original_mape'])
                }
        
        # Band performance
        for band, metrics in self.band_performance.items():
            if metrics['original_mape']:
                avg_original = np.mean(metrics['original_mape'])
                avg_adjusted = np.mean(metrics['adjusted_mape'])
                improvement = (avg_original - avg_adjusted) / avg_original if avg_original > 0 else 0
                
                summary['bands'][band] = {
                    'improvement': improvement,
                    'sample_count': len(metrics['original_mape'])
                }
        
        return summary
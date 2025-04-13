"""
Data loading and management for the Forecast Adjustment RL system.
Provides access to ML forecasts, actual sales, and adjustment history.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

class DataProvider:
    """
    Provides data access for the RL system, loading and managing forecasts and actuals.
    Also provides metrics calculation and feature extraction.
    """
    
    def __init__(self, config):
        """
        Initialize the data provider.
        
        Args:
            config: Configuration dictionary with data settings
        """
        self.config = config
        self.data_config = config['DATA_CONFIG']
        
        # Data storage
        self.ml_forecasts = None
        self.actual_sales = None
        self.adjustments = None
        
        # Load data
        self._load_data()
        
        # Cache for computed metrics
        self.metric_cache = {}
    
    def _load_data(self):
        """Load forecast and sales data."""
        try:
            # Load ML forecasts
            if os.path.exists(self.data_config['forecasts_file']):
                self.ml_forecasts = pd.read_csv(
                    self.data_config['forecasts_file'],
                    parse_dates=['date']
                )
                logger.info(f"Loaded ML forecasts: {len(self.ml_forecasts)} records")
            else:
                # For POC, create a synthetic dataset if file doesn't exist
                self.ml_forecasts = self._create_synthetic_forecasts()
                logger.info(f"Created synthetic ML forecasts: {len(self.ml_forecasts)} records")
            
            # Load actual sales
            if os.path.exists(self.data_config['actuals_file']):
                self.actual_sales = pd.read_csv(
                    self.data_config['actuals_file'],
                    parse_dates=['date']
                )
                logger.info(f"Loaded actual sales: {len(self.actual_sales)} records")
            else:
                # For POC, create a synthetic dataset if file doesn't exist
                self.actual_sales = self._create_synthetic_actuals()
                logger.info(f"Created synthetic actual sales: {len(self.actual_sales)} records")
            
            # Load or create adjustments history
            if os.path.exists(self.data_config['adjustment_file']):
                self.adjustments = pd.read_csv(
                    self.data_config['adjustment_file'],
                    parse_dates=['date', 'adjustment_date']
                )
                logger.info(f"Loaded adjustment history: {len(self.adjustments)} records")
            else:
                # Create empty adjustments dataframe
                self.adjustments = pd.DataFrame(columns=[
                    'sku', 'category', 'band', 'date', 'adjustment_date',
                    'ml_forecast', 'adjusted_forecast', 'adjustment_factor'
                ])
                logger.info("Created empty adjustment history")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _create_synthetic_forecasts(self):
        """Create synthetic forecast data for POC."""
        # Define a date range for the synthetic data
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now() + timedelta(days=35)
        dates = pd.date_range(start=start_date, end=end_date)
        
        # Define categories and bands
        categories = [f"Category_{i}" for i in range(1, 11)]
        bands = ['A', 'B', 'C']
        
        # Create SKUs for each category and band
        skus = []
        for category in categories:
            # Create category-specific characteristics
            # Some categories inherently have forecast bias patterns
            category_bias_type = np.random.choice([
                'unbiased',        # generally accurate
                'wom_underbias',   # strong WoM1 underbias 
                'constant_under',  # consistently underforecasted
                'constant_over',   # consistently overforecasted
                'seasonal_bias'    # bias changes by season
            ], p=[0.3, 0.3, 0.2, 0.1, 0.1])  # weighted probabilities
            
            for band in bands:
                # More SKUs for higher bands
                num_skus = 20 if band == 'A' else (10 if band == 'B' else 5)
                for i in range(1, num_skus + 1):
                    skus.append((f"SKU_{category}_{band}_{i}", category, band, category_bias_type))
        
        # Create dataframe
        data = []
        for date in dates:
            # Extract date features
            week_of_month = (date.day - 1) // 7 + 1
            month = date.month
            day_of_week = date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            for sku, category, band, bias_type in skus:
                # Base forecast depends on band
                base_forecast = 1000 if band == 'A' else (500 if band == 'B' else 200)
                
                # Add category-specific variation (10-30%)
                category_factor = 1.0 + (int(category.split('_')[1]) % 3) * 0.1
                
                # Add WoM effect - ML forecast doesn't fully account for WoM1 increases
                wom_factor = 1.0
                if bias_type == 'wom_underbias' and week_of_month == 1:
                    # In WoM1, ML forecast is systematically low
                    wom_factor = 0.85  # ML underforecasts by 15% in WoM1
                
                # Add constant bias patterns
                constant_bias_factor = 1.0
                if bias_type == 'constant_under':
                    constant_bias_factor = 0.9  # ML consistently underforecasts by 10%
                elif bias_type == 'constant_over':
                    constant_bias_factor = 1.1  # ML consistently overforecasts by 10%
                
                # Add seasonal bias
                seasonal_factor = 1.0
                if bias_type == 'seasonal_bias':
                    # Q1 underforecasted, Q3 overforecasted
                    if month in [1, 2, 3]:
                        seasonal_factor = 0.9
                    elif month in [7, 8, 9]:
                        seasonal_factor = 1.1
                
                # Add day of week pattern
                dow_factor = 1.0 - 0.2 * is_weekend  # Lower forecasts on weekends
                
                # Add randomness to each SKU (±5%)
                sku_factor = 1.0 + (hash(sku) % 100) / 1000 - 0.05
                
                # Add random noise (±10%)
                noise = np.random.normal(0, 0.1)
                
                # Calculate final forecast
                forecast = (base_forecast 
                           * category_factor 
                           * wom_factor 
                           * constant_bias_factor
                           * seasonal_factor
                           * dow_factor
                           * sku_factor
                           * (1 + noise))
                
                data.append({
                    'date': date,
                    'sku': sku,
                    'category': category,
                    'band': band,
                    'bias_type': bias_type,  # store bias type for reference
                    'forecast': max(0, forecast)  # Ensure non-negative
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # For simplicity in the POC, drop the bias_type column from the final dataframe
        # In a real system, we wouldn't have this labeled information
        return df.drop(columns=['bias_type'])
    
    def _create_synthetic_actuals(self):
        """Create synthetic actual sales data for POC."""
        if self.ml_forecasts is None:
            raise ValueError("Cannot create synthetic actuals without forecasts")
        
        # Filter only past dates for actuals
        past_forecasts = self.ml_forecasts[
            self.ml_forecasts['date'] <= datetime.now()
        ].copy()
        
        # Add bias_type back for internal use
        # In real-world, we'd use actual historical patterns
        category_bias_types = {}
        for category in past_forecasts['category'].unique():
            category_bias_types[category] = np.random.choice([
                'unbiased', 'wom_underbias', 'constant_under', 
                'constant_over', 'seasonal_bias'
            ], p=[0.3, 0.3, 0.2, 0.1, 0.1])
        
        past_forecasts['bias_type'] = past_forecasts['category'].map(category_bias_types)
        
        # Add actuals based on forecasts with deliberate biases
        data = []
        for _, row in past_forecasts.iterrows():
            # Extract date features
            date = row['date']
            week_of_month = (date.day - 1) // 7 + 1
            month = date.month
            
            # Base actuals depend on the forecast
            base_actual = row['forecast']
            
            # Apply WoM effect - first week of month has higher actual sales
            wom_factor = 1.0
            if row['bias_type'] == 'wom_underbias' and week_of_month == 1:
                # In WoM1, actual sales are higher than forecasted
                wom_factor = 1.2  # 20% higher sales in WoM1
            
            # Apply constant bias patterns
            constant_bias_factor = 1.0
            if row['bias_type'] == 'constant_under':
                constant_bias_factor = 1.15  # Consistently 15% higher than forecast
            elif row['bias_type'] == 'constant_over':
                constant_bias_factor = 0.9  # Consistently 10% lower than forecast
            
            # Apply seasonal bias
            seasonal_factor = 1.0
            if row['bias_type'] == 'seasonal_bias':
                # Q1 underforecasted, Q3 overforecasted
                if month in [1, 2, 3]:
                    seasonal_factor = 1.15  # Q1 actuals higher than forecast
                elif month in [7, 8, 9]:
                    seasonal_factor = 0.95  # Q3 actuals lower than forecast
            
            # Add random noise (±15%)
            noise = np.random.normal(0, 0.15)
            
            # Calculate final actual sales
            actual = base_actual * wom_factor * constant_bias_factor * seasonal_factor * (1 + noise)
            
            data.append({
                'date': row['date'],
                'sku': row['sku'],
                'category': row['category'],
                'band': row['band'],
                'actual_sales': max(0, actual)  # Ensure non-negative
            })
        
        return pd.DataFrame(data)
    
    def get_skus_for_category_band(self, category, band):
        """
        Get all SKUs for a specific category-band combination.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            List of SKUs
        """
        skus = self.ml_forecasts[
            (self.ml_forecasts['category'] == category) & 
            (self.ml_forecasts['band'] == band)
        ]['sku'].unique().tolist()
        
        return skus
    
    def get_ml_forecasts(self, skus, date):
        """
        Get ML forecasts for specific SKUs on a specific date.
        
        Args:
            skus: List of SKU identifiers
            date: Date for which to get forecasts
            
        Returns:
            Dictionary mapping SKUs to their ML forecasts
        """
        # Normalize the date to ensure consistent comparison
        # Convert to datetime date only (no time component)
        if isinstance(date, datetime):
            normalized_date = date.date()
        else:
            normalized_date = date
            
        # Filter forecasts for the specified SKUs and normalize dates for comparison
        forecasts = self.ml_forecasts[
            (self.ml_forecasts['sku'].isin(skus)) & 
            (pd.to_datetime(self.ml_forecasts['date']).dt.date == normalized_date)
        ]
        
        # Create SKU-to-forecast mapping
        result = dict(zip(forecasts['sku'], forecasts['forecast']))
        
        # Ensure all SKUs have a forecast
        for sku in skus:
            if sku not in result:
                logger.warning(f"No forecast found for SKU {sku} on {normalized_date}")
                result[sku] = 0.0
        
        return result
    
    def save_adjustments(self, skus, adjusted_forecasts, date, adjustment_factor):
        """
        Save adjustment information.
        
        Args:
            skus: List of adjusted SKUs
            adjusted_forecasts: Dictionary mapping SKUs to adjusted forecasts
            date: Date for which adjustments were made
            adjustment_factor: Applied adjustment factor
        """
        # Import utils here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import ensure_dir
        
        # Get original ML forecasts
        ml_forecasts = self.get_ml_forecasts(skus, date)
        
        # Create records for each adjustment
        new_adjustments = []
        for sku in skus:
            if sku not in adjusted_forecasts:
                continue
                
            # Get category and band for this SKU
            sku_data = self.ml_forecasts[self.ml_forecasts['sku'] == sku].iloc[0]
            category = sku_data['category']
            band = sku_data['band']
            
            new_adjustments.append({
                'sku': sku,
                'category': category,
                'band': band,
                'date': date,
                'adjustment_date': datetime.now(),
                'ml_forecast': ml_forecasts.get(sku, 0.0),
                'adjusted_forecast': adjusted_forecasts[sku],
                'adjustment_factor': adjustment_factor
            })
        
        # Add to adjustment history
        if new_adjustments:
            self.adjustments = pd.concat([
                self.adjustments, 
                pd.DataFrame(new_adjustments)
            ], ignore_index=True)
            
            # Ensure directory exists
            adjustment_file = self.data_config['adjustment_file']
            ensure_dir(os.path.dirname(adjustment_file))
            
            # Save to file
            self.adjustments.to_csv(adjustment_file, index=False)
            
            logger.debug(f"Saved {len(new_adjustments)} adjustments")
    
    def get_mape(self, category, band, lookback_weeks=1):
        """
        Calculate MAPE for a category-band over a lookback period.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            lookback_weeks: Number of weeks to look back
            
        Returns:
            MAPE value
        """
        # Calculate start date for lookback
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7 * lookback_weeks)
        
        # Generate cache key
        cache_key = f"mape_{category}_{band}_{lookback_weeks}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Check if result is cached
        if cache_key in self.metric_cache:
            return self.metric_cache[cache_key]
        
        # Filter forecasts and actuals for this category-band and date range
        forecasts = self.ml_forecasts[
            (self.ml_forecasts['category'] == category) & 
            (self.ml_forecasts['band'] == band) & 
            (self.ml_forecasts['date'] >= start_date) & 
            (self.ml_forecasts['date'] <= end_date)
        ]
        
        actuals = self.actual_sales[
            (self.actual_sales['category'] == category) & 
            (self.actual_sales['band'] == band) & 
            (self.actual_sales['date'] >= start_date) & 
            (self.actual_sales['date'] <= end_date)
        ]
        
        # Join forecasts and actuals
        merged = pd.merge(
            forecasts, 
            actuals, 
            on=['sku', 'category', 'band', 'date']
        )
        
        if merged.empty:
            # If no data, return a high MAPE
            result = 0.5
        else:
            # Calculate MAPE
            merged['ape'] = abs(merged['forecast'] - merged['actual_sales']) / (merged['actual_sales'] + 1e-5)
            result = merged['ape'].mean()
        
        # Cache the result
        self.metric_cache[cache_key] = result
        
        return result
    
    def get_bias(self, category, band, lookback_weeks=1):
        """
        Calculate bias for a category-band over a lookback period.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            lookback_weeks: Number of weeks to look back
            
        Returns:
            Bias value (negative means underforecasting)
        """
        # Calculate start date for lookback
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7 * lookback_weeks)
        
        # Generate cache key
        cache_key = f"bias_{category}_{band}_{lookback_weeks}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Check if result is cached
        if cache_key in self.metric_cache:
            return self.metric_cache[cache_key]
        
        # Filter forecasts and actuals for this category-band and date range
        forecasts = self.ml_forecasts[
            (self.ml_forecasts['category'] == category) & 
            (self.ml_forecasts['band'] == band) & 
            (self.ml_forecasts['date'] >= start_date) & 
            (self.ml_forecasts['date'] <= end_date)
        ]
        
        actuals = self.actual_sales[
            (self.actual_sales['category'] == category) & 
            (self.actual_sales['band'] == band) & 
            (self.actual_sales['date'] >= start_date) & 
            (self.actual_sales['date'] <= end_date)
        ]
        
        # Join forecasts and actuals
        merged = pd.merge(
            forecasts, 
            actuals, 
            on=['sku', 'category', 'band', 'date']
        )
        
        if merged.empty:
            # If no data, return zero bias
            result = 0.0
        else:
            # Calculate bias (forecast - actual) / actual
            # Negative means underforecasting
            merged['bias'] = (merged['forecast'] - merged['actual_sales']) / (merged['actual_sales'] + 1e-5)
            result = merged['bias'].mean()
        
        # Cache the result
        self.metric_cache[cache_key] = result
        
        return result
    
    def get_mape_trend(self, category, band):
        """
        Calculate MAPE trend (positive means improving).
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            MAPE trend value
        """
        # Import utils here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import z_score_normalize
        
        # Get short-term and longer-term MAPE
        short_mape = self.get_mape(category, band, lookback_weeks=1)
        long_mape = self.get_mape(category, band, lookback_weeks=4)
        
        # Calculate trend (negative means MAPE is improving)
        return long_mape - short_mape
    
    def get_bias_trend(self, category, band):
        """
        Calculate bias trend (positive means less underforecasting).
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Bias trend value
        """
        # Import path_engineering here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.feature_engineering import FeatureEngineer
        
        # Get short-term and longer-term bias
        short_bias = self.get_bias(category, band, lookback_weeks=1)
        long_bias = self.get_bias(category, band, lookback_weeks=4)
        
        # Use FeatureEngineer's method to calculate bias trend
        dummy_engineer = FeatureEngineer(self.config)
        return dummy_engineer._calculate_bias_trend(short_bias, long_bias)
    
    def get_sales_volume(self, category, band):
        """
        Get average sales volume for a category-band.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Average sales volume
        """
        # Generate cache key
        cache_key = f"volume_{category}_{band}"
        
        # Check if result is cached
        if cache_key in self.metric_cache:
            return self.metric_cache[cache_key]
        
        # Filter actuals for this category-band
        actuals = self.actual_sales[
            (self.actual_sales['category'] == category) & 
            (self.actual_sales['band'] == band)
        ]
        
        if actuals.empty:
            # If no data, return zero
            result = 0.0
        else:
            # Calculate average sales
            result = actuals['actual_sales'].mean()
        
        # Cache the result
        self.metric_cache[cache_key] = result
        
        return result
    
    def get_sales_volatility(self, category, band):
        """
        Get sales volatility (coefficient of variation) for a category-band.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Sales volatility value
        """
        # Generate cache key
        cache_key = f"volatility_{category}_{band}"
        
        # Check if result is cached
        if cache_key in self.metric_cache:
            return self.metric_cache[cache_key]
        
        # Filter actuals for this category-band
        actuals = self.actual_sales[
            (self.actual_sales['category'] == category) & 
            (self.actual_sales['band'] == band)
        ]
        
        if actuals.empty or actuals['actual_sales'].mean() == 0:
            # If no data or zero mean, return high volatility
            result = 1.0
        else:
            # Calculate coefficient of variation (std/mean)
            result = actuals['actual_sales'].std() / actuals['actual_sales'].mean()
        
        # Cache the result
        self.metric_cache[cache_key] = result
        
        return result
    
    def get_forecast_momentum(self, category, band):
        """
        Calculate forecast momentum (change over time).
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Forecast momentum value
        """
        # Calculate start and end dates for momentum calculation
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        # Filter forecasts for this category-band for today and a week ago
        recent_forecasts = self.ml_forecasts[
            (self.ml_forecasts['category'] == category) & 
            (self.ml_forecasts['band'] == band) & 
            (self.ml_forecasts['date'] == now.date())
        ]
        
        prev_forecasts = self.ml_forecasts[
            (self.ml_forecasts['category'] == category) & 
            (self.ml_forecasts['band'] == band) & 
            (self.ml_forecasts['date'] == week_ago.date())
        ]
        
        if recent_forecasts.empty or prev_forecasts.empty:
            # If no data, return zero momentum
            return 0.0
        
        # Calculate average forecast for each time period
        recent_avg = recent_forecasts['forecast'].mean()
        prev_avg = prev_forecasts['forecast'].mean()
        
        # Calculate momentum as percentage change
        if prev_avg == 0:
            return 0.0
        
        return (recent_avg - prev_avg) / prev_avg
    
    def get_forecast_revision_rate(self, category, band):
        """
        Calculate how much ML forecast is revising itself.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Forecast revision rate
        """
        # Will be implemented in a real system
        # For POC, return random small value
        return np.random.uniform(-0.05, 0.05)
    
    def get_previous_adjustment(self, category, band):
        """
        Get most recent adjustment factor for a category-band.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Previous adjustment factor or 1.0 if none
        """
        # Filter adjustments for this category-band
        adj = self.adjustments[
            (self.adjustments['category'] == category) & 
            (self.adjustments['band'] == band)
        ]
        
        if adj.empty:
            # If no previous adjustment, return 1.0 (no adjustment)
            return 1.0
        
        # Get the most recent adjustment
        latest = adj.sort_values('adjustment_date', ascending=False).iloc[0]
        return latest['adjustment_factor']
    
    def get_adjustment_age(self, category, band):
        """
        Get age of the most recent adjustment in days.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Age in days or a large value if no previous adjustment
        """
        # Filter adjustments for this category-band
        adj = self.adjustments[
            (self.adjustments['category'] == category) & 
            (self.adjustments['band'] == band)
        ]
        
        if adj.empty:
            # If no previous adjustment, return a large value
            return 30.0
        
        # Get the most recent adjustment
        latest = adj.sort_values('adjustment_date', ascending=False).iloc[0]
        
        # Calculate age in days
        age = (datetime.now() - latest['adjustment_date']).total_seconds() / (24 * 3600)
        return age
    
    def get_adjustment_success_rate(self, category, band):
        """
        Calculate success rate of previous adjustments.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Success rate (0-1) or 0.5 if insufficient data
        """
        # For POC, return 0.5 (neutral)
        # In a real system, would compare adjustment outcomes with ML forecast outcomes
        return 0.5
    
    def get_ml_forecast_change_percent(self, category, band):
        """
        Calculate percentage change in ML forecast from previous day.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            
        Returns:
            Percentage change in forecast
        """
        # For POC, return small random value
        # In a real system, would compare today's ML forecast with yesterday's
        return np.random.uniform(-0.02, 0.02)
    
    def get_historical_bias(self, category, band, date, before_adjustment=True):
        """
        Get historical bias for evaluating adjustments.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            date: Date for which to get bias
            before_adjustment: Whether to use ML forecast (True) or adjusted forecast (False)
            
        Returns:
            Historical bias value
        """
        # For synthetic data, simulate systematic biases based on category and date
        
        # Get the first digit of the category number for deterministic bias
        if isinstance(category, str) and '_' in category:
            category_num = int(category.split('_')[1])
        else:
            category_num = hash(str(category)) % 10
            
        # Week of month effect
        week_of_month = self._get_week_of_month(date)
        
        # Determine bias type based on category
        bias_types = ['unbiased', 'wom_underbias', 'constant_under', 
                     'constant_over', 'seasonal_bias']
        bias_type = bias_types[category_num % len(bias_types)]
        
        # Base bias value
        base_bias = 0.02  # Slight overforecasting by default
        
        # Apply specific bias patterns based on type
        if bias_type == 'wom_underbias' and week_of_month == 1:
            # Systematic WoM1 underforecasting
            base_bias = -0.15
        elif bias_type == 'constant_under':
            # Consistently underforecasting
            base_bias = -0.1
        elif bias_type == 'constant_over':
            # Consistently overforecasting
            base_bias = 0.1
        elif bias_type == 'seasonal_bias':
            # Seasonal bias
            month = date.month
            if month in [1, 2, 3]:
                base_bias = -0.12  # Q1 underforecasting
            elif month in [7, 8, 9]:
                base_bias = 0.08   # Q3 overforecasting
                
        # If after adjustment, simulate improved bias
        if not before_adjustment:
            # Adjustment typically reduces bias magnitude by 60-80%
            correction_factor = np.random.uniform(0.6, 0.8)
            base_bias *= correction_factor
            
        return base_bias
    
    def get_historical_mape(self, category, band, date, before_adjustment=True):
        """
        Get historical MAPE for evaluating adjustments.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            date: Date for which to get MAPE
            before_adjustment: Whether to use ML forecast (True) or adjusted forecast (False)
            
        Returns:
            Historical MAPE value
        """
        # For synthetic data, MAPE is related to bias magnitude but adds additional error
        
        # Get bias (as we'll base MAPE partly on this)
        bias = self.get_historical_bias(category, band, date, before_adjustment)
        
        # Base MAPE value - depends on band (higher volume typically has lower MAPE)
        if band == 'A':
            base_mape = 0.15
        elif band == 'B':
            base_mape = 0.20
        else:  # band C
            base_mape = 0.25
            
        # Add component based on bias magnitude (higher bias typically means higher MAPE)
        bias_component = abs(bias) * 0.7  # bias contributes to MAPE but isn't the only factor
        
        # Calculate final MAPE
        mape = base_mape + bias_component
        
        # If after adjustment, simulate improved MAPE
        if not before_adjustment:
            # Adjustment typically reduces MAPE by 10-20%
            correction_factor = np.random.uniform(0.8, 0.9)
            mape *= correction_factor
            
        return mape
    
    def get_bias_with_actuals(self, category, band, adjustment_date, evaluation_date):
        """
        Calculate actual bias after adjustment when actuals become available.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            adjustment_date: Date when adjustment was made
            evaluation_date: Date when actuals became available
            
        Returns:
            Bias value with actuals
        """
        # For POC, return simulated bias
        # In a real system, would compare adjusted forecasts with actuals
        
        # Get all adjustments for this category-band on the adjustment date
        adj = self.adjustments[
            (self.adjustments['category'] == category) & 
            (self.adjustments['band'] == band) & 
            (self.adjustments['date'] == adjustment_date)
        ]
        
        if adj.empty:
            # If no adjustment was made, return the original bias
            return self.get_historical_bias(category, band, adjustment_date, before_adjustment=True)
        
        # Get the adjustment factor
        adjustment_factor = adj['adjustment_factor'].iloc[0]
        
        # Simulate the effect of the adjustment
        original_bias = self.get_historical_bias(category, band, adjustment_date, before_adjustment=True)
        
        # Perfect adjustment would reduce bias to 0
        # For POC, simulate effectiveness based on how close to ideal the adjustment was
        week_of_month = (adjustment_date.day - 1) // 7 + 1
        
        if week_of_month == 1 and original_bias < 0:
            # Week 1 underforecasting - ideal adjustment is around 1.15
            ideal_adjustment = 1.15
            effectiveness = 1.0 - abs(adjustment_factor - ideal_adjustment) / 0.2
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            # Improved bias is closer to 0
            improved_bias = original_bias * (1.0 - effectiveness)
            return improved_bias
        elif original_bias > 0:
            # Overforecasting - ideal adjustment is < 1.0
            ideal_adjustment = 0.95
            effectiveness = 1.0 - abs(adjustment_factor - ideal_adjustment) / 0.2
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            # Improved bias is closer to 0
            improved_bias = original_bias * (1.0 - effectiveness)
            return improved_bias
        else:
            # No strong bias - ideal adjustment is 1.0
            ideal_adjustment = 1.0
            effectiveness = 1.0 - abs(adjustment_factor - ideal_adjustment) / 0.2
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            # Adjustment could worsen bias if not needed
            if adjustment_factor > 1.0:
                worsened_bias = original_bias + 0.1 * (1.0 - effectiveness)
                return worsened_bias
            elif adjustment_factor < 1.0:
                worsened_bias = original_bias - 0.1 * (1.0 - effectiveness)
                return worsened_bias
            else:
                return original_bias
    
    def get_mape_with_actuals(self, category, band, adjustment_date, evaluation_date):
        """
        Calculate actual MAPE after adjustment when actuals become available.
        
        Args:
            category: Category identifier
            band: Band identifier (A, B, C)
            adjustment_date: Date when adjustment was made
            evaluation_date: Date when actuals became available
            
        Returns:
            MAPE value with actuals
        """
        # For POC, return simulated MAPE
        # In a real system, would compare adjusted forecasts with actuals
        
        # Get all adjustments for this category-band on the adjustment date
        adj = self.adjustments[
            (self.adjustments['category'] == category) & 
            (self.adjustments['band'] == band) & 
            (self.adjustments['date'] == adjustment_date)
        ]
        
        if adj.empty:
            # If no adjustment was made, return the original MAPE
            return self.get_historical_mape(category, band, adjustment_date, before_adjustment=True)
        
        # Get the adjustment factor
        adjustment_factor = adj['adjustment_factor'].iloc[0]
        
        # Simulate the effect of the adjustment
        original_mape = self.get_historical_mape(category, band, adjustment_date, before_adjustment=True)
        original_bias = self.get_historical_bias(category, band, adjustment_date, before_adjustment=True)
        
        # MAPE improvement depends on how well the adjustment corrects bias
        week_of_month = (adjustment_date.day - 1) // 7 + 1
        
        if week_of_month == 1 and original_bias < 0:
            # Week 1 underforecasting - ideal adjustment is around 1.15
            ideal_adjustment = 1.15
            effectiveness = 1.0 - abs(adjustment_factor - ideal_adjustment) / 0.2
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            # Improved MAPE with diminishing returns
            mape_improvement = 0.04 * effectiveness
            return original_mape - mape_improvement
        elif original_bias > 0:
            # Overforecasting - ideal adjustment is < 1.0
            ideal_adjustment = 0.95
            effectiveness = 1.0 - abs(adjustment_factor - ideal_adjustment) / 0.2
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            # Improved MAPE with diminishing returns
            mape_improvement = 0.03 * effectiveness
            return original_mape - mape_improvement
        else:
            # No strong bias - ideal adjustment is 1.0
            ideal_adjustment = 1.0
            effectiveness = 1.0 - abs(adjustment_factor - ideal_adjustment) / 0.2
            effectiveness = max(0.0, min(1.0, effectiveness))
            
            # Adjustment could worsen MAPE if not needed
            if adjustment_factor != 1.0:
                mape_worsening = 0.02 * (1.0 - effectiveness)
                return original_mape + mape_worsening
            else:
                return original_mape
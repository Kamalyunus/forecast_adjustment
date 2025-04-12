"""
Data generator for forecast adjustment system.
Creates synthetic forecast data with patterns for training and evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional, Tuple


def generate_forecast_dataset(
    num_skus: int = 20,
    forecast_horizon: int = 14,
    historical_days: int = 120,
    start_date: Optional[str] = None,
    output_dir: str = "data",
    include_bands: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate a dataset of historical forecasts and actuals with clear patterns for RL training.
    
    Args:
        num_skus: Number of SKUs to generate
        forecast_horizon: Number of days in each forecast horizon
        historical_days: Number of days of historical data to generate
        start_date: Starting date (defaults to historical_days ago)
        output_dir: Directory to save data
        include_bands: Whether to generate and assign SKU bands (A-E)
        logger: Logger instance
        
    Returns:
        Dictionary containing forecast_data, actual_data, holiday_data, promotion_data, and sku_band_data
    """
    if logger is None:
        logger = logging.getLogger("DataGenerator")
        
    logger.info(f"Generating forecast dataset with {num_skus} SKUs over {historical_days} days")
    
    if start_date is None:
        # Default to historical_days ago
        start_date = (datetime.now() - timedelta(days=historical_days)).strftime("%Y-%m-%d")
    
    start_date = pd.to_datetime(start_date)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate SKU IDs with specific pattern types to create clear scenarios
    skus = []
    sku_patterns = {}
    
    # Create SKUs with different patterns
    # 40% with underbias pattern
    underbias_count = int(num_skus * 0.3)
    for i in range(underbias_count):
        sku_id = f"SKU_UNDER_{i:03d}"
        skus.append(sku_id)
        sku_patterns[sku_id] = "underbias"
        
    # 30% with promo/holiday sensitivity
    promo_count = int(num_skus * 0.3)
    for i in range(promo_count):
        sku_id = f"SKU_PROMO_{i:03d}"
        skus.append(sku_id)
        sku_patterns[sku_id] = "promo_holiday"
        
    # 30% with day-of-week patterns
    dow_count = num_skus - underbias_count - promo_count
    for i in range(dow_count):
        sku_id = f"SKU_DOW_{i:03d}"
        skus.append(sku_id)
        sku_patterns[sku_id] = "day_pattern"
    
    logger.info(f"Created SKUs with patterns: {underbias_count} underbias, {promo_count} promo/holiday, {dow_count} day pattern")
    
    # Generate holiday data
    holiday_records = _generate_holiday_data(start_date, historical_days)
    
    # Create holiday DataFrame
    holiday_df = pd.DataFrame(holiday_records)
    
    # Create a set of holiday dates for easy lookup
    holiday_dates = set(holiday_df['date'])
    
    # Generate promotion data
    promo_records = _generate_promotion_data(skus, sku_patterns, start_date, historical_days)
    
    # Create promotion DataFrame
    promo_df = pd.DataFrame(promo_records)
    
    # Generate SKU band data if requested
    sku_band_df = None
    if include_bands:
        sku_band_records = _generate_sku_bands(skus, sku_patterns)
        sku_band_df = pd.DataFrame(sku_band_records)
    
    # Generate true demand patterns
    true_demand = _generate_true_demand(
        skus, 
        sku_patterns, 
        start_date, 
        historical_days, 
        forecast_horizon, 
        holiday_dates, 
        promo_df
    )
    
    # Generate forecast data with clear, learnable biases
    forecast_records, actual_records = _generate_forecast_actual_data(
        skus,
        sku_patterns,
        true_demand,
        start_date,
        historical_days,
        forecast_horizon,
        holiday_dates,
        promo_df
    )
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame(forecast_records)
    
    # Create actual DataFrame
    actual_df = pd.DataFrame(actual_records)
    
    # Save datasets to CSV
    forecast_df.to_csv(os.path.join(output_dir, 'historical_forecasts.csv'), index=False)
    actual_df.to_csv(os.path.join(output_dir, 'historical_actuals.csv'), index=False)
    holiday_df.to_csv(os.path.join(output_dir, 'holidays.csv'), index=False)
    promo_df.to_csv(os.path.join(output_dir, 'promotions.csv'), index=False)
    
    if sku_band_df is not None:
        sku_band_df.to_csv(os.path.join(output_dir, 'sku_bands.csv'), index=False)
    
    # Save SKU patterns for analysis
    sku_pattern_df = pd.DataFrame([
        {'sku_id': sku, 'pattern_type': pattern} 
        for sku, pattern in sku_patterns.items()
    ])
    sku_pattern_df.to_csv(os.path.join(output_dir, 'sku_patterns.csv'), index=False)
    
    logger.info(f"Generated {len(forecast_df)} historical forecasts for {len(skus)} SKUs")
    logger.info(f"Generated {len(actual_df)} actual values for {len(skus)} SKUs")
    logger.info(f"Generated {len(holiday_df)} holidays")
    logger.info(f"Generated {len(promo_df)} promotions")
    
    result = {
        'forecast': forecast_df,
        'historical': actual_df,
        'holiday': holiday_df,
        'promotion': promo_df
    }
    
    if sku_band_df is not None:
        result['sku_band'] = sku_band_df
    
    return result


def _generate_holiday_data(start_date: pd.Timestamp, historical_days: int) -> List[Dict]:
    """Generate synthetic holiday data."""
    holiday_records = []
    
    # Create holidays approximately once a month
    num_holidays = historical_days // 30 + 2  # Add a couple extra to ensure coverage
    # Space them out roughly evenly, with some randomness
    holiday_days = []
    for i in range(num_holidays):
        base_day = (i * 30) + np.random.randint(-5, 5)
        if 0 <= base_day < historical_days:
            holiday_days.append(base_day)
    
    holiday_names = ["New Year's Day", "Valentine's Day", "Easter", "Memorial Day", 
                    "Independence Day", "Labor Day", "Halloween", "Thanksgiving", "Black Friday", "Christmas"]
    
    for i, day in enumerate(sorted(holiday_days)):
        date = start_date + timedelta(days=int(day))
        holiday_records.append({
            'date': date.strftime('%Y-%m-%d'),
            'holiday_name': holiday_names[i % len(holiday_names)]
        })
        
    return holiday_records


def _generate_promotion_data(
    skus: List[str], 
    sku_patterns: Dict[str, str], 
    start_date: pd.Timestamp, 
    historical_days: int
) -> List[Dict]:
    """Generate synthetic promotion data."""
    promo_records = []
    
    # Assign promotions to all SKUs with promo_holiday pattern and some underbias SKUs
    promo_skus = [sku for sku in skus if sku_patterns[sku] == "promo_holiday"]
    # Add 20% of underbias SKUs to promos
    additional_promo_skus = [sku for sku in skus if sku_patterns[sku] == "underbias"]
    additional_promo_count = max(1, int(len(additional_promo_skus) * 0.2))
    additional_promo_skus = np.random.choice(additional_promo_skus, size=additional_promo_count, replace=False)
    promo_skus.extend(additional_promo_skus)
    
    for sku in promo_skus:
        # Generate 2-4 promotions
        num_promos = np.random.randint(2, 5)
        
        for _ in range(num_promos):
            # Random 5-10 day promotion
            promo_length = np.random.randint(5, 11)
            promo_start_day = np.random.randint(0, historical_days - promo_length)
            
            start_date_promo = start_date + timedelta(days=promo_start_day)
            end_date_promo = start_date_promo + timedelta(days=promo_length - 1)
            
            promo_records.append({
                'sku_id': sku,
                'start_date': start_date_promo.strftime('%Y-%m-%d'),
                'end_date': end_date_promo.strftime('%Y-%m-%d'),
                'promo_type': np.random.choice(["Price Discount", "BOGO", "Bundle"])
            })
            
    return promo_records


def _generate_sku_bands(skus: List[str], sku_patterns: Dict[str, str]) -> List[Dict]:
    """Generate SKU band assignments based on patterns and with some randomness."""
    sku_band_records = []
    
    # Base band assignment on pattern type and add some randomness
    for sku in skus:
        pattern = sku_patterns[sku]
        
        # Assign probability distributions based on pattern
        if pattern == "promo_holiday":
            # Promo/holiday SKUs tend to be higher volume
            band_probs = {'A': 0.3, 'B': 0.4, 'C': 0.2, 'D': 0.07, 'E': 0.03}
        elif pattern == "underbias":
            # Underbias can happen across any band
            band_probs = {'A': 0.15, 'B': 0.25, 'C': 0.3, 'D': 0.2, 'E': 0.1}
        else:  # day_pattern
            # Day patterns are common for medium volume items
            band_probs = {'A': 0.1, 'B': 0.2, 'C': 0.4, 'D': 0.2, 'E': 0.1}
        
        # Sample a band based on the probabilities
        bands = list(band_probs.keys())
        probabilities = list(band_probs.values())
        band = np.random.choice(bands, p=probabilities)
        
        # Add the band assignment and a base volume
        volume_ranges = {
            'A': (5000, 10000),  # High volume
            'B': (1000, 5000),   # Above average
            'C': (500, 1000),    # Average
            'D': (100, 500),     # Below average
            'E': (10, 100)       # Low volume
        }
        
        base_volume = np.random.randint(*volume_ranges[band])
        
        sku_band_records.append({
            'sku_id': sku,
            'band': band,
            'base_volume': base_volume
        })
    
    return sku_band_records


def _generate_true_demand(
    skus: List[str],
    sku_patterns: Dict[str, str],
    start_date: pd.Timestamp,
    historical_days: int,
    forecast_horizon: int,
    holiday_dates: set,
    promo_df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """Generate true demand patterns for SKUs."""
    true_demand = {}
    for sku in skus:
        pattern_type = sku_patterns[sku]
        
        # Base demand level
        base = np.random.randint(50, 200)
        
        # Generate daily demand for the entire period
        total_days = historical_days + forecast_horizon
        demand = np.zeros(total_days)
        
        # Get promotions for this SKU
        sku_promos = []
        if not promo_df.empty:
            sku_promos = promo_df[promo_df['sku_id'] == sku].to_dict('records')
        
        for day in range(total_days):
            date = start_date + timedelta(days=day)
            date_str = date.strftime('%Y-%m-%d')
            
            # Base value
            value = base
            
            # Apply pattern effects based on SKU type
            if pattern_type == "day_pattern" or pattern_type == "promo_holiday":
                # Strong day-of-week pattern (50-100% swing)
                dow = date.weekday()
                if dow == 0:  # Monday
                    value *= 0.6  # Strong drop
                elif dow == 5:  # Saturday
                    value *= 1.8  # Strong spike
                elif dow == 6:  # Sunday
                    value *= 1.5  # Moderate spike
                
                # Add day-of-month pattern (end of month spike)
                if date.day >= 28:
                    value *= 1.4
            
            # Apply holiday effects
            if date_str in holiday_dates:
                if pattern_type == "promo_holiday":
                    # Strong holiday spike
                    value *= 2.5
                else:
                    # Moderate holiday spike for other patterns
                    value *= 1.5
            
            # Apply promotion effects
            is_promo_day = False
            for promo in sku_promos:
                promo_start = pd.to_datetime(promo['start_date'])
                promo_end = pd.to_datetime(promo['end_date'])
                if promo_start <= date <= promo_end:
                    is_promo_day = True
                    break
            
            if is_promo_day:
                if pattern_type == "promo_holiday":
                    # Strong promotion spike
                    value *= 2.2
                else:
                    # Moderate promotion spike for other patterns
                    value *= 1.4
            
            # Add noise (less noise to make patterns clearer)
            noise = np.random.normal(0, value * 0.05)
            demand[day] = max(0, round(value + noise))
        
        true_demand[sku] = demand
        
    return true_demand


def _generate_forecast_actual_data(
    skus: List[str],
    sku_patterns: Dict[str, str],
    true_demand: Dict[str, np.ndarray],
    start_date: pd.Timestamp,
    historical_days: int,
    forecast_horizon: int,
    holiday_dates: set,
    promo_df: pd.DataFrame
) -> Tuple[List[Dict], List[Dict]]:
    """Generate forecast and actual data with clear patterns."""
    forecast_records = []
    actual_records = []
    
    # Generate forecast data with clear, learnable biases
    for day in range(historical_days - forecast_horizon):
        forecast_date = start_date + timedelta(days=day)
        
        for sku in skus:
            pattern_type = sku_patterns[sku]
            true_values = true_demand[sku]
            
            row = {
                'sku_id': sku,
                'forecast_date': forecast_date.strftime('%Y-%m-%d')
            }
            
            # Configure pattern-specific biases
            if pattern_type == "underbias":
                # Consistent underbias of 15-25%
                bias_tendency = -0.2  # Systematic underprediction
                base_error = 0.15
            elif pattern_type == "promo_holiday":
                # Usually accurate but fails on promos/holidays
                bias_tendency = 0.05
                base_error = 0.1
            elif pattern_type == "day_pattern":
                # Misses day-of-week patterns
                bias_tendency = 0.0
                base_error = 0.1
            
            # Add MAPE metrics
            row['ml_mape_7d'] = base_error
            row['ml_mape_30d'] = base_error * 1.5
            row['ml_bias_7d'] = bias_tendency
            row['ml_bias_30d'] = bias_tendency * 1.2
            
            # Get promotions for this SKU
            sku_promos = []
            if not promo_df.empty:
                sku_promos = promo_df[promo_df['sku_id'] == sku].to_dict('records')
            
            # Generate forecasts for each day in the horizon
            for horizon_day in range(1, forecast_horizon + 1):
                target_day = day + horizon_day
                target_date = forecast_date + timedelta(days=horizon_day)
                target_date_str = target_date.strftime('%Y-%m-%d')
                
                if target_day < len(true_values):
                    true_value = true_values[target_day]
                    
                    # Start with baseline forecast (without pattern effects)
                    forecast_value = true_value
                    
                    # Check if this is a holiday
                    is_holiday = target_date_str in holiday_dates
                    
                    # Check if this is a promotion
                    is_promo = False
                    for promo in sku_promos:
                        promo_start = pd.to_datetime(promo['start_date'])
                        promo_end = pd.to_datetime(promo['end_date'])
                        if promo_start <= target_date <= promo_end:
                            is_promo = True
                            break
                    
                    # Apply pattern-specific biases
                    if pattern_type == "underbias":
                        # Consistent underprediction
                        forecast_value *= (1.0 + bias_tendency)
                    elif pattern_type == "promo_holiday":
                        # Apply strong underprediction for holidays and promos
                        if is_holiday:
                            forecast_value *= 0.5  # Severely underpredict holidays
                        elif is_promo:
                            forecast_value *= 0.6  # Severely underpredict promos
                    elif pattern_type == "day_pattern":
                        # Not sensitive to day patterns
                        dow = target_date.weekday()
                        if dow == 0:  # Monday
                            forecast_value *= 0.9  # Doesn't fully capture Monday drop
                        elif dow >= 5:  # Weekend
                            forecast_value *= 0.7  # Severely underpredicts weekend spike
                        
                        # For day_pattern SKUs, we also make holidays and end-of-month less accurate
                        if is_holiday:
                            forecast_value *= 0.8  # Underpredicts holidays somewhat
                        if target_date.day >= 28:
                            forecast_value *= 0.9  # Underpredicts month-end
                    
                    # Add some random noise
                    forecast_value = max(0, round(forecast_value * (1 + np.random.normal(0, 0.05))))
                    
                    # Store forecast
                    row[f'ml_day_{horizon_day}'] = forecast_value
            
            forecast_records.append(row)
    
    # Generate actual data
    for day in range(historical_days):
        date = start_date + timedelta(days=day)
        
        for sku in skus:
            # Get true demand for this SKU and day
            if day < len(true_demand[sku]):
                actual_value = true_demand[sku][day]
                
                actual_records.append({
                    'sku_id': sku,
                    'date': date.strftime('%Y-%m-%d'),
                    'actual_value': actual_value,
                    'pattern_type': sku_patterns[sku]  # Store pattern type for analysis
                })
    
    return forecast_records, actual_records
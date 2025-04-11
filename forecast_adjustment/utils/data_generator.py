"""
Data Generator for Historical Forecast Training - Creates historical forecasts and actuals
for training the forecast adjustment system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os


def generate_historical_forecast_dataset(
    num_skus=20,
    forecast_horizon=14,
    historical_days=120,
    start_date=None,
    output_dir="data",
    logger=None):
    """
    Generate a dataset of historical forecasts and actuals for training.
    
    Args:
        num_skus: Number of SKUs to generate
        forecast_horizon: Number of days in each forecast horizon
        historical_days: Number of days of historical data to generate
        start_date: Starting date (defaults to 120 days ago)
        output_dir: Directory to save data
        logger: Logger instance
        
    Returns:
        Tuple of (forecast_data, actual_data, holiday_data, promotion_data)
    """
    if logger is None:
        logger = logging.getLogger("DataGenerator")
        
    logger.info(f"Generating historical forecast dataset with {num_skus} SKUs over {historical_days} days")
    
    if start_date is None:
        # Default to historical_days ago
        start_date = datetime.now() - timedelta(days=historical_days)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate SKU IDs
    skus = [f"SKU_{i:04d}" for i in range(num_skus)]
    
    # Generate true demand patterns
    true_demand = {}
    for sku in skus:
        # Randomly select a pattern type
        pattern_type = np.random.choice(['stable', 'trend', 'seasonal', 'volatile'])
        
        # Base demand level
        if pattern_type == 'stable':
            base = np.random.randint(50, 200)
        elif pattern_type == 'trend':
            base = np.random.randint(30, 150)
        elif pattern_type == 'seasonal':
            base = np.random.randint(70, 250)
        else:  # volatile
            base = np.random.randint(20, 100)
        
        # Generate daily demand for the entire period
        total_days = historical_days + forecast_horizon
        demand = np.zeros(total_days)
        
        for day in range(total_days):
            date = start_date + timedelta(days=day)
            
            # Base value
            value = base
            
            # Apply pattern effects
            if pattern_type == 'trend':
                # Linear trend
                trend_factor = 1.0 + 0.002 * day
                value *= trend_factor
            elif pattern_type == 'seasonal':
                # Weekly seasonality
                dow = date.weekday()
                if dow >= 5:  # Weekend
                    value *= 1.4
                elif dow == 4:  # Friday
                    value *= 1.2
                elif dow == 0:  # Monday
                    value *= 0.8
            elif pattern_type == 'volatile':
                # Random fluctuations
                value *= np.random.uniform(0.7, 1.3)
            
            # Add random noise
            noise = np.random.normal(0, value * 0.1)
            demand[day] = max(0, round(value + noise))
        
        true_demand[sku] = demand
    
    # Generate forecast data (forecasts made on each historical day)
    forecast_records = []
    
    # For each historical day, generate a forecast for the next N days
    for day in range(historical_days - forecast_horizon):
        forecast_date = start_date + timedelta(days=day)
        
        for sku in skus:
            # Get true demand for this SKU
            true_values = true_demand[sku]
            
            # Create forecast row
            row = {
                'sku_id': sku,
                'forecast_date': forecast_date.strftime('%Y-%m-%d')
            }
            
            # Forecast metrics - more accurate for more recent and stable SKUs
            base_error = np.random.uniform(0.1, 0.3)
            
            # Add bias tendency (some SKUs tend to over/under forecast)
            bias_tendency = np.random.uniform(-0.2, 0.2)
            
            # Add MAPE metrics
            row['ml_mape_7d'] = base_error
            row['ml_mape_30d'] = base_error * 1.5
            row['ml_bias_7d'] = bias_tendency
            row['ml_bias_30d'] = bias_tendency * 1.2
            
            # Generate forecasts for each day in the horizon
            for horizon_day in range(1, forecast_horizon + 1):
                target_day = day + horizon_day
                
                if target_day < len(true_values):
                    true_value = true_values[target_day]
                    
                    # Error increases with horizon
                    horizon_factor = 1.0 + (horizon_day / forecast_horizon) * 0.5
                    
                    # Calculate forecast with bias and noise
                    error_level = base_error * horizon_factor
                    bias = bias_tendency * horizon_factor
                    
                    # Add noise based on error level
                    noise = np.random.normal(bias * true_value, error_level * true_value)
                    forecast_value = max(0, round(true_value + noise))
                    
                    # Store forecast
                    row[f'ml_day_{horizon_day}'] = forecast_value
            
            forecast_records.append(row)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame(forecast_records)
    
    # Generate actual data
    actual_records = []
    
    for day in range(historical_days):
        date = start_date + timedelta(days=day)
        
        for sku in skus:
            # Get true demand for this SKU and day
            if day < len(true_demand[sku]):
                actual_value = true_demand[sku][day]
                
                actual_records.append({
                    'sku_id': sku,
                    'date': date.strftime('%Y-%m-%d'),
                    'actual_value': actual_value
                })
    
    # Create actual DataFrame
    actual_df = pd.DataFrame(actual_records)
    
    # Generate holiday data
    holiday_records = []
    
    # Randomly select some holiday dates
    num_holidays = 5
    holiday_days = np.random.choice(range(historical_days), size=num_holidays, replace=False)
    
    holiday_names = ["New Year's Day", "Independence Day", "Thanksgiving", "Black Friday", "Christmas"]
    
    for i, day in enumerate(sorted(holiday_days)):
        date = start_date + timedelta(days=int(day))
        holiday_records.append({
            'date': date.strftime('%Y-%m-%d'),
            'holiday_name': holiday_names[i % len(holiday_names)]
        })
    
    # Create holiday DataFrame
    holiday_df = pd.DataFrame(holiday_records)
    
    # Generate promotion data
    promo_records = []
    
    # Randomly select some SKUs for promotions
    promo_skus = np.random.choice(skus, size=int(len(skus) * 0.3), replace=False)
    
    for sku in promo_skus:
        # Generate 1-3 promotions
        num_promos = np.random.randint(1, 4)
        
        for _ in range(num_promos):
            # Random 3-7 day promotion
            promo_length = np.random.randint(3, 8)
            promo_start_day = np.random.randint(0, historical_days - promo_length)
            
            start_date_promo = start_date + timedelta(days=promo_start_day)
            end_date_promo = start_date_promo + timedelta(days=promo_length - 1)
            
            promo_records.append({
                'sku_id': sku,
                'start_date': start_date_promo.strftime('%Y-%m-%d'),
                'end_date': end_date_promo.strftime('%Y-%m-%d'),
                'promo_type': np.random.choice(["Price Discount", "BOGO", "Bundle"])
            })
    
    # Create promotion DataFrame
    promo_df = pd.DataFrame(promo_records)
    
    # Save datasets to CSV
    forecast_df.to_csv(os.path.join(output_dir, 'historical_forecasts.csv'), index=False)
    actual_df.to_csv(os.path.join(output_dir, 'historical_actuals.csv'), index=False)
    holiday_df.to_csv(os.path.join(output_dir, 'holidays.csv'), index=False)
    promo_df.to_csv(os.path.join(output_dir, 'promotions.csv'), index=False)
    
    logger.info(f"Generated {len(forecast_df)} historical forecasts for {num_skus} SKUs")
    logger.info(f"Generated {len(actual_df)} actual values for {num_skus} SKUs")
    logger.info(f"Generated {len(holiday_df)} holidays")
    logger.info(f"Generated {len(promo_df)} promotions")
    
    return forecast_df, actual_df, holiday_df, promo_df
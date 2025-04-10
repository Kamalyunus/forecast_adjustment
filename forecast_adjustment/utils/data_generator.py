"""
Data Generator Module - Functions for generating sample data for forecast adjustment
with support for calendar effects, holidays, and promotions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os


def generate_forecast_data(num_skus: int = 50, 
                          forecast_days: int = 14, 
                          start_date: Optional[datetime] = None,
                          output_file: Optional[str] = None,
                          logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Generate sample forecast data with weekly patterns.
    
    Args:
        num_skus: Number of SKUs to generate
        forecast_days: Number of days to forecast
        start_date: Starting date for the forecast
        output_file: Optional path to save CSV output
        logger: Optional logger instance
        
    Returns:
        DataFrame containing generated forecast data
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger("DataGenerator")
    
    logger.info(f"Generating forecast data for {num_skus} SKUs and {forecast_days} days")
    
    # Set start date if not provided
    if start_date is None:
        start_date = datetime.now()
    
    # Create SKU IDs
    sku_ids = [f"SKU_{i:04d}" for i in range(num_skus)]
    
    # Generate forecasts with different patterns
    forecast_data = []
    
    for i, sku_id in enumerate(sku_ids):
        # Create base demand with seasonal patterns
        if i % 4 == 0:
            # High demand SKUs with weekend peaks
            base_demand = np.random.randint(50, 150)
            weekly_pattern = np.array([0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.2])  # Weekend peaks
        elif i % 4 == 1:
            # Medium demand SKUs with weekday peaks
            base_demand = np.random.randint(20, 60)
            weekly_pattern = np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.9])  # Weekday peaks
        elif i % 4 == 2:
            # Low demand SKUs with flat pattern
            base_demand = np.random.randint(5, 20)
            weekly_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Flat pattern
        else:
            # Very low demand SKUs with strong weekend peaks
            base_demand = np.random.randint(1, 5)
            weekly_pattern = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 1.0])  # Strong weekend peaks
        
        # Create row with SKU info
        row = {'sku_id': sku_id}
        
        # Generate ML forecasts
        for day in range(1, forecast_days + 1):
            # Calculate date
            forecast_date = start_date + timedelta(days=day-1)
            # Add weekly seasonality
            day_of_week = forecast_date.weekday()
            seasonality = weekly_pattern[day_of_week]
            
            # Add trend component
            trend = 1.0 + 0.01 * day if i % 3 == 0 else 1.0
            
            # Generate forecast with some noise
            ml_forecast = base_demand * seasonality * trend * np.random.normal(1.0, 0.05)
            row[f'ml_day_{day}'] = max(0, round(ml_forecast))
        
        # Add forecast accuracy metrics
        row['ml_mape_7d'] = np.random.uniform(0.15, 0.35)
        row['ml_mape_30d'] = np.random.uniform(0.20, 0.40)
        row['ml_bias_7d'] = np.random.uniform(-0.2, 0.2)
        row['ml_bias_30d'] = np.random.uniform(-0.15, 0.15)
        
        forecast_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(forecast_data)
    
    # Save to CSV if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Forecast data saved to {output_file}")
    
    return df


def generate_historical_data(forecast_data: pd.DataFrame,
                           num_days: int = 60,
                           start_date: Optional[datetime] = None,
                           output_file: Optional[str] = None,
                           logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Generate historical data based on forecast patterns.
    
    Args:
        forecast_data: DataFrame with forecast data
        num_days: Number of historical days to generate
        start_date: Starting date for historical data (works backward)
        output_file: Optional path to save CSV output
        logger: Optional logger instance
        
    Returns:
        DataFrame containing generated historical data
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger("DataGenerator")
    
    logger.info(f"Generating {num_days} days of historical data for {forecast_data['sku_id'].nunique()} SKUs")
    
    # Set start date if not provided
    if start_date is None:
        start_date = datetime.now()
    
    # Historical record will go backwards from start_date
    historical_records = []
    
    # Extract ML forecast columns
    ml_cols = [col for col in forecast_data.columns if col.startswith('ml_day_')]
    
    for _, sku_row in forecast_data.iterrows():
        sku_id = sku_row['sku_id']
        
        # Extract forecast pattern for this SKU
        ml_values = sku_row[ml_cols].values
        
        # Get accuracy metrics
        mape = sku_row['ml_mape_30d']
        bias = sku_row['ml_bias_30d']
        
        # Create historical data for the past num_days
        for day in range(num_days):
            # Calculate date (going backwards from start_date)
            history_date = start_date - timedelta(days=num_days - day)
            
            # Use cyclic pattern from forecast with slight variations
            pattern_idx = day % len(ml_values)
            base_value = ml_values[pattern_idx]
            
            # Day of week effect
            dow = history_date.weekday()
            
            # Apply bias, day of week effect, and random noise
            biased_value = base_value * (1 + bias)
            noise_range = biased_value * mape
            actual_value = max(0, biased_value + np.random.uniform(-noise_range, noise_range))
            
            # Store in historical records
            historical_records.append({
                'sku_id': sku_id,
                'sale_date': history_date.strftime('%Y-%m-%d'),
                'quantity': int(actual_value)
            })
    
    # Create DataFrame
    df = pd.DataFrame(historical_records)
    
    # Save to CSV if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Historical data saved to {output_file}")
    
    return df


def generate_holiday_data(start_date: Optional[datetime] = None,
                        num_days: int = 60,
                        num_holidays: int = 5,
                        output_file: Optional[str] = None,
                        logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Generate sample holiday data.
    
    Args:
        start_date: Starting date for the holiday range
        num_days: Number of days to cover
        num_holidays: Number of holidays to generate
        output_file: Optional path to save CSV output
        logger: Optional logger instance
        
    Returns:
        DataFrame containing generated holiday data
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger("DataGenerator")
    
    logger.info(f"Generating {num_holidays} holidays over {num_days} days")
    
    # Set start date if not provided
    if start_date is None:
        start_date = datetime.now()
    
    # List of sample holiday names
    holiday_names = [
        "New Year's Day", "Valentine's Day", "Memorial Day", 
        "Independence Day", "Labor Day", "Halloween", "Thanksgiving", 
        "Black Friday", "Christmas Eve", "Christmas Day", "New Year's Eve"
    ]
    
    # Generate random holiday dates
    holiday_indices = np.random.choice(range(num_days), size=min(num_holidays, num_days), replace=False)
    holiday_indices.sort()  # Sort for chronological order
    
    holidays = []
    
    for idx in holiday_indices:
        holiday_date = start_date + timedelta(days=int(idx))
        
        # Select a random holiday name
        holiday_name = np.random.choice(holiday_names)
        
        holidays.append({
            'date': holiday_date.strftime('%Y-%m-%d'),
            'holiday_name': holiday_name
        })
    
    # Create DataFrame
    df = pd.DataFrame(holidays)
    
    # Save to CSV if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Holiday data saved to {output_file}")
    
    return df


def generate_promotion_data(sku_ids: List[str],
                          start_date: Optional[datetime] = None,
                          num_days: int = 60,
                          promo_ratio: float = 0.2,
                          output_file: Optional[str] = None,
                          logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Generate sample promotion data.
    
    Args:
        sku_ids: List of SKU IDs to use
        start_date: Starting date for the promotion range
        num_days: Number of days to cover
        promo_ratio: Ratio of SKUs that will have promotions
        output_file: Optional path to save CSV output
        logger: Optional logger instance
        
    Returns:
        DataFrame containing generated promotion data
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger("DataGenerator")
    
    logger.info(f"Generating promotion data for {len(sku_ids)} SKUs with {promo_ratio:.0%} promo ratio")
    
    # Set start date if not provided
    if start_date is None:
        start_date = datetime.now()
    
    # Promotion types
    promo_types = ["Price Discount", "BOGO", "Bundle Deal", "Gift with Purchase"]
    
    # Randomly select SKUs for promotions
    num_promo_skus = int(len(sku_ids) * promo_ratio)
    promo_skus = np.random.choice(sku_ids, size=num_promo_skus, replace=False)
    
    promotions = []
    
    for sku in promo_skus:
        # Generate 1-3 promotions per SKU
        num_promos = np.random.randint(1, 4)
        
        for _ in range(num_promos):
            # Random start day
            start_day = np.random.randint(0, num_days - 3)
            # Random duration between 1-5 days
            duration = np.random.randint(1, 6)
            end_day = min(start_day + duration, num_days - 1)
            
            promo_start = start_date + timedelta(days=start_day)
            promo_end = start_date + timedelta(days=end_day)
            
            # Randomly select promotion type
            promo_type = np.random.choice(promo_types)
            
            promotions.append({
                'sku_id': sku,
                'start_date': promo_start.strftime('%Y-%m-%d'),
                'end_date': promo_end.strftime('%Y-%m-%d'),
                'promo_type': promo_type
            })
    
    # Create DataFrame
    df = pd.DataFrame(promotions)
    
    # Save to CSV if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Promotion data saved to {output_file}")
    
    return df


def generate_complete_dataset(num_skus: int = 100,
                             forecast_days: int = 14,
                             history_days: int = 60,
                             num_holidays: int = 5,
                             promo_ratio: float = 0.2,
                             start_date: Optional[datetime] = None,
                             output_dir: str = "data",
                             logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate a complete dataset with forecast, historical, holiday, and promotion data.
    
    Args:
        num_skus: Number of SKUs to generate
        forecast_days: Number of days to forecast
        history_days: Number of historical days
        num_holidays: Number of holidays to generate
        promo_ratio: Ratio of SKUs that will have promotions
        start_date: Starting date (defaults to current date)
        output_dir: Directory to save the generated data
        logger: Optional logger instance
        
    Returns:
        Dictionary of generated DataFrames
    """
    # Set up logger if not provided
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("DataGenerator")
    
    # Set start date if not provided
    if start_date is None:
        start_date = datetime.now()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate forecast data
    forecast_file = os.path.join(output_dir, "forecast_data.csv")
    forecast_data = generate_forecast_data(
        num_skus=num_skus,
        forecast_days=forecast_days,
        start_date=start_date,
        output_file=forecast_file,
        logger=logger
    )
    
    # Generate historical data
    historical_file = os.path.join(output_dir, "historical_data.csv")
    historical_data = generate_historical_data(
        forecast_data=forecast_data,
        num_days=history_days,
        start_date=start_date,
        output_file=historical_file,
        logger=logger
    )
    
    # Generate holiday data
    holiday_file = os.path.join(output_dir, "holiday_data.csv")
    holiday_data = generate_holiday_data(
        start_date=start_date,
        num_days=forecast_days,
        num_holidays=num_holidays,
        output_file=holiday_file,
        logger=logger
    )
    
    # Generate promotion data
    promotion_file = os.path.join(output_dir, "promotion_data.csv")
    promotion_data = generate_promotion_data(
        sku_ids=forecast_data['sku_id'].unique().tolist(),
        start_date=start_date,
        num_days=forecast_days,
        promo_ratio=promo_ratio,
        output_file=promotion_file,
        logger=logger
    )
    
    logger.info(f"Complete dataset generated and saved to {output_dir}")
    
    return {
        'forecast': forecast_data,
        'historical': historical_data,
        'holiday': holiday_data,
        'promotion': promotion_data
    }
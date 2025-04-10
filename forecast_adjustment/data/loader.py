"""
Data Loading Module - Functions for loading various data types for forecast adjustment.
"""

import pandas as pd
import logging
from typing import Optional


def load_data(filepath: str, data_type: str = "forecast", logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load data from a CSV file based on data type.
    
    Args:
        filepath: Path to the CSV file
        data_type: Type of data to load ('forecast', 'historical', 'holiday', 'promotion')
        logger: Optional logger instance
        
    Returns:
        DataFrame containing the loaded data
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger("DataLoader")
    
    logger.info(f"Loading {data_type} data from {filepath}")
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Validate based on data type
        if data_type == "forecast":
            _validate_forecast_data(df, logger)
        elif data_type == "historical":
            _validate_historical_data(df, logger)
        elif data_type == "holiday":
            _validate_holiday_data(df, logger)
        elif data_type == "promotion":
            _validate_promotion_data(df, logger)
        else:
            logger.warning(f"Unknown data type: {data_type}, skipping validation")
        
        logger.info(f"Successfully loaded {len(df)} rows of {data_type} data")
        return df
    
    except Exception as e:
        logger.error(f"Error loading {data_type} data: {str(e)}")
        # Return empty DataFrame if error
        return pd.DataFrame()


def _validate_forecast_data(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Validate forecast data format.
    
    Args:
        df: DataFrame to validate
        logger: Logger instance
    """
    # Check required columns
    if 'sku_id' not in df.columns:
        logger.warning("Forecast data missing 'sku_id' column")
    
    # Check forecast day columns
    ml_cols = [col for col in df.columns if col.startswith('ml_day_')]
    if not ml_cols:
        logger.warning("Forecast data doesn't contain any 'ml_day_X' columns")
    
    # Check accuracy metrics
    accuracy_cols = ['ml_mape_7d', 'ml_mape_30d', 'ml_bias_7d', 'ml_bias_30d']
    missing_metrics = [col for col in accuracy_cols if col not in df.columns]
    if missing_metrics:
        logger.info(f"Forecast data missing optional accuracy metrics: {missing_metrics}")


def _validate_historical_data(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Validate historical data format.
    
    Args:
        df: DataFrame to validate
        logger: Logger instance
    """
    # Check required columns
    required_cols = ['sku_id', 'sale_date', 'quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Historical data missing required columns: {missing_cols}")
    
    # Check date format if column exists
    if 'sale_date' in df.columns:
        try:
            df['sale_date'] = pd.to_datetime(df['sale_date'])
        except:
            logger.warning("Historical data 'sale_date' column is not in a valid date format")


def _validate_holiday_data(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Validate holiday data format.
    
    Args:
        df: DataFrame to validate
        logger: Logger instance
    """
    # Check required columns
    required_cols = ['date', 'holiday_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Holiday data missing required columns: {missing_cols}")
    
    # Check date format if column exists
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            logger.warning("Holiday data 'date' column is not in a valid date format")


def _validate_promotion_data(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Validate promotion data format.
    
    Args:
        df: DataFrame to validate
        logger: Logger instance
    """
    # Check required columns
    required_cols = ['sku_id', 'start_date', 'end_date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Promotion data missing required columns: {missing_cols}")
    
    # Check date formats if columns exist
    if 'start_date' in df.columns:
        try:
            df['start_date'] = pd.to_datetime(df['start_date'])
        except:
            logger.warning("Promotion data 'start_date' column is not in a valid date format")
    
    if 'end_date' in df.columns:
        try:
            df['end_date'] = pd.to_datetime(df['end_date'])
        except:
            logger.warning("Promotion data 'end_date' column is not in a valid date format")
    
    # Check for optional promo_type column
    if 'promo_type' not in df.columns:
        logger.info("Promotion data missing optional 'promo_type' column")
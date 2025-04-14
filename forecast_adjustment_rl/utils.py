"""
Utility functions for the Forecast Adjustment RL system.
Includes metrics, visualization, date handling, and logging utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

# Forecasting Metrics Functions
def calculate_mape(forecasts, actuals):
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        forecasts: Array-like of forecast values
        actuals: Array-like of actual values
        
    Returns:
        MAPE value as float
    """
    if len(forecasts) != len(actuals):
        raise ValueError("Length of forecasts and actuals must be the same")
    
    if len(forecasts) == 0:
        return 0.0
    
    # Convert to numpy arrays for vectorized operations
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-5
    
    # Calculate APE (Absolute Percentage Error)
    ape = np.abs(forecasts - actuals) / (np.abs(actuals) + epsilon)
    
    # Return mean
    return np.mean(ape)

def calculate_bias(forecasts, actuals):
    """
    Calculate bias (positive means overforecasting, negative means underforecasting).
    
    Args:
        forecasts: Array-like of forecast values
        actuals: Array-like of actual values
        
    Returns:
        Bias value as float
    """
    if len(forecasts) != len(actuals):
        raise ValueError("Length of forecasts and actuals must be the same")
    
    if len(forecasts) == 0:
        return 0.0
    
    # Convert to numpy arrays for vectorized operations
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-5
    
    # Calculate bias
    bias = (forecasts - actuals) / (np.abs(actuals) + epsilon)
    
    # Return mean
    return np.mean(bias)

# Date and Calendar Utilities
def get_week_of_month(date):
    """
    Calculate week of month (1-4) for a given date.
    
    Args:
        date: Datetime object
        
    Returns:
        Week of month (1-4)
    """
    day = date.day
    
    if day <= 7:
        return 1
    elif day <= 14:
        return 2
    elif day <= 21:
        return 3
    else:
        return 4

def get_month_week_dates(year, month):
    """
    Get dates for each week of a month.
    
    Args:
        year: Year
        month: Month
        
    Returns:
        List of lists containing dates for each week
    """
    # Get first day of month
    first_day = datetime(year, month, 1)
    
    # Get last day of month
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # Create list of all days in month
    days = [first_day + timedelta(days=i) for i in range((last_day - first_day).days + 1)]
    
    # Group by week of month
    weeks = [[] for _ in range(4)]
    for day in days:
        wom = get_week_of_month(day)
        weeks[wom - 1].append(day)
    
    return weeks

# Visualization Utilities
def plot_adjustment_factors(adjustment_data, category, band, save_dir=None):
    """
    Plot adjustment factors over time.
    
    Args:
        adjustment_data: DataFrame with columns ['date', 'adjustment_factor']
        category: Category name
        band: Band name
        save_dir: Directory to save plot (if None, just shows plot)
        
    Returns:
        Plot figure
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by date
    adj_df = adjustment_data.sort_values('date')
    
    # Plot adjustment factors
    plt.plot(adj_df['date'], adj_df['adjustment_factor'], 'b-o', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    # Format chart
    plt.title(f'Adjustment Factors for {category}-{band} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjustment Factor')
    plt.ylim(0.85, 1.15)
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Add WoM shading if enough dates
    if len(adj_df) > 7:
        for i, date in enumerate(adj_df['date']):
            wom = get_week_of_month(date)
            color = 'lightblue' if wom == 1 else 'white'
            plt.axvspan(adj_df['date'].iloc[max(0, i-1)], date, alpha=0.2, color=color)
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"adjustments_{category}_{band}.png"))
        logger.info(f"Saved adjustment plot to {save_dir}/adjustments_{category}_{band}.png")
    
    return plt.gcf()

def plot_category_metrics(metrics_data, metric_name, save_dir=None):
    """
    Plot metrics by category.
    
    Args:
        metrics_data: Dictionary mapping (category, band) to list of metric values
        metric_name: Name of metric (for title)
        save_dir: Directory to save plot (if None, just shows plot)
        
    Returns:
        Plot figure
    """
    plt.figure(figsize=(12, 8))
    
    for (category, band), metrics in metrics_data.items():
        if len(metrics) > 0:
            if isinstance(metrics[0], dict) and metric_name in metrics[0]:
                # If metrics is a list of dictionaries
                values = [m[metric_name] for m in metrics]
            elif isinstance(metrics[0], (int, float)):
                # If metrics is a list of values
                values = metrics
            else:
                logger.warning(f"Unsupported metrics format for {category}-{band}")
                continue
                
            plt.plot(values, label=f"{category}-{band}")
    
    plt.title(f'{metric_name} by Category-Band')
    plt.xlabel('Time')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{metric_name.lower()}_by_category.png"))
        logger.info(f"Saved metrics plot to {save_dir}/{metric_name.lower()}_by_category.png")
    
    return plt.gcf()

# Normalization Utilities
def min_max_normalize(values, min_val=None, max_val=None):
    """
    Normalize values to range [0, 1] using min-max scaling.
    
    Args:
        values: Array-like of values to normalize
        min_val: Minimum value (if None, uses min of values)
        max_val: Maximum value (if None, uses max of values)
        
    Returns:
        Normalized values
    """
    values = np.array(values)
    
    # Use provided min/max or calculate from data
    min_val = min_val if min_val is not None else np.min(values)
    max_val = max_val if max_val is not None else np.max(values)
    
    # Check if min equals max (constant values)
    if min_val == max_val:
        return np.ones_like(values) * 0.5
    
    # Normalize
    normalized = (values - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(values, mean=None, std=None):
    """
    Normalize values using z-score (mean 0, std 1).
    
    Args:
        values: Array-like of values to normalize
        mean: Mean value (if None, uses mean of values)
        std: Standard deviation (if None, uses std of values)
        
    Returns:
        Normalized values
    """
    values = np.array(values)
    
    # Use provided mean/std or calculate from data
    mean = mean if mean is not None else np.mean(values)
    std = std if std is not None else np.std(values)
    
    # Check if std is zero (constant values)
    if std == 0:
        return np.zeros_like(values)
    
    # Normalize
    normalized = (values - mean) / std
    return normalized

# File and path utilities
def ensure_dir(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def print_config_summary(config):
    """
    Print a summary of the configuration for debugging.
    
    Args:
        config: Configuration dictionary
    """
    print("\n==== Configuration Summary ====")
    
    if isinstance(config, dict):
        # Check for expected structure
        if 'SYSTEM_CONFIG' in config:
            print(f"√ SYSTEM_CONFIG found")
            device = config['SYSTEM_CONFIG'].get('device', 'Not specified')
            print(f"  - Device: {device}")
        else:
            print("× SYSTEM_CONFIG not found")
        
        if 'AGENT_CONFIG' in config:
            print(f"√ AGENT_CONFIG found")
            print(f"  - Learning rate: {config['AGENT_CONFIG'].get('learning_rate', 'Not specified')}")
        else:
            print("× AGENT_CONFIG not found")
            
        # Check for other expected configs
        for config_name in ['ACTION_CONFIG', 'STATE_CONFIG', 'REWARD_CONFIG', 
                            'TRAINING_CONFIG', 'DATA_CONFIG', 'FEATURE_CONFIG']:
            if config_name in config:
                print(f"√ {config_name} found")
            else:
                print(f"× {config_name} not found")
    else:
        print(f"! Config is not a dictionary: {type(config)}")
        
        # Try to extract __dict__ if it's a module
        if hasattr(config, '__dict__'):
            print("  Config appears to be a module, checking __dict__:")
            module_dict = config.__dict__
            for key in ['SYSTEM_CONFIG', 'AGENT_CONFIG', 'ACTION_CONFIG']:
                if key in module_dict:
                    print(f"  √ {key} found in module.__dict__")
                else:
                    print(f"  × {key} NOT found in module.__dict__")
    
    print("==============================\n")
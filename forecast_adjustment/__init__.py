"""
Forecast Adjustment System - A machine learning system for adjusting forecasts
using reinforcement learning with support for calendar effects, holidays, and promotions.
"""

from forecast_adjustment.core.agent import ForecastAgent
from forecast_adjustment.core.environment import ForecastEnvironment
from forecast_adjustment.core.trainer import ForecastTrainer
from forecast_adjustment.data.generator import generate_forecast_dataset

__version__ = "0.1.0"
__all__ = [
    'ForecastAgent', 
    'ForecastEnvironment', 
    'ForecastTrainer',
    'generate_forecast_dataset'
]
"""
Forecast Adjustment System - A machine learning system for adjusting forecasts
using reinforcement learning with support for calendar effects, holidays, and promotions.
"""

from forecast_adjustment.agent import ForecastAgent
from forecast_adjustment.environment import ForecastEnvironment
from forecast_adjustment.trainer import ForecastTrainer

__version__ = "0.1.0"
__all__ = ['ForecastAgent', 'ForecastEnvironment', 'ForecastTrainer']
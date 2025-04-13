"""
Configuration settings for the Forecast Adjustment RL system.
Contains all parameters and settings in one centralized place.
"""

# RL Agent Configuration
AGENT_CONFIG = {
    'learning_rate': 0.001,         # Learning rate for neural networks
    'gamma': 0.95,                  # Discount factor for future rewards
    'batch_size': 64,               # Batch size for training
    'memory_size': 100000,          # Size of experience replay buffer
    'update_frequency': 10,         # How often to update the policy (in days)
    'hidden_size_policy': [128, 64],  # Hidden layer sizes for policy network
    'hidden_size_value': [128, 64],   # Hidden layer sizes for value network
}

# Action Space Configuration
ACTION_CONFIG = {
    'adjustment_factors': [0.9, 0.95, 1.0, 1.05, 1.1]  # Available adjustment factors
}

# Reward Configuration
REWARD_CONFIG = {
    'bias_weight': 0.7,                # Weight for bias improvement in reward
    'mape_weight': 0.3,                # Weight for MAPE improvement in reward
    'consistency_penalty': 0.2,        # Penalty for unnecessary changes
    'flip_flop_penalty': 0.3,          # Penalty for reversing recent adjustments
    'extreme_adjustment_penalty': 0.5  # Penalty for large adjustments that worsen metrics
}

# State Configuration
STATE_CONFIG = {
    'mape_lookback_periods': [1, 4],   # Periods for MAPE calculation (weeks)
    'bias_lookback_periods': [1, 4],   # Periods for bias calculation (weeks)
    'include_week_of_month': True,     # Whether to include WoM indicators
    'include_adjustment_history': True  # Whether to include previous adjustments
}

# Training Configuration
TRAINING_CONFIG = {
    'num_episodes': 50,                # Number of episodes to train
    'episode_length_days': 7,          # Length of each episode in days
    'eval_frequency': 5,               # How often to evaluate (in episodes)
    'save_frequency': 10,              # How often to save the model (in episodes)
    'model_dir': './saved_models'      # Directory to save models
}

# Data Configuration
DATA_CONFIG = {
    'forecasts_file': './data/ml_forecasts.csv',  # Path to ML forecasts data
    'actuals_file': './data/actual_sales.csv',    # Path to actual sales data
    'adjustment_file': './data/adjustments.csv',  # Path to save adjustments
}

# System Configuration
SYSTEM_CONFIG = {
    'random_seed': 42,               # Random seed for reproducibility
    'log_level': 'INFO',             # Logging level
    'device': 'cpu'                  # 'cpu' or 'cuda' for GPU
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    # Category embedding size (if using embeddings)
    'category_embedding_size': 16,
    
    # Features to include in state representation
    'features': [
        'mape_short',              # Short-term MAPE
        'mape_long',               # Long-term MAPE
        'bias_short',              # Short-term bias
        'bias_long',               # Long-term bias
        'mape_trend',              # MAPE trend (improving/worsening)
        'bias_trend',              # Bias trend
        'week_of_month',           # Week of month indicators
        'month_of_year',           # Month seasonality
        'sales_volume',            # Average sales volume
        'sales_volatility',        # Sales volatility
        'band',                    # Band (A, B, C)
        'forecast_momentum',       # ML forecast momentum
        'forecast_revision_rate',  # ML forecast revision rate
        'previous_adjustment',     # Last adjustment factor
        'adjustment_age',          # Days since last adjustment
        'adjustment_success_rate'  # How effective past adjustments were
    ]
}
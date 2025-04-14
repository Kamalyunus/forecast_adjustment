"""
Enhanced configuration settings for the Forecast Adjustment RL system.
Adjusted for larger datasets and deeper neural networks.
"""

# RL Agent Configuration
AGENT_CONFIG = {
    'learning_rate': 0.0005,        # Lower initial learning rate for deeper networks
    'gamma': 0.97,                  # Slightly higher discount factor (more future-focused)
    'batch_size': 256,              # Larger batch size for more stable training with GPU
    'memory_size': 500000,          # Larger replay buffer for more diverse experiences
    'update_frequency': 5,          # Update more frequently
    'hidden_size_policy': [256, 128, 128, 64],  # Deeper network for policy
    'hidden_size_value': [256, 128, 128, 64],   # Deeper network for value function
    'use_prioritized_replay': True,  # Use prioritized experience replay
    'priority_alpha': 0.6,          # Priority exponent
    'priority_beta': 0.4,           # Importance sampling start beta
    'beta_increment': 0.0001,       # Increase beta over time
    'lr_decay': 0.9999,             # Learning rate decay per update
    'min_lr': 0.00005,              # Minimum learning rate
}

# Action Space Configuration
ACTION_CONFIG = {
    # More fine-grained adjustment factors for greater precision
    'adjustment_factors': [0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.15]
}

# Reward Configuration
REWARD_CONFIG = {
    'bias_weight': 0.7,                # Weight for bias improvement in reward
    'mape_weight': 0.3,                # Weight for MAPE improvement in reward
    'consistency_penalty': 0.15,       # Slightly reduced consistency penalty
    'flip_flop_penalty': 0.3,          # Penalty for reversing recent adjustments
    'extreme_adjustment_penalty': 0.4,  # Slightly reduced extreme adjustment penalty
    'sparse_reward_bonus': 0.2,        # Bonus for significant improvements
}

# State Configuration
STATE_CONFIG = {
    'mape_lookback_periods': [1, 2, 4, 8],   # More lookback periods
    'bias_lookback_periods': [1, 2, 4, 8],   # More lookback periods for bias
    'include_week_of_month': True,     # Whether to include WoM indicators
    'include_adjustment_history': True,  # Whether to include previous adjustments
    'include_temporal_features': True,  # Include more temporal features
}

# Training Configuration
TRAINING_CONFIG = {
    'num_episodes': 30,               # More episodes for deeper learning
    'episode_length_days': 14,          # Longer episode length
    'eval_frequency': 5,               # How often to evaluate (in episodes)
    'save_frequency': 10,              # How often to save the model (in episodes)
    'model_dir': './saved_models',     # Directory to save models
    'max_samples_per_wom': 50,         # More samples per Week of Month
    'eval_window_days': 35,            # 5 weeks evaluation window
    'save_best_only': True,            # Only save best model based on evaluation
    'early_stopping_patience': 15,     # Early stopping if no improvement for N episodes
    'checkpoint_top_k': 3,             # Keep top K best models
}

# Data Configuration
DATA_CONFIG = {
    'forecasts_file': './data/ml_forecasts.csv',  # Path to ML forecasts data
    'actuals_file': './data/actual_sales.csv',    # Path to actual sales data
    'adjustment_file': './data/adjustments.csv',  # Path to save adjustments
    'synthetic_data_size': 'large',    # Generate large synthetic dataset
    'test_train_split': 0.2,           # Reserve 20% of data for testing
    'validation_split': 0.1,           # 10% of training data for validation
}

# System Configuration
SYSTEM_CONFIG = {
    'random_seed': 42,               # Random seed for reproducibility
    'log_level': 'INFO',             # Logging level
    'device': 'cuda',                # Use GPU by default
    'mixed_precision': True,         # Enable mixed precision training
    'num_workers': 4,                # Number of workers for data loading
    'pin_memory': True,              # Pin memory for faster data transfer to GPU
    'parallel_processing': True,     # Enable parallel processing where possible
    'profiling': False,              # Enable performance profiling
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    # Category embedding size (if using embeddings)
    'category_embedding_size': 32,   # Larger embeddings
    
    # Features to include in state representation - expanded set
    'features': [
        'mape_short',              # Short-term MAPE
        'mape_medium',             # Medium-term MAPE
        'mape_long',               # Long-term MAPE
        'bias_short',              # Short-term bias
        'bias_medium',             # Medium-term bias
        'bias_long',               # Long-term bias
        'mape_trend',              # MAPE trend (improving/worsening)
        'bias_trend',              # Bias trend
        'week_of_month',           # Week of month indicators
        'month_of_year',           # Month seasonality
        'day_of_week',             # Day of week indicators
        'is_weekend',              # Weekend indicator
        'is_holiday',              # Holiday indicator
        'is_quarter_end',          # Quarter end indicator
        'sales_volume',            # Average sales volume
        'sales_volatility',        # Sales volatility
        'sales_trend',             # Sales trend
        'band',                    # Band (A, B, C)
        'forecast_momentum',       # ML forecast momentum
        'forecast_revision_rate',  # ML forecast revision rate
        'forecast_volatility',     # Volatility in ML forecasts
        'previous_adjustment',     # Last adjustment factor
        'previous_adjustment_effect', # Effect of previous adjustment
        'adjustment_age',          # Days since last adjustment
        'adjustment_success_rate', # How effective past adjustments were
        'category_bias_pattern',   # Category-specific bias pattern embedding
    ],
    
    # Feature normalization method
    'normalization': 'z_score',    # Z-score normalization by default
    'outlier_handling': 'clip',    # Clip outliers to min/max
}
# Forecast Adjustment System

A machine learning system for adjusting forecasts using reinforcement learning with support for calendar effects, holidays, and promotions. The system learns to apply adjustment factors to ML-generated forecasts to improve forecast accuracy.

## Overview

The Forecast Adjustment System uses Q-learning with linear function approximation to learn optimal adjustment factors for each SKU based on forecast patterns, historical accuracy, and contextual information like calendar effects, holidays, and promotions.

## Key Features

- **Calendar Effects**: Automatically detects and adjusts for day-of-week patterns and weekend effects
- **Holiday Handling**: Learns specific adjustment patterns for holidays
- **Promotion Awareness**: Adapts forecasts based on promotional activity
- **SKU-Specific Learning**: Customizes adjustments for each product's unique demand patterns
- **Comprehensive Metrics**: Tracks improvements in MAPE (Mean Absolute Percentage Error) and bias

## Project Structure

```
forecast_adjustment/
│
├── forecast_adjustment/            # Main package directory
│   ├── __init__.py                 # Package initialization
│   ├── agent.py                    # Enhanced linear agent implementation
│   ├── environment.py              # Enhanced forecast environment implementation
│   ├── trainer.py                  # Enhanced trainer implementation
│   ├── cli.py                      # Command line interface
│   │
│   ├── utils/                      # Utility functions and helpers
│   │   ├── __init__.py
│   │   └── data_generator.py       # Data generation with calendar effects
│   │
│   └── data/                       # Data handling and processing
│       ├── __init__.py
│       └── loader.py               # Data loading and validation utilities
│
└── examples/                       # Example scripts
    ├── basic_example.py            # Basic usage example
    └── calendar_comparison.py      # Comparison of different calendar effects
```

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- Pandas
- Matplotlib
- tqdm

### Install from source

```bash
git clone https://github.com/kamalyunus/forecast-adjustment.git
cd forecast-adjustment
pip install -e .
```

## Command Line Usage

The system provides a user-friendly command-line interface with various modes of operation:

### Generate Sample Data

```bash
forecast-adjust --mode generate-data --output-dir data --num-skus 100 --forecast-days 14
```

### Train a Model

```bash
forecast-adjust --mode train \
  --forecast-file data/forecast_data.csv \
  --historical-file data/historical_data.csv \
  --holiday-file data/holiday_data.csv \
  --promotion-file data/promotion_data.csv \
  --output-dir output \
  --episodes 100 \
  --optimize-for both
```

### Evaluate a Model

```bash
forecast-adjust --mode evaluate \
  --forecast-file data/forecast_data.csv \
  --historical-file data/historical_data.csv \
  --holiday-file data/holiday_data.csv \
  --promotion-file data/promotion_data.csv \
  --model-path output/models/final_model.pkl \
  --output-dir output/evaluation
```

### Generate Adjusted Forecasts

```bash
forecast-adjust --mode adjust \
  --forecast-file data/forecast_data.csv \
  --holiday-file data/holiday_data.csv \
  --promotion-file data/promotion_data.csv \
  --model-path output/models/final_model.pkl \
  --output-dir output/adjusted \
  --start-date 2023-01-01
```

## Python API Usage

Here's a simple example of how to use the forecast adjustment system programmatically:

```python
from forecast_adjustment import ForecastAgent, ForecastEnvironment, ForecastTrainer
from forecast_adjustment.utils.data_generator import generate_complete_dataset

# Generate sample data
datasets = generate_complete_dataset(
    num_skus=50,
    forecast_days=14,
    output_dir="data"
)

# Create environment
env = ForecastEnvironment(
    forecast_data=datasets['forecast'],
    historical_data=datasets['historical'],
    holiday_data=datasets['holiday'],
    promotion_data=datasets['promotion'],
    start_date="2023-01-01"
)

# Create agent
feature_dims = env.get_feature_dims()
agent = ForecastAgent(
    feature_dim=feature_dims[-1],
    learning_rate=0.01,
    gamma=0.99
)

# Create trainer
trainer = ForecastTrainer(
    agent=agent,
    environment=env,
    output_dir="output",
    num_episodes=100
)

# Train the agent
train_metrics = trainer.train()

# Generate adjusted forecasts
adjustments = trainer.generate_adjusted_forecasts(num_days=14)
adjustments.to_csv("adjusted_forecasts.csv", index=False)
```

## Input Data Format

The system works with the following data formats:

### Forecast Data

A CSV file with the following columns:
- `sku_id`: Identifier for each product/SKU
- `ml_day_1`, `ml_day_2`, etc.: ML forecasts for each day
- `ml_mape_7d`, `ml_mape_30d`: Historical MAPE metrics (optional)
- `ml_bias_7d`, `ml_bias_30d`: Historical bias metrics (optional)

### Historical Data

A CSV file with the following columns:
- `sku_id`: Identifier for each product/SKU
- `sale_date`: Date of the sale
- `quantity`: Actual sales quantity

### Holiday Data

A CSV file with the following columns:
- `date`: Date of the holiday
- `holiday_name`: Name of the holiday

### Promotion Data

A CSV file with the following columns:
- `sku_id`: Identifier for each product/SKU
- `start_date`: Start date of the promotion
- `end_date`: End date of the promotion
- `promo_type`: Type of promotion (optional)

## Adjustment Factors

The system learns to select from a range of adjustment factors for each forecast:
- 0.5× (reduce forecast by 50%)
- 0.6× (reduce forecast by 40%)
- 0.7× (reduce forecast by 30%)
- 0.8× (reduce forecast by 20%)
- 0.9× (reduce forecast by 10%)
- 1.0× (no adjustment)
- 1.1× (increase forecast by 10%)
- 1.2× (increase forecast by 20%)
- 1.3× (increase forecast by 30%)
- 1.5× (increase forecast by 50%)
- 2.0× (double the forecast)

The wider range of factors allows the system to handle more extreme adjustments needed for promotions and special events.

## Performance Metrics

The system tracks several key metrics:

- **MAPE Improvement**: Reduction in Mean Absolute Percentage Error
- **Bias Improvement**: Reduction in systematic forecast bias
- **Context-Specific Metrics**: Separate tracking for holidays, promotions, weekends, and weekdays

## Advanced Usage

### Optimization Targets

You can configure the system to optimize for different metrics:
```python
# Optimize for MAPE reduction
env = ForecastEnvironment(..., optimize_for="mape")

# Optimize for bias reduction
env = ForecastEnvironment(..., optimize_for="bias")

# Optimize for both metrics (default)
env = ForecastEnvironment(..., optimize_for="both")
```

### Custom Adjustment Factors

You can specify custom adjustment factors if needed:
```python
# Define custom adjustment factors
custom_factors = [0.4, 0.7, 1.0, 1.3, 1.6, 2.0]

# Create agent with custom factors
agent = ForecastAgent(..., adjustment_factors=custom_factors)
```

### Saving and Loading Models

Models can be saved and loaded for later use:
```python
# Save the model
agent.save("models/my_model.pkl")

# Load the model
loaded_agent = ForecastAgent.load("models/my_model.pkl")
```

## Visualization

The system generates several visualizations to help understand the adjustment patterns:

- Training progress plots
- Evaluation summary with context-specific metrics
- Adjustment factor distributions
- Day-of-week patterns
- Holiday vs. non-holiday comparisons
- Promotion vs. non-promotion comparisons

All visualizations are automatically saved to the specified output directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This system is inspired by research in reinforcement learning for time series forecasting and demand planning applications.
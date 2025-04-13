# Forecast Adjustment RL System

A reinforcement learning (RL) solution for automating forecast adjustments at scale.

## Overview

This system applies RL to automate the process of making forecast adjustments, particularly focusing on handling systematic biases such as Week-of-Month (WoM) effects and category-specific patterns. The RL agent learns when and how to adjust forecasts to reduce bias while maintaining or improving forecast accuracy (MAPE).

## Project Structure

```
forecast-adjustment-rl/
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── config.py                       # Configuration settings
│
├── data/                           # Data handling
│   ├── __init__.py
│   ├── data_loader.py              # Load forecasts and actuals
│   └── feature_engineering.py      # Feature creation
│
├── models/                         # RL models
│   ├── __init__.py
│   ├── networks.py                 # Policy and Value networks
│   └── agent.py                    # RL agent implementation
│
├── environment/                    # RL environment
│   ├── __init__.py
│   ├── state.py                    # State representation
│   ├── reward.py                   # Reward calculation
│   └── actions.py                  # Apply adjustments
│
├── training.py                     # Training loop
├── inference.py                    # Make predictions
├── utils.py                        # Utility functions
└── example.py                      # Example script
```

## Features

- **RL-based Adjustment**: Uses policy gradient methods to learn optimal adjustment factors
- **Single Agent Architecture**: One agent handles all category-band combinations, enabling cross-category learning
- **Delayed Reward Handling**: Properly attributes rewards when actuals become available (1-5 weeks later)
- **Rolling Forecast Management**: Handles daily ML forecast updates while maintaining adjustment consistency
- **Pattern Recognition**: Automatically identifies systematic biases like WoM effects
- **Explainable Decisions**: Provides explanations for why specific adjustments were made
- **Confidence Scores**: Reports confidence levels for each adjustment recommendation

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Pandas, NumPy, Matplotlib

### Installation

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

### Running the Example

To run the end-to-end example:

```
python example.py
```

This will:
1. Train the model on synthetic data
2. Run inference to generate adjustments
3. Visualize the Week of Month effect handling

## Component Details

### State Representation

The state includes:
- ML forecast performance metrics (MAPE, Bias)
- Temporal features (Week of Month indicators)
- Category and band encodings
- Previous adjustment information
- ML forecast revision metrics

### Actions

Discrete adjustment factors: [0.9, 0.95, 1.0, 1.05, 1.1]

### Reward Function

Two-part reward system:
- Immediate reward: Penalizes unnecessary changes and flip-flopping
- Delayed reward: Based on improvement in bias and MAPE when actuals become available

## Training Process

The training process:
1. For each day in an episode:
   - Observes the state for each category-band
   - Selects and applies adjustments
   - Calculates immediate rewards
2. When actuals become available:
   - Calculates delayed rewards
   - Updates policy
3. Periodically evaluates and saves the model

## Inference

The inference module:
- Loads a trained model
- Applies adjustments to new forecasts
- Provides explanations for adjustments
- Reports confidence scores

## Configuration

All parameters are centralized in `config.py`:
- RL agent hyperparameters
- State and action space settings
- Reward weights
- Training settings

## Customization

To use with real data:
1. Modify the data loading in `data_loader.py`
2. Adjust the reward function in `reward.py` to match business priorities
3. Tune hyperparameters in `config.py`
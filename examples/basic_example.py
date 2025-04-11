"""
Example script for training a forecast adjustment agent on historical forecasts.
This script demonstrates how to train an agent that properly evaluates full forecast horizons.
"""

import os
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Import components
from forecast_adjustment.agent import ForecastAgent
from forecast_adjustment.environment import ForecastEnvironment
from forecast_adjustment.trainer import ForecastTrainer
from forecast_adjustment.utils.data_generator import generate_historical_forecast_dataset


def setup_logging(log_file=None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger("ForecastAdjustment")


def analyze_adjustment_patterns(adjustment_history, output_dir):
    """
    Analyze patterns in how adjustments are applied over the forecast horizon.
    
    Args:
        adjustment_history: List of adjustment records
        output_dir: Directory to save analysis outputs
    """
    # Convert to DataFrame
    df = pd.DataFrame(adjustment_history)
    
    # Group by horizon day
    horizon_groups = df.groupby('horizon_day')
    
    # Calculate average adjustment factor by horizon day
    avg_factors = horizon_groups.apply(
        lambda x: x['adjusted_forecast'].mean() / x['original_forecast'].replace(0, 1).mean()
    )
    
    # Plot adjustment factors by horizon day
    plt.figure(figsize=(12, 6))
    plt.bar(avg_factors.index, avg_factors.values)
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.title('Average Adjustment Factor by Horizon Day')
    plt.xlabel('Days in Future')
    plt.ylabel('Adjustment Factor')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjustment_by_horizon.png'))
    plt.close()
    
    # Calculate average MAPE improvement by horizon day (for records with actuals)
    actuals_df = df[df['has_actual']]
    
    if not actuals_df.empty:
        # Calculate MAPE for each record
        actuals_df['original_mape'] = abs(actuals_df['original_forecast'] - actuals_df['actual_value']) / actuals_df['actual_value'].replace(0, 1)
        actuals_df['adjusted_mape'] = abs(actuals_df['adjusted_forecast'] - actuals_df['actual_value']) / actuals_df['actual_value'].replace(0, 1)
        actuals_df['mape_improvement'] = actuals_df['original_mape'] - actuals_df['adjusted_mape']
        
        # Group by horizon day
        mape_by_horizon = actuals_df.groupby('horizon_day')['mape_improvement'].mean()
        
        # Plot MAPE improvement by horizon day
        plt.figure(figsize=(12, 6))
        plt.bar(mape_by_horizon.index, mape_by_horizon.values)
        plt.title('Average MAPE Improvement by Horizon Day')
        plt.xlabel('Days in Future')
        plt.ylabel('MAPE Improvement')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mape_improvement_by_horizon.png'))
        plt.close()
    
    # Calculate action frequency by horizon day
    action_counts = {}
    
    for day, group in horizon_groups:
        action_counts[day] = group['action_idx'].value_counts().to_dict()
    
    # Plot action distributions for select horizon days
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Select days to plot (day 0, 3, 6, 9, 12, 13)
    days_to_plot = [0, 3, 6, 9, 12, 13]
    
    for i, day in enumerate(days_to_plot):
        if day in action_counts:
            # Get action counts and sort by index
            counts = action_counts[day]
            sorted_actions = sorted(counts.items())
            
            if sorted_actions:
                indices, values = zip(*sorted_actions)
                axes[i].bar(indices, values)
                axes[i].set_title(f'Horizon Day {day}')
                axes[i].set_xlabel('Action Index')
                axes[i].set_ylabel('Count')
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distribution_by_horizon.png'))
    plt.close()
    
    return avg_factors


def main():
    """Main function to run the historical forecast training example."""
    # Set up logging
    output_dir = "output/historical_training"
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "example.log"))
    
    logger.info("Starting historical forecast training example")
    
    # Generate historical forecast dataset
    logger.info("Generating historical forecast dataset")
    data_dir = os.path.join(output_dir, "data")
    
    forecast_data, actual_data, holiday_data, promo_data = generate_historical_forecast_dataset(
        num_skus=20,
        forecast_horizon=14,
        historical_days=120,
        output_dir=data_dir,
        logger=logger
    )
    
    # Create environment with historical data
    logger.info("Creating forecast environment with historical data")
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        actual_data=actual_data,
        holiday_data=holiday_data,
        promotion_data=promo_data,
        forecast_horizon=14,
        optimize_for="both",
        logger=logger
    )
    
    # Get feature dimensions
    feature_dims = env.get_feature_dims()
    total_feature_dim = feature_dims[-1]
    
    logger.info(f"Feature dimensions: {feature_dims}")
    
    # Create agent
    logger.info("Creating forecast agent")
    agent = ForecastAgent(
        feature_dim=total_feature_dim,
        action_size=11,
        learning_rate=0.01,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        logger=logger
    )
    
    # Create trainer
    logger.info("Creating forecast trainer")
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=20,
        max_steps=500,  # Large enough to process all historical forecasts
        batch_size=32,
        save_every=5,
        optimize_for="both",
        logger=logger
    )
    
    # Train the agent on historical data
    logger.info("Training forecast agent on historical data")
    start_time = time.time()
    train_metrics = trainer.train(verbose=True)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Display training metrics
    logger.info("Training metrics:")
    logger.info(f"  Average Score (last 5): {np.mean(train_metrics['scores'][-5:]):.2f}")
    logger.info(f"  Average MAPE Improvement (last 5): {np.mean(train_metrics['mape_improvements'][-5:]):.4f}")
    logger.info(f"  Average Bias Improvement (last 5): {np.mean(train_metrics['bias_improvements'][-5:]):.4f}")
    
    # Get adjustment history
    adjustment_history = env.get_adjustment_history()
    
    # Analyze adjustment patterns by horizon day
    logger.info("Analyzing adjustment patterns by horizon day")
    avg_factors = analyze_adjustment_patterns(adjustment_history, output_dir)
    
    # Print summary of adjustment factors by horizon day
    logger.info("Summary of adjustments by horizon day:")
    for day, factor in avg_factors.items():
        logger.info(f"  Horizon day {day}: avg factor = {factor:.4f}")
    
    # Save the trained model
    model_path = os.path.join(output_dir, "models", "historical_trained_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    logger.info(f"Saved trained model to {model_path}")
    
    logger.info("Historical forecast training example completed successfully")


if __name__ == "__main__":
    main()
"""
Basic example demonstrating the forecast adjustment system workflow.
"""

import os
import logging
import matplotlib.pyplot as plt
import pandas as pd

from forecast_adjustment import (
    ForecastAgent, 
    ForecastEnvironment, 
    ForecastTrainer,
    generate_forecast_dataset
)
from forecast_adjustment.utils import visualize_adjustments


def setup_logging(output_dir):
    """Set up logging for the example."""
    log_file = os.path.join(output_dir, "forecast_adjustment.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("ForecastAdjustment")


def main():
    """Run a basic example of the forecast adjustment system."""
    # Setup
    output_dir = "output/basic_example"
    logger = setup_logging(output_dir)
    
    logger.info("Starting basic forecast adjustment example")
    
    # Step 1: Generate sample data
    logger.info("Generating sample forecast data")
    datasets = generate_forecast_dataset(
        num_skus=20,
        forecast_horizon=14,
        historical_days=90,
        output_dir=os.path.join(output_dir, "data"),
        include_bands=True,
        logger=logger
    )
    
    # Step 2: Create the environment
    logger.info("Creating forecast environment")
    env = ForecastEnvironment(
        forecast_data=datasets['forecast'],
        actual_data=datasets['historical'],
        sku_band_data=datasets['sku_band'],
        holiday_data=datasets['holiday'],
        promotion_data=datasets['promotion'],
        forecast_horizon=14,
        optimize_for="both",
        logger=logger
    )
    
    # Step 3: Create the agent
    logger.info("Creating forecast agent")
    feature_dims = env.get_feature_dims()
    agent = ForecastAgent(
        feature_dim=feature_dims[-1],
        action_size=11,  # 11 adjustment factors from 0.5x to 2.0x
        learning_rate=0.005,
        gamma=0.95,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        context_learning=True,
        logger=logger
    )
    
    # Step 4: Define training phases for curriculum learning
    training_phases = [
        {
            'episodes': 50,  # Initial exploration
            'epsilon': 0.9,
            'learning_rate': 0.008,
            'batch_size': 32,
            'description': 'Initial exploration phase'
        },
        {
            'episodes': 100,  # Pattern and band learning
            'epsilon': 0.5,
            'learning_rate': 0.005,
            'batch_size': 64,
            'description': 'Pattern and band learning phase'
        },
        {
            'episodes': 150,  # Refinement
            'epsilon': 0.1,
            'learning_rate': 0.001,
            'batch_size': 128,
            'description': 'Fine-tuning phase'
        }
    ]
    
    # Step 5: Create the trainer with curriculum learning
    logger.info("Creating forecast trainer")
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=300,  # For a quick example
        max_steps=14,
        save_every=25,
        training_phases=training_phases,
        logger=logger
    )
    
    # Step 6: Train the agent
    logger.info("Training forecast agent")
    train_metrics = trainer.train(verbose=True)
    
    # Step 7: Evaluate the trained agent
    logger.info("Evaluating forecast agent")
    eval_metrics = trainer.evaluate(num_episodes=5, verbose=True)
    
    # Step 8: Generate adjusted forecasts
    logger.info("Generating adjusted forecasts")
    adjusted_forecasts = trainer.generate_adjusted_forecasts(num_days=14)
    
    # Save adjusted forecasts
    adjusted_forecasts.to_csv(os.path.join(output_dir, "adjusted_forecasts.csv"), index=False)
    
    # Additional visualization
    visualize_adjustments(
        adjusted_forecasts,
        output_dir=os.path.join(output_dir, "visualizations"),
        logger=logger
    )
    
    # Print summary
    logger.info("Training and evaluation complete")
    logger.info(f"MAPE Improvement: {eval_metrics['avg_mape_improvement']:.4f}")
    logger.info(f"Bias Improvement: {eval_metrics['avg_bias_improvement']:.4f}")
    
    # Print context-specific improvements
    if 'context_metrics' in eval_metrics:
        logger.info("Context-specific improvements:")
        for context, metrics in eval_metrics['context_metrics'].items():
            logger.info(f"  {context}: MAPE Improvement = {metrics['avg_mape_improvement']:.4f}")
    
    # Print band-specific improvements
    if 'band_metrics' in eval_metrics:
        logger.info("Band-specific improvements:")
        for band, metrics in eval_metrics['band_metrics'].items():
            logger.info(f"  Band {band}: MAPE Improvement = {metrics['avg_mape_improvement']:.4f}")
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
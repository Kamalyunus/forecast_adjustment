"""
Streamlined example demonstrating the forecast adjustment system.
"""

import os
import logging
import pandas as pd
from forecast_adjustment import (
    ForecastAgent, 
    ForecastEnvironment, 
    ForecastTrainer,
    generate_forecast_dataset
)


def setup_logging(output_dir):
    """Set up logging."""
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
    """Run streamlined forecast adjustment example."""
    # Setup
    output_dir = "output/basic_example"
    logger = setup_logging(output_dir)
    
    logger.info("Starting forecast adjustment example")
    
    # Generate sample data
    logger.info("Generating sample data")
    datasets = generate_forecast_dataset(
        num_skus=50,
        forecast_horizon=14,
        historical_days=90,
        output_dir=os.path.join(output_dir, "data"),
        include_bands=True,
        logger=logger
    )
    
    # Create environment
    logger.info("Creating environment")
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
    
    # Create agent
    logger.info("Creating agent")
    feature_dims = env.get_feature_dims()
    agent = ForecastAgent(
        feature_dim=feature_dims[-1],
        action_size=11,  # 11 adjustment factors from 0.5x to 2.0x
        learning_rate=0.005,
        gamma=0.95,
        ucb_constant=2.0,
        conservative_factor=0.7,
        context_learning=True,
        logger=logger
    )
    
    # Define training phases
    training_phases = [
        {
            'episodes': 50,
            'ucb_constant': 3.0,
            'learning_rate': 0.008,
            'batch_size': 32,
            'description': 'Exploration phase'
        },
        {
            'episodes': 100,
            'ucb_constant': 2.0,
            'learning_rate': 0.005,
            'batch_size': 64,
            'description': 'Learning phase'
        },
        {
            'episodes': 150,
            'ucb_constant': 0.5,
            'learning_rate': 0.001,
            'batch_size': 128,
            'description': 'Fine-tuning phase'
        }
    ]
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=300,
        max_steps=14,
        save_every=25,
        training_phases=training_phases,
        logger=logger
    )
    
    # Train the agent
    logger.info("Training agent")
    train_metrics = trainer.train(verbose=True)
    
    # Evaluate the agent
    logger.info("Evaluating agent")
    eval_metrics = trainer.evaluate(num_episodes=5, verbose=True)
    
    # Generate adjusted forecasts
    logger.info("Generating adjusted forecasts")
    adjusted_forecasts = trainer.generate_adjusted_forecasts(num_days=14)
    
    # Print summary
    logger.info("Training and evaluation complete")
    logger.info(f"MAPE Improvement: {eval_metrics['avg_mape_improvement']:.4f}")
    logger.info(f"Bias Improvement: {eval_metrics['avg_bias_improvement']:.4f}")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
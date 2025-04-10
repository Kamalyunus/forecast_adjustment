"""
Basic Example Script - Demonstrates the usage of the forecast adjustment system.
"""

import os
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime

# Import components from our forecast adjustment package
from forecast_adjustment.agent import ForecastAgent
from forecast_adjustment.environment import ForecastEnvironment
from forecast_adjustment.trainer import ForecastTrainer
from forecast_adjustment.utils.data_generator import generate_complete_dataset


def setup_logging(log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Logger instance
    """
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


def main():
    """Main function to run the forecast adjustment example."""
    # Set up logging
    output_dir = "output/basic_example"
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "example.log"))
    
    logger.info("Starting forecast adjustment example")
    
    # Generate sample data with calendar effects, holidays, and promotions
    logger.info("Generating sample data")
    data_dir = os.path.join(output_dir, "data")
    start_date = datetime.now()
    
    datasets = generate_complete_dataset(
        num_skus=50,
        forecast_days=14,
        history_days=60,
        num_holidays=5,
        promo_ratio=0.2,
        start_date=start_date,
        output_dir=data_dir,
        logger=logger
    )
    
    # Load the generated data
    forecast_data = datasets['forecast']
    historical_data = datasets['historical']
    holiday_data = datasets['holiday']
    promotion_data = datasets['promotion']
    
    # Create environment
    logger.info("Creating forecast environment")
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        holiday_data=holiday_data,
        promotion_data=promotion_data,
        validation_length=30,
        forecast_horizon=7,
        optimize_for="both",
        start_date=start_date.strftime('%Y-%m-%d'),
        logger=logger
    )
    
    # Get feature dimensions
    feature_dims = env.get_feature_dims()
    total_feature_dim = feature_dims[-1]  # Last item is the total dimension
    
    logger.info(f"Feature dimensions: {feature_dims}")
    
    # Create agent
    logger.info("Creating forecast agent")
    agent = ForecastAgent(
        feature_dim=total_feature_dim,
        action_size=11,  # Will be set by default adjustment factors
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
        num_episodes=50,  # Set lower for the example
        max_steps=14,
        batch_size=32,
        save_every=10,
        optimize_for="both",
        logger=logger
    )
    
    # Train the agent
    logger.info("Training forecast agent")
    start_time = time.time()
    train_metrics = trainer.train(verbose=True)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Display training metrics
    logger.info("Training metrics:")
    logger.info(f"  Average Score (last 20): {np.mean(train_metrics['scores'][-20:]):.2f}")
    logger.info(f"  Average MAPE Improvement (last 20): {np.mean(train_metrics['mape_improvements'][-20:]):.4f}")
    logger.info(f"  Average Bias Improvement (last 20): {np.mean(train_metrics['bias_improvements'][-20:]):.4f}")
    
    # Display context-specific metrics
    logger.info("Context-specific performance (last 20 episodes):")
    
    holiday_mape = np.mean(train_metrics['holiday_metrics']['mape_improvements'][-20:])
    holiday_bias = np.mean(train_metrics['holiday_metrics']['bias_improvements'][-20:])
    logger.info(f"  Holidays: MAPE Imp = {holiday_mape:.4f}, Bias Imp = {holiday_bias:.4f}")
    
    promo_mape = np.mean(train_metrics['promo_metrics']['mape_improvements'][-20:])
    promo_bias = np.mean(train_metrics['promo_metrics']['bias_improvements'][-20:])
    logger.info(f"  Promotions: MAPE Imp = {promo_mape:.4f}, Bias Imp = {promo_bias:.4f}")
    
    weekend_mape = np.mean(train_metrics['weekend_metrics']['mape_improvements'][-20:])
    weekend_bias = np.mean(train_metrics['weekend_metrics']['bias_improvements'][-20:])
    logger.info(f"  Weekends: MAPE Imp = {weekend_mape:.4f}, Bias Imp = {weekend_bias:.4f}")
    
    weekday_mape = np.mean(train_metrics['weekday_metrics']['mape_improvements'][-20:])
    weekday_bias = np.mean(train_metrics['weekday_metrics']['bias_improvements'][-20:])
    logger.info(f"  Weekdays: MAPE Imp = {weekday_mape:.4f}, Bias Imp = {weekday_bias:.4f}")
    
    # Evaluate the agent
    logger.info("Evaluating forecast agent")
    eval_metrics = trainer.evaluate(num_episodes=10, verbose=True)
    
    logger.info("Evaluation metrics:")
    logger.info(f"  Average Score: {eval_metrics['avg_score']:.2f}")
    logger.info(f"  Average MAPE Improvement: {eval_metrics['avg_mape_improvement']:.4f}")
    logger.info(f"  Average Bias Improvement: {eval_metrics['avg_bias_improvement']:.4f}")
    
    # Log top performing SKUs
    logger.info("Top 5 performing SKUs:")
    top_skus = eval_metrics['top_skus'][:5]
    for i, (sku, metrics) in enumerate(top_skus):
        logger.info(f"  {i+1}. {sku}: MAPE Imp = {metrics['overall']['mape_improvement']:.4f}, "
                 f"Bias Imp = {metrics['overall']['bias_improvement']:.4f}")
    
    # Generate adjusted forecasts
    logger.info("Generating adjusted forecasts")
    adjustments = trainer.generate_adjusted_forecasts(num_days=14)
    
    # Save adjustments
    adjustments_path = os.path.join(output_dir, "adjusted_forecasts.csv")
    adjustments.to_csv(adjustments_path, index=False)
    
    logger.info(f"Adjusted forecasts saved to {adjustments_path}")
    
    # Display summary statistics
    logger.info("Adjustment summary:")
    
    # Context-specific factor averages
    contexts = ['Overall']
    avg_factors = [adjustments['adjustment_factor'].mean()]
    
    holiday_adj = adjustments[adjustments['is_holiday'] == True]
    if not holiday_adj.empty:
        contexts.append('Holiday')
        avg_factors.append(holiday_adj['adjustment_factor'].mean())
    
    promo_adj = adjustments[adjustments['is_promotion'] == True]
    if not promo_adj.empty:
        contexts.append('Promotion')
        avg_factors.append(promo_adj['adjustment_factor'].mean())
    
    weekend_adj = adjustments[adjustments['is_weekend'] == True]
    if not weekend_adj.empty:
        contexts.append('Weekend')
        avg_factors.append(weekend_adj['adjustment_factor'].mean())
    
    weekday_adj = adjustments[adjustments['is_weekend'] == False]
    if not weekday_adj.empty:
        contexts.append('Weekday')
        avg_factors.append(weekday_adj['adjustment_factor'].mean())
    
    # Log average adjustment factors by context
    for context, avg_factor in zip(contexts, avg_factors):
        logger.info(f"  {context} average adjustment factor: {avg_factor:.4f}")
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
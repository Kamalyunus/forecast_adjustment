"""
Pattern Learning Example - Demonstrates the RL agent learning specific forecast patterns
and evaluates the improvement for different pattern types.
"""

import os
import numpy as np
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import components
from forecast_adjustment.agent import ForecastAgent
from forecast_adjustment.environment import ForecastEnvironment
from forecast_adjustment.trainer import ForecastTrainer
from forecast_adjustment.utils.data_generator import generate_historical_forecast_dataset
from forecast_adjustment.utils.visualization import (
    visualize_sku_improvements,
    calculate_context_specific_improvements,
    plot_metrics_summary
)


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
    
    return logging.getLogger("PatternLearning")


def generate_pattern_dataset(output_dir, logger):
    """Generate a dataset with clear patterns for the RL agent to learn."""
    logger.info("Generating pattern dataset with clear learning opportunities")
    
    # Generate synthetic data with strong patterns
    forecast_data, actual_data, holiday_data, promo_data = generate_historical_forecast_dataset(
        num_skus=30,  # More SKUs for better pattern learning
        forecast_horizon=14,
        historical_days=180,  # More historical data for better learning
        output_dir=os.path.join(output_dir, "data"),
        logger=logger
    )
    
    # Load SKU pattern information
    sku_pattern_path = os.path.join(output_dir, "data", "sku_patterns.csv")
    if os.path.exists(sku_pattern_path):
        sku_patterns = pd.read_csv(sku_pattern_path)
        logger.info(f"Loaded {len(sku_patterns)} SKU patterns")
        
        # Count by pattern
        pattern_counts = sku_patterns['pattern_type'].value_counts()
        for pattern, count in pattern_counts.items():
            logger.info(f"Pattern '{pattern}': {count} SKUs")
    else:
        logger.warning("SKU pattern information not found")
    
    return forecast_data, actual_data, holiday_data, promo_data


def train_pattern_agent(forecast_data, actual_data, holiday_data, promo_data, output_dir, logger):
    """Train an agent to learn forecast adjustment patterns."""
    logger.info("Creating forecast environment with pattern data")
    
    # Create environment with improved reward structure
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        actual_data=actual_data,
        holiday_data=holiday_data,
        promotion_data=promo_data,
        forecast_horizon=14,
        optimize_for="both",
        reward_scaling=10.0,  # Higher reward scaling for stronger learning signal
        pattern_emphasis=2.0,  # Emphasis on pattern-specific rewards
        logger=logger
    )
    
    # Get feature dimensions
    feature_dims = env.get_feature_dims()
    total_feature_dim = feature_dims[-1]
    
    logger.info(f"Feature dimensions: {feature_dims}")
    
    # Create agent with improved learning capabilities
    logger.info("Creating forecast agent with context-specific learning")
    agent = ForecastAgent(
        feature_dim=total_feature_dim,
        action_size=11,
        learning_rate=0.005,  # Lower learning rate for more stable learning
        gamma=0.95,  # Slightly reduced discount factor to focus more on immediate rewards
        epsilon_start=1.0,
        epsilon_end=0.05,  # Increased minimum exploration
        epsilon_decay=0.998,  # Slower decay for more exploration
        context_learning=True,  # Enable context-specific learning
        logger=logger
    )
    
    # Define curriculum learning phases
    training_phases = [
        {
            'episodes': 100,  # First 20% - exploration phase
            'epsilon': 0.9,
            'learning_rate': 0.008,
            'batch_size': 64,
            'description': 'Initial exploration phase'
        },
        {
            'episodes': 250,  # Next 50% - pattern learning phase
            'epsilon': 0.5,
            'learning_rate': 0.005,
            'batch_size': 128,
            'description': 'Pattern learning phase'
        },
        {
            'episodes': 350,  # Final 30% - refinement phase
            'epsilon': 0.1,
            'learning_rate': 0.001,
            'batch_size': 256,
            'description': 'Fine-tuning phase'
        }
    ]
    
    # Create trainer with curriculum learning
    logger.info("Creating forecast trainer with curriculum learning")
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=700,  # More episodes for better learning
        max_steps=14,
        batch_size=64,
        save_every=50,
        optimize_for="both",
        training_phases=training_phases,
        logger=logger
    )
    
    # Train the agent on historical data
    logger.info("Training forecast agent on pattern data")
    start_time = time.time()
    train_metrics = trainer.train(verbose=True)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Display training metrics
    logger.info("Training metrics:")
    logger.info(f"  Final Score: {train_metrics['scores'][-1]:.2f}")
    logger.info(f"  Final MAPE Improvement: {train_metrics['mape_improvements'][-1]:.4f}")
    logger.info(f"  Final Bias Improvement: {train_metrics['bias_improvements'][-1]:.4f}")
    
    # Context-specific metrics
    logger.info("Context-specific final improvements:")
    logger.info(f"  Holiday MAPE Improvement: {train_metrics['holiday_metrics']['mape_improvements'][-1]:.4f}")
    logger.info(f"  Promotion MAPE Improvement: {train_metrics['promo_metrics']['mape_improvements'][-1]:.4f}")
    logger.info(f"  Weekend MAPE Improvement: {train_metrics['weekend_metrics']['mape_improvements'][-1]:.4f}")
    logger.info(f"  Weekday MAPE Improvement: {train_metrics['weekday_metrics']['mape_improvements'][-1]:.4f}")
    
    # Pattern-specific metrics
    if 'pattern_metrics' in train_metrics:
        logger.info("Pattern-specific final improvements:")
        for pattern, metrics in train_metrics['pattern_metrics'].items():
            if metrics['mape_improvements']:
                logger.info(f"  {pattern} MAPE Improvement: {metrics['mape_improvements'][-1]:.4f}")
    
    return env, agent, trainer, train_metrics


def evaluate_pattern_learning(env, agent, trainer, output_dir, logger):
    """Evaluate how well the agent learned different forecast patterns."""
    logger.info("Evaluating agent's pattern learning capabilities")
    
    # Run a thorough evaluation
    eval_metrics = trainer.evaluate(num_episodes=20, verbose=True)
    
    logger.info("Evaluation metrics:")
    logger.info(f"  Average Score: {eval_metrics['avg_score']:.2f}")
    logger.info(f"  Average MAPE Improvement: {eval_metrics['avg_mape_improvement']:.4f}")
    logger.info(f"  Average Bias Improvement: {eval_metrics['avg_bias_improvement']:.4f}")
    
    # Context-specific evaluation metrics
    logger.info("Context-specific evaluation results:")
    logger.info(f"  Holiday MAPE Improvement: {eval_metrics['context_metrics']['holiday']['avg_mape_improvement']:.4f}")
    logger.info(f"  Promotion MAPE Improvement: {eval_metrics['context_metrics']['promotion']['avg_mape_improvement']:.4f}")
    logger.info(f"  Weekend MAPE Improvement: {eval_metrics['context_metrics']['weekend']['avg_mape_improvement']:.4f}")
    logger.info(f"  Weekday MAPE Improvement: {eval_metrics['context_metrics']['weekday']['avg_mape_improvement']:.4f}")
    
    # Pattern-specific evaluation metrics
    if 'pattern_metrics' in eval_metrics:
        logger.info("Pattern-specific evaluation results:")
        for pattern, metrics in eval_metrics['pattern_metrics'].items():
            logger.info(f"  {pattern} MAPE Improvement: {metrics['avg_mape_improvement']:.4f}")
    
    # Extract top performing SKUs by pattern
    logger.info("Top performing SKUs by pattern:")
    sku_metrics = eval_metrics['sku_metrics']
    
    # Load SKU pattern information
    sku_pattern_path = os.path.join(output_dir, "data", "sku_patterns.csv")
    if os.path.exists(sku_pattern_path):
        sku_patterns = pd.read_csv(sku_pattern_path)
        sku_pattern_dict = dict(zip(sku_patterns['sku_id'], sku_patterns['pattern_type']))
        
        # Group by pattern and get top performers
        pattern_metrics = {}
        for sku, metrics in sku_metrics.items():
            if sku in sku_pattern_dict:
                pattern = sku_pattern_dict[sku]
                if pattern not in pattern_metrics:
                    pattern_metrics[pattern] = []
                
                pattern_metrics[pattern].append((sku, metrics['overall']['mape_improvement']))
        
        # Print top performers by pattern
        for pattern, sku_list in pattern_metrics.items():
            sorted_skus = sorted(sku_list, key=lambda x: x[1], reverse=True)
            logger.info(f"  Pattern '{pattern}' top SKUs:")
            for sku, improvement in sorted_skus[:3]:
                logger.info(f"    {sku}: MAPE Improvement = {improvement:.4f}")
    
    # Generate adjusted forecasts
    logger.info("Generating adjusted forecasts")
    adjusted_forecasts = trainer.generate_adjusted_forecasts(num_days=30)
    
    return eval_metrics, adjusted_forecasts


def visualize_pattern_learning(env, adjusted_forecasts, output_dir, logger):
    """Create visualizations showing pattern learning results."""
    logger.info("Creating pattern learning visualizations")
    
    # Load SKU pattern information
    sku_patterns = {}
    sku_pattern_path = os.path.join(output_dir, "data", "sku_patterns.csv")
    if os.path.exists(sku_pattern_path):
        sku_pattern_df = pd.read_csv(sku_pattern_path)
        sku_patterns = dict(zip(sku_pattern_df['sku_id'], sku_pattern_df['pattern_type']))
    
    # Extract original forecasts from environment history
    adjustment_history = env.get_adjustment_history()
    original_forecasts_df = pd.DataFrame(adjustment_history)
    
    # Check and fix column names in adjustment history
    logger.info(f"Adjustment history columns: {original_forecasts_df.columns.tolist()}")
    
    # Rename columns if needed to match expected names
    column_mapping = {}
    if 'sku' in original_forecasts_df.columns and 'sku_id' not in original_forecasts_df.columns:
        column_mapping['sku'] = 'sku_id'
    if 'target_date' in original_forecasts_df.columns and 'date' not in original_forecasts_df.columns:
        column_mapping['target_date'] = 'date'
    
    if column_mapping:
        original_forecasts_df = original_forecasts_df.rename(columns=column_mapping)
        logger.info(f"Renamed columns: {column_mapping}")
    
    # Format adjusted forecasts for visualization
    if 'date' in adjusted_forecasts.columns:
        date_col = 'date'
    else:
        date_col = 'target_date'
        if date_col in adjusted_forecasts.columns:
            adjusted_forecasts = adjusted_forecasts.rename(columns={date_col: 'date'})
    
    # Add pattern_type to adjusted forecasts if not present
    if 'pattern_type' not in adjusted_forecasts.columns and sku_patterns:
        adjusted_forecasts['pattern_type'] = adjusted_forecasts['sku_id'].map(sku_patterns)
    
    # Select needed columns from adjusted forecasts
    needed_cols = ['sku_id', 'date', 'original_forecast', 'adjusted_forecast', 'pattern_type']
    context_cols = ['is_holiday', 'is_promotion', 'is_weekend']
    available_cols = [col for col in needed_cols + context_cols if col in adjusted_forecasts.columns]
    
    adjusted_forecasts_df = adjusted_forecasts[available_cols].copy()
    
    # Add missing context columns if not present
    for col in context_cols:
        if col not in adjusted_forecasts_df.columns:
            adjusted_forecasts_df[col] = False
            logger.warning(f"Added missing column: {col}")
    
    # Load actual data
    actual_data_path = os.path.join(output_dir, "data", "historical_actuals.csv")
    actuals_df = pd.read_csv(actual_data_path)
    
    # Add pattern_type to actuals if not present
    if 'pattern_type' not in actuals_df.columns and sku_patterns:
        actuals_df['pattern_type'] = actuals_df['sku_id'].map(sku_patterns)
    
    # Load holiday and promotion data
    holiday_data_path = os.path.join(output_dir, "data", "holidays.csv")
    holiday_df = pd.read_csv(holiday_data_path) if os.path.exists(holiday_data_path) else None
    
    promo_data_path = os.path.join(output_dir, "data", "promotions.csv")
    promo_df = pd.read_csv(promo_data_path) if os.path.exists(promo_data_path) else None
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Log data sizes to help debug
    logger.info(f"Original forecasts shape: {original_forecasts_df.shape}")
    logger.info(f"Adjusted forecasts shape: {adjusted_forecasts_df.shape}")
    logger.info(f"Actuals shape: {actuals_df.shape}")
    
    # Create SKU improvement visualizations
    try:
        visualize_sku_improvements(
            original_forecasts=original_forecasts_df,
            adjusted_forecasts=adjusted_forecasts_df,
            actuals=actuals_df,
            sku_patterns=sku_patterns,
            holiday_data=holiday_df,
            promotion_data=promo_df,
            output_dir=vis_dir,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Error in SKU improvement visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Calculate and visualize context-specific metrics
    try:
        metrics = calculate_context_specific_improvements(
            original_forecasts=original_forecasts_df,
            adjusted_forecasts=adjusted_forecasts_df,
            actuals=actuals_df,
            logger=logger
        )
        
        # Plot metrics summary
        plot_metrics_summary(metrics, output_dir=vis_dir, logger=logger)
        
        logger.info(f"Pattern learning visualizations saved to {vis_dir}")
    except Exception as e:
        logger.error(f"Error in calculating metrics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        metrics = {"overall": {"improvement": 0}}
    
    return metrics


def main():
    """Main function to demonstrate pattern learning in forecast adjustment."""
    # Set up output directory and logging
    output_dir = "output/pattern_learning"
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "pattern_learning.log"))
    
    logger.info("Starting pattern learning demonstration")
    
    # Step 1: Generate dataset with clear patterns
    forecast_data, actual_data, holiday_data, promo_data = generate_pattern_dataset(
        output_dir=output_dir,
        logger=logger
    )
    
    # Step 2: Train agent to learn patterns
    env, agent, trainer, train_metrics = train_pattern_agent(
        forecast_data=forecast_data,
        actual_data=actual_data,
        holiday_data=holiday_data,
        promo_data=promo_data,
        output_dir=output_dir,
        logger=logger
    )
    
    # Step 3: Evaluate pattern learning
    eval_metrics, adjusted_forecasts = evaluate_pattern_learning(
        env=env,
        agent=agent,
        trainer=trainer,
        output_dir=output_dir,
        logger=logger
    )
    
    # Step 4: Visualize pattern learning results
    metrics = visualize_pattern_learning(
        env=env,
        adjusted_forecasts=adjusted_forecasts,
        output_dir=output_dir,
        logger=logger
    )
    
    # Display overall performance improvement
    overall_improvement = metrics['overall'].get('improvement', 0)
    logger.info(f"Overall MAPE improvement: {overall_improvement:.2f}%")
    
    # Display pattern-specific improvements
    logger.info("Pattern-specific MAPE improvements:")
    for pattern, pattern_metrics in metrics.get('by_pattern', {}).items():
        improvement = pattern_metrics.get('improvement', 0)
        logger.info(f"  {pattern}: {improvement:.2f}%")
    
    logger.info("Pattern learning demonstration completed successfully")


if __name__ == "__main__":
    main()
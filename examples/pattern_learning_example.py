"""
Example script demonstrating the forecast adjustment system with SKU banding.
This example shows how to use the system to adjust forecasts with differentiated
strategies for high-volume (bands A-B) vs. low-volume (bands D-E) SKUs.
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the components
from forecast_adjustment.agent import ForecastAgent
from forecast_adjustment.environment import ForecastEnvironment
from forecast_adjustment.trainer import ForecastTrainer
from forecast_adjustment.utils.data_generator import generate_historical_forecast_dataset


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


def generate_sample_data(output_dir, logger):
    """Generate sample data with SKU banding information."""
    logger.info("Generating sample data with SKU banding")
    
    # Generate synthetic forecast data using the built-in generator
    forecast_data, actual_data, holiday_data, promo_data = generate_historical_forecast_dataset(
        num_skus=50,
        forecast_horizon=14,
        historical_days=90,
        output_dir=os.path.join(output_dir, "data"),
        logger=logger
    )
    
    # Create SKU banding information (A-E) based on sales volume
    # Extract unique SKUs
    unique_skus = forecast_data['sku_id'].unique()
    
    # Calculate total volume for each SKU to determine banding
    sku_volumes = {}
    for sku in unique_skus:
        sku_data = actual_data[actual_data['sku_id'] == sku]
        sku_volumes[sku] = sku_data['actual_value'].sum() if not sku_data.empty else 0
    
    # Sort SKUs by volume
    sorted_skus = sorted(sku_volumes.items(), key=lambda x: x[1], reverse=True)
    
    # Assign bands (A: top 10%, B: next 20%, C: middle 40%, D: next 20%, E: bottom 10%)
    num_skus = len(sorted_skus)
    band_thresholds = {
        'A': int(num_skus * 0.1),
        'B': int(num_skus * 0.3),
        'C': int(num_skus * 0.7),
        'D': int(num_skus * 0.9),
        'E': num_skus
    }
    
    # Create SKU band mapping
    sku_bands = []
    for i, (sku, volume) in enumerate(sorted_skus):
        if i < band_thresholds['A']:
            band = 'A'
        elif i < band_thresholds['B']:
            band = 'B'
        elif i < band_thresholds['C']:
            band = 'C'
        elif i < band_thresholds['D']:
            band = 'D'
        else:
            band = 'E'
        
        sku_bands.append({'sku_id': sku, 'band': band, 'volume': volume})
    
    # Create and save SKU band DataFrame
    sku_band_df = pd.DataFrame(sku_bands)
    sku_band_path = os.path.join(output_dir, "data", "sku_bands.csv")
    sku_band_df.to_csv(sku_band_path, index=False)
    
    # Log band distribution
    band_counts = sku_band_df['band'].value_counts().sort_index()
    for band, count in band_counts.items():
        logger.info(f"Band {band}: {count} SKUs")
    
    # Return all data
    return forecast_data, actual_data, holiday_data, promo_data, sku_band_df


def train_forecast_model(forecast_data, actual_data, sku_band_data, holiday_data, promo_data, output_dir, logger):
    """Train the forecast adjustment model with SKU banding."""
    logger.info("Setting up environment and agent with SKU banding")
    
    # Create environment with SKU banding
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        actual_data=actual_data,
        sku_band_data=sku_band_data,  # Add SKU banding information
        holiday_data=holiday_data,
        promotion_data=promo_data,
        forecast_horizon=14,
        optimize_for="both",
        reward_scaling=5.0,
        pattern_emphasis=1.5,
        band_emphasis=1.8,  # Emphasis for band-specific rewards
        logger=logger
    )
    
    # Get feature dimensions
    feature_dims = env.get_feature_dims()
    total_feature_dim = feature_dims[-1]
    
    # Create agent with band-specific learning
    agent = ForecastAgent(
        feature_dim=total_feature_dim,
        action_size=11,  # Default 11 adjustment factors from 0.5x to 2.0x
        learning_rate=0.005,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        context_learning=True,  # Enable context-specific learning (including bands)
        logger=logger
    )
    
    # Define curriculum learning phases
    training_phases = [
        {
            'episodes': 100,  # Initial exploration
            'epsilon': 0.9,
            'learning_rate': 0.008,
            'batch_size': 32,
            'description': 'Initial exploration phase'
        },
        {
            'episodes': 200,  # Pattern and band learning
            'epsilon': 0.5,
            'learning_rate': 0.005,
            'batch_size': 64,
            'description': 'Pattern and band learning phase'
        },
        {
            'episodes': 200,  # Refinement
            'epsilon': 0.1,
            'learning_rate': 0.001,
            'batch_size': 128,
            'description': 'Fine-tuning phase'
        }
    ]
    
    # Create trainer with curriculum learning
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=500,
        max_steps=14,
        batch_size=64,
        save_every=50,
        optimize_for="both",
        training_phases=training_phases,
        logger=logger
    )
    
    # Train the agent
    logger.info("Starting training...")
    train_metrics = trainer.train(verbose=True)
    
    logger.info("Training complete")
    logger.info(f"Best MAPE improvement: {train_metrics['best_mape_improvement']:.4f}")
    logger.info(f"Best bias improvement: {train_metrics['best_bias_improvement']:.4f}")
    
    return trainer, train_metrics


def evaluate_model(trainer, output_dir, logger):
    """Evaluate the trained model with band-specific analysis."""
    logger.info("Evaluating model performance with band focus")
    
    # Run evaluation
    eval_metrics = trainer.evaluate(num_episodes=10, verbose=True)
    
    # Log overall performance
    logger.info(f"Overall MAPE improvement: {eval_metrics['avg_mape_improvement']:.4f}")
    logger.info(f"Overall bias improvement: {eval_metrics['avg_bias_improvement']:.4f}")
    
    # Log band-specific performance
    if 'band_metrics' in eval_metrics:
        logger.info("Band-specific performance:")
        for band, metrics in eval_metrics['band_metrics'].items():
            logger.info(f"  Band {band}: MAPE improvement = {metrics['avg_mape_improvement']:.4f}, "
                      f"bias improvement = {metrics['avg_bias_improvement']:.4f}")
    
    # Generate adjusted forecasts for analysis
    logger.info("Generating adjusted forecasts...")
    adjustments = trainer.generate_adjusted_forecasts(num_days=14)
    
    # Analyze adjustment patterns by band
    analyze_band_adjustments(adjustments, output_dir, logger)
    
    return eval_metrics, adjustments


def analyze_band_adjustments(adjustments, output_dir, logger):
    """Analyze adjustment patterns by SKU bands."""
    logger.info("Analyzing adjustment patterns by SKU bands")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Summarize adjustment factors by band
    if 'sku_band' in adjustments.columns:
        # Calculate average adjustment factors
        band_factors = adjustments.groupby('sku_band')['adjustment_factor'].agg(['mean', 'std', 'count']).reset_index()
        
        # Log summary
        logger.info("Adjustment factors by band:")
        for _, row in band_factors.iterrows():
            logger.info(f"  Band {row['sku_band']}: mean = {row['mean']:.3f}, std = {row['std']:.3f}, count = {row['count']}")
        
        # Create band comparison plot
        plt.figure(figsize=(10, 6))
        
        # Calculate percentage of upward and downward adjustments by band
        band_direction = {}
        for band in adjustments['sku_band'].unique():
            band_df = adjustments[adjustments['sku_band'] == band]
            upward = len(band_df[band_df['adjustment_factor'] > 1.0]) / len(band_df) * 100
            no_change = len(band_df[band_df['adjustment_factor'] == 1.0]) / len(band_df) * 100
            downward = len(band_df[band_df['adjustment_factor'] < 1.0]) / len(band_df) * 100
            band_direction[band] = (upward, no_change, downward)
        
        # Plot adjustment direction by band
        bands = sorted(band_direction.keys())
        upward = [band_direction[b][0] for b in bands]
        no_change = [band_direction[b][1] for b in bands]
        downward = [band_direction[b][2] for b in bands]
        
        x = np.arange(len(bands))
        width = 0.25
        
        plt.bar(x - width, upward, width, label='Upward (>1.0)', color='green')
        plt.bar(x, no_change, width, label='No Change (=1.0)', color='gray')
        plt.bar(x + width, downward, width, label='Downward (<1.0)', color='red')
        
        plt.title('Adjustment Direction by SKU Band')
        plt.xlabel('SKU Band')
        plt.ylabel('Percentage of Adjustments')
        plt.xticks(x, bands)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(upward):
            plt.text(i - width, v/2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        for i, v in enumerate(no_change):
            if v > 5:  # Only label if there's enough space
                plt.text(i, v/2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        for i, v in enumerate(downward):
            plt.text(i + width, v/2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        
        plt.savefig(os.path.join(viz_dir, 'band_adjustment_direction.png'))
        plt.close()
        
        # Create adjustment factor distribution plot
        plt.figure(figsize=(12, 8))
        
        # Focus on the extreme bands for clarity
        bands_to_plot = ['A', 'E'] if 'A' in bands and 'E' in bands else bands[:2]
        colors = ['blue', 'red']
        
        for i, band in enumerate(bands_to_plot):
            band_df = adjustments[adjustments['sku_band'] == band]
            plt.hist(band_df['adjustment_factor'], bins=20, alpha=0.7, 
                     range=(0.4, 2.1), label=f'Band {band}', color=colors[i])
        
        plt.title('Adjustment Factor Distribution by Band')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(viz_dir, 'band_adjustment_distribution.png'))
        plt.close()
        
        # Context-specific analysis by band (holidays, promotions, weekends)
        if 'is_holiday' in adjustments.columns and 'is_promotion' in adjustments.columns:
            plt.figure(figsize=(14, 8))
            
            # Calculate average adjustment for different contexts per band
            contexts = []
            band_values = []
            
            for band in bands_to_plot:
                band_df = adjustments[adjustments['sku_band'] == band]
                
                # Regular days
                regular_df = band_df[~band_df['is_holiday'] & ~band_df['is_promotion'] & ~band_df['is_weekend']]
                avg_regular = regular_df['adjustment_factor'].mean() if len(regular_df) > 0 else 1.0
                
                # Holiday days
                holiday_df = band_df[band_df['is_holiday']]
                avg_holiday = holiday_df['adjustment_factor'].mean() if len(holiday_df) > 0 else 1.0
                
                # Promotion days
                promo_df = band_df[band_df['is_promotion']]
                avg_promo = promo_df['adjustment_factor'].mean() if len(promo_df) > 0 else 1.0
                
                # Weekend days (excluding holidays and promos)
                weekend_df = band_df[band_df['is_weekend'] & ~band_df['is_holiday'] & ~band_df['is_promotion']]
                avg_weekend = weekend_df['adjustment_factor'].mean() if len(weekend_df) > 0 else 1.0
                
                # Combine into series
                band_values.append([avg_regular, avg_weekend, avg_holiday, avg_promo])
                contexts.append(band)
            
            # Plot side by side
            x = np.arange(4)  # Four contexts
            width = 0.35
            
            plt.bar(x - width/2, band_values[0], width, label=f'Band {bands_to_plot[0]}', color='blue')
            plt.bar(x + width/2, band_values[1], width, label=f'Band {bands_to_plot[1]}', color='red')
            
            plt.title('Average Adjustment Factor by Context and Band')
            plt.ylabel('Adjustment Factor')
            plt.xticks(x, ['Regular', 'Weekend', 'Holiday', 'Promotion'])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, vals in enumerate(band_values):
                offset = -width/2 if i == 0 else width/2
                for j, v in enumerate(vals):
                    plt.text(j + offset, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
            
            plt.savefig(os.path.join(viz_dir, 'band_context_comparison.png'))
            plt.close()


def main():
    """Main function to demonstrate the forecast adjustment system with SKU banding."""
    # Set up output directory and logging
    output_dir = "output/sku_band_example"
    logger = setup_logging(output_dir)
    
    logger.info("Starting forecast adjustment example with SKU banding")
    
    # Step 1: Generate sample data with SKU banding
    forecast_data, actual_data, holiday_data, promo_data, sku_band_data = generate_sample_data(
        output_dir=output_dir,
        logger=logger
    )
    
    # Step 2: Train model with SKU bands
    trainer, train_metrics = train_forecast_model(
        forecast_data=forecast_data,
        actual_data=actual_data,
        sku_band_data=sku_band_data,
        holiday_data=holiday_data,
        promo_data=promo_data,
        output_dir=output_dir,
        logger=logger
    )
    
    # Step 3: Evaluate and analyze results
    eval_metrics, adjustments = evaluate_model(
        trainer=trainer,
        output_dir=output_dir,
        logger=logger
    )
    
    logger.info("Example completed successfully")
    logger.info(f"Results saved to {output_dir}")
    logger.info("Key findings:")
    
    # Summarize key band-related findings
    if 'band_metrics' in eval_metrics:
        best_band = max(eval_metrics['band_metrics'].items(), 
                       key=lambda x: x[1]['avg_mape_improvement'])[0]
        worst_band = min(eval_metrics['band_metrics'].items(), 
                        key=lambda x: x[1]['avg_mape_improvement'])[0]
        
        logger.info(f"  Best performing band: {best_band} with "
                  f"{eval_metrics['band_metrics'][best_band]['avg_mape_improvement']:.4f} MAPE improvement")
        logger.info(f"  Worst performing band: {worst_band} with "
                  f"{eval_metrics['band_metrics'][worst_band]['avg_mape_improvement']:.4f} MAPE improvement")
    
    # Final summary of band-specific adjustments
    if 'sku_band' in adjustments.columns:
        for band in sorted(adjustments['sku_band'].unique()):
            band_df = adjustments[adjustments['sku_band'] == band]
            upward = len(band_df[band_df['adjustment_factor'] > 1.0]) / len(band_df) * 100
            downward = len(band_df[band_df['adjustment_factor'] < 1.0]) / len(band_df) * 100
            logger.info(f"  Band {band}: {upward:.1f}% upward, {downward:.1f}% downward adjustments")


if __name__ == "__main__":
    main()
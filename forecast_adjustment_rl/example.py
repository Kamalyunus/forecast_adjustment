"""
Example script demonstrating the forecast adjustment RL system.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils import get_week_of_month, plot_adjustment_factors, plot_category_metrics


# Import configuration and components
import config
from data.data_loader import DataProvider
from models.agent import ForecastAdjustmentAgent
from environment.state import StateBuilder
from environment.actions import ActionHandler
from environment.reward import RewardCalculator
from training import ForecastAdjustmentTrainer
from inference import ForecastAdjuster

def setup_logging():
    """Set up logging configuration."""
    log_level = config.SYSTEM_CONFIG['log_level']
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger

def train_model(num_episodes=10):
    """Train the RL model."""
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = ForecastAdjustmentTrainer(config.__dict__)
    
    # Train model
    training_metrics = trainer.train(num_episodes=num_episodes)
    
    logger.info(f"Training complete. Final reward: {training_metrics['final_reward']:.4f}")
    return trainer

def run_inference():
    """Run inference with the trained model."""
    logger.info("Running inference with trained model...")
    
    try:
        # Initialize adjuster
        adjuster = ForecastAdjuster(config.__dict__)
        
        # Get current date
        today = datetime.now()
        
        # Adjust forecasts for top 5 categories
        adjustments = adjuster.adjust_forecasts(
            date=today,
            top_n=5
        )
        
        logger.info(f"Applied {len(adjustments)} adjustments")
        
        # Print adjustment summary
        if not adjustments.empty:
            print("\nAdjustment Summary:")
            print(adjustments[['category', 'band', 'adjustment_factor', 'num_skus', 'confidence']])
            
            # Get detailed explanation for first adjustment
            if len(adjustments) > 0:
                first_adj = adjustments.iloc[0]
                try:
                    explanation = adjuster.explain_adjustment(
                        first_adj['category'],
                        first_adj['band'],
                        date=today
                    )
                    
                    print(f"\nDetailed explanation for {explanation['category']}-{explanation['band']} adjustment:")
                    print(f"Adjustment Factor: {explanation['best_adjustment']:.2f}")
                    print(f"Primary Reason: {explanation['primary_reason']}")
                    print(f"Week of Month: {explanation['week_of_month']}")
                    print(f"Short-term Bias: {explanation['short_bias']:.4f}")
                    print(f"Long-term Bias: {explanation['long_bias']:.4f}")
                    print(f"MAPE: {explanation['mape']:.4f}")
                    print(f"Confidence: {explanation['confidence']:.2f}")
                    
                    # Show adjustment probabilities
                    try:
                        probs = explanation['action_probabilities']
                        factors = sorted([float(k) for k in probs.keys()])
                        values = [probs[str(f)] for f in factors]
                        
                        plt.figure(figsize=(10, 6))
                        plt.bar(factors, values)
                        plt.title(f"Adjustment Probabilities for {explanation['category']}-{explanation['band']}")
                        plt.xlabel("Adjustment Factor")
                        plt.ylabel("Probability")
                        plt.xticks(factors)
                        plt.grid(axis='y')
                        
                        # Save plot
                        os.makedirs('plots', exist_ok=True)
                        plt.savefig(f"plots/{explanation['category']}_{explanation['band']}_probs.png")
                        plt.close()
                        
                        logger.info(f"Saved probability plot to plots/{explanation['category']}_{explanation['band']}_probs.png")
                    except Exception as e:
                        logger.error(f"Error creating probability plot: {str(e)}")
                except Exception as e:
                    logger.error(f"Error getting explanation: {str(e)}")
        else:
            print("No adjustments were applied")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        print("Inference failed - see log for details")

def compare_rl_vs_baseline():
    """Compare forecast performance with and without RL adjustments."""
    logger.info("Comparing RL adjustments vs baseline ML forecasts...")
    
    # Import the utility function
    from utils import get_week_of_month
    
    # Initialize components
    data_provider = DataProvider(config.__dict__)
    adjuster = ForecastAdjuster(config.__dict__)
    
    # Select categories and bands to evaluate
    categories = sorted(data_provider.ml_forecasts['category'].unique())[:3]
    bands = ['A', 'B']
    
    # Import visualization utilities
    try:
        from visualization_utils import plot_comparison_with_without_rl
    except ImportError:
        logger.warning("visualization_utils module not found, skipping visualization")
        plot_comparison_with_without_rl = None
    
    # Simulate 4 weeks of forecasting - one complete month
    start_date = datetime.now()
    dates = []
    for week in range(4):
        # Use middle of each week
        week_date = start_date + timedelta(days=(week*7 + 3))
        dates.append(week_date)
    
    # Initialize metrics storage
    with_rl_metrics = {'mape': [], 'bias': []}
    without_rl_metrics = {'mape': [], 'bias': []}
    
    # For each week, compare performance with and without adjustments
    for week, date in enumerate(dates):
        week_of_month = get_week_of_month(date)  # Use utility function
        
        # For each category-band, get metrics
        week_with_mape = []
        week_with_bias = []
        week_without_mape = []
        week_without_bias = []
        
        for category in categories:
            for band in bands:
                # Get original ML forecast metrics
                ml_mape = data_provider.get_historical_mape(category, band, date, before_adjustment=True)
                ml_bias = data_provider.get_historical_bias(category, band, date, before_adjustment=True)
                
                # Get metrics with RL adjustment
                rl_mape = data_provider.get_historical_mape(category, band, date, before_adjustment=False)
                rl_bias = data_provider.get_historical_bias(category, band, date, before_adjustment=False)
                
                # Store metrics
                week_with_mape.append(rl_mape)
                week_with_bias.append(rl_bias)
                week_without_mape.append(ml_mape)
                week_without_bias.append(ml_bias)
        
        # Average metrics across all category-bands for this week
        with_rl_metrics['mape'].append(np.mean(week_with_mape))
        with_rl_metrics['bias'].append(np.mean(week_with_bias))
        without_rl_metrics['mape'].append(np.mean(week_without_mape))
        without_rl_metrics['bias'].append(np.mean(week_without_bias))
    
    # Calculate overall improvement
    mape_improvement = np.mean(without_rl_metrics['mape']) - np.mean(with_rl_metrics['mape'])
    mape_pct_improvement = mape_improvement / np.mean(without_rl_metrics['mape']) * 100
    
    bias_improvement = np.mean([abs(b) for b in without_rl_metrics['bias']]) - np.mean([abs(b) for b in with_rl_metrics['bias']])
    bias_pct_improvement = bias_improvement / np.mean([abs(b) for b in without_rl_metrics['bias']]) * 100
    
    # Plot comparison if visualization module is available
    if plot_comparison_with_without_rl:
        os.makedirs('plots', exist_ok=True)
        plot_comparison_with_without_rl(with_rl_metrics, without_rl_metrics, save_dir='plots')
    
    # Print results
    print("\nRL vs Baseline Comparison:")
    print(f"MAPE: {mape_pct_improvement:.2f}% improvement with RL")
    print(f"Bias: {bias_pct_improvement:.2f}% improvement with RL")
    
    logger.info(f"Comparison complete. Plots saved to plots/rl_comparison.png")
    
    return with_rl_metrics, without_rl_metrics

def simulate_wom_effect():
    """Simulate and visualize the Week of Month effect handling."""
    logger.info("Simulating Week of Month effect handling...")
    
    # Initialize components
    data_provider = DataProvider(config.__dict__)
    trainer = ForecastAdjustmentTrainer(config.__dict__)
    
    # Train a simple model
    training_metrics = trainer.train(num_episodes=3)
    
    # Initialize adjuster with the trained model
    adjuster = ForecastAdjuster(config.__dict__)
    
    # Simulate adjustments for 28 days (4 weeks)
    start_date = datetime.now()
    dates = [start_date + timedelta(days=i) for i in range(28)]
    
    # Select a specific category-band to monitor
    category = data_provider.ml_forecasts['category'].unique()[0]
    band = 'A'
    
    # Get adjustments for each day
    adjustments = []
    for date in dates:
        week_of_month = get_week_of_month(date)
        
        # Get adjustment from model
        adjustment_factor, _, _ = adjuster.agent.get_adjustment_for_category_band(
            data_provider, category, band, date, training=False
        )
        
        adjustments.append({
            'date': date,
            'week_of_month': week_of_month,
            'adjustment_factor': adjustment_factor
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(adjustments)
    
    # Plot adjustments by week of month
    plt.figure(figsize=(12, 6))
    
    # Plot adjustment factors
    plt.subplot(1, 2, 1)
    plt.plot(range(28), df['adjustment_factor'], 'b-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    # Add WoM shading
    for wom in range(1, 5):
        color = 'lightgray' if wom % 2 == 0 else 'lightblue'
        start_idx = (wom - 1) * 7
        end_idx = start_idx + 7
        plt.axvspan(start_idx, end_idx, alpha=0.3, color=color)
        plt.text(start_idx + 3.5, 1.12, f"WoM {wom}", 
                ha='center', va='center', fontsize=12)
    
    plt.title(f'Adjustment Factors for {category}-{band} by Week of Month')
    plt.xlabel('Day of Month')
    plt.ylabel('Adjustment Factor')
    plt.ylim(0.85, 1.15)
    plt.grid(True)
    
    # Plot average adjustment by WoM
    plt.subplot(1, 2, 2)
    wom_avg = df.groupby('week_of_month')['adjustment_factor'].mean()
    
    bars = plt.bar(wom_avg.index, wom_avg.values, color=['blue', 'green', 'orange', 'purple'])
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('Average Adjustment by Week of Month')
    plt.xlabel('Week of Month')
    plt.ylabel('Avg Adjustment Factor')
    plt.ylim(0.85, 1.15)
    plt.xticks([1, 2, 3, 4], ['WoM 1', 'WoM 2', 'WoM 3', 'WoM 4'])
    plt.grid(axis='y')
    
    plt.tight_layout()
    
    # Use utility function to ensure directory exists
    from utils import ensure_dir
    ensure_dir('plots')
    
    # Save plot
    plt.savefig(f"plots/wom_effect_{category}_{band}.png")
    plt.close()
    
    logger.info(f"Saved WoM effect plot to plots/wom_effect_{category}_{band}.png")
    
    # Print insights
    print("\nWeek of Month Effect Analysis:")
    print(f"Category: {category}, Band: {band}")
    print(f"Average adjustment factors by Week of Month:")
    for wom, adj in wom_avg.items():
        print(f"  Week {wom}: {adj:.4f}")
    
    # Check if WoM1 is highest adjustment (underforecasting correction)
    if wom_avg[1] > wom_avg[2:].max():
        print("\nInsight: Week 1 shows the highest positive adjustment factor,")
        print("indicating the agent has learned to correct for WoM1 underforecasting.")
    else:
        print("\nInsight: The agent has not yet clearly learned the WoM pattern.")
        print("More training episodes may be needed.")

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    
    print("===== Forecast Adjustment RL System Demo =====\n")
    
    try:
        # Train the model
        train_model(num_episodes=2)
        
        # Run inference
        run_inference()
        
        # Simulate WoM effect handling
        simulate_wom_effect()
        compare_rl_vs_baseline()
        print("\n===== Demo Complete =====")
    except Exception as e:
        logger.error(f"Error running demo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n===== Demo Failed - See log for details =====")
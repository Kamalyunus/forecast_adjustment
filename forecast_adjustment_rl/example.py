"""
Enhanced example script demonstrating the forecast adjustment RL system.
Includes parallel data processing and extended training.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import torch.multiprocessing as mp
from utils import get_week_of_month, plot_adjustment_factors, plot_category_metrics, print_config_summary

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

def train_model(num_episodes=None):
    """Train the RL model."""
    logger.info("Starting model training...")
    
    # Create a proper configuration dictionary
    full_config = {
        'AGENT_CONFIG': config.AGENT_CONFIG,
        'ACTION_CONFIG': config.ACTION_CONFIG,
        'REWARD_CONFIG': config.REWARD_CONFIG,
        'STATE_CONFIG': config.STATE_CONFIG,
        'TRAINING_CONFIG': config.TRAINING_CONFIG, 
        'DATA_CONFIG': config.DATA_CONFIG,
        'SYSTEM_CONFIG': config.SYSTEM_CONFIG,
        'FEATURE_CONFIG': config.FEATURE_CONFIG
    }
    
    # Debug the configuration
    print_config_summary(full_config)
    
    # Initialize trainer
    trainer = ForecastAdjustmentTrainer(full_config)
    
    # If num_episodes is not provided, use the config value
    if num_episodes is None:
        num_episodes = full_config['TRAINING_CONFIG']['num_episodes']
    
    start_time = time.time() 
    
    # Train model
    training_metrics = trainer.train(num_episodes=num_episodes)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Training complete. Final reward: {training_metrics['final_reward']:.4f}")
    logger.info(f"Training took {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Additional training statistics
    if hasattr(trainer.agent, 'policy_losses') and len(trainer.agent.policy_losses) > 0:
        final_policy_loss = np.mean(trainer.agent.policy_losses[-10:])
        logger.info(f"Final average policy loss: {final_policy_loss:.4f}")
    
    if hasattr(trainer.agent, 'value_losses') and len(trainer.agent.value_losses) > 0:
        final_value_loss = np.mean(trainer.agent.value_losses[-10:])
        logger.info(f"Final average value loss: {final_value_loss:.4f}")
    
    return trainer

def run_inference(top_n=10):
    """Run inference with the trained model."""
    logger.info("Running inference with trained model...")
    
    try:
        # Create a proper configuration dictionary
        full_config = {
            'AGENT_CONFIG': config.AGENT_CONFIG,
            'ACTION_CONFIG': config.ACTION_CONFIG,
            'REWARD_CONFIG': config.REWARD_CONFIG,
            'STATE_CONFIG': config.STATE_CONFIG,
            'TRAINING_CONFIG': config.TRAINING_CONFIG,
            'DATA_CONFIG': config.DATA_CONFIG,
            'SYSTEM_CONFIG': config.SYSTEM_CONFIG,
            'FEATURE_CONFIG': config.FEATURE_CONFIG
        }
        
        # Initialize adjuster
        adjuster = ForecastAdjuster(full_config)
        
        # Get current date
        today = datetime.now()
        
        # Adjust forecasts for top N categories
        adjustments = adjuster.adjust_forecasts(
            date=today,
            top_n=top_n
        )
        
        logger.info(f"Applied {len(adjustments)} adjustments")
        
        # Print adjustment summary
        if not adjustments.empty:
            print("\nAdjustment Summary:")
            summary = adjustments.groupby('band')['adjustment_factor'].agg(['mean', 'min', 'max', 'count'])
            print(summary)
            
            print("\nTop 10 adjustments by confidence:")
            top_by_confidence = adjustments.sort_values('confidence', ascending=False).head(10)
            print(top_by_confidence[['category', 'band', 'adjustment_factor', 'confidence']])
            
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
                        from visualization_utils import plot_wom_adjustments
                        
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
            
        return adjustments
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print("Inference failed - see log for details")
        return None

def compare_rl_vs_baseline():
    """Compare forecast performance with and without RL adjustments."""
    logger.info("Comparing RL adjustments vs baseline ML forecasts...")
    
    # Import the utility function
    from utils import get_week_of_month
    
    # Create a proper configuration dictionary
    full_config = {
        'AGENT_CONFIG': config.AGENT_CONFIG,
        'ACTION_CONFIG': config.ACTION_CONFIG,
        'REWARD_CONFIG': config.REWARD_CONFIG,
        'STATE_CONFIG': config.STATE_CONFIG,
        'TRAINING_CONFIG': config.TRAINING_CONFIG,
        'DATA_CONFIG': config.DATA_CONFIG,
        'SYSTEM_CONFIG': config.SYSTEM_CONFIG,
        'FEATURE_CONFIG': config.FEATURE_CONFIG
    }
    
    # Initialize components
    data_provider = DataProvider(full_config)
    adjuster = ForecastAdjuster(full_config)
    
    # Select categories and bands to evaluate - more comprehensive testing
    categories = sorted(data_provider.ml_forecasts['category'].unique())[:10]  # Top 10 categories
    bands = ['A', 'B', 'C']  # All bands
    
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
    
    # By category metrics
    category_metrics = {}
    for category in categories:
        category_metrics[category] = {
            'with_rl': {'mape': [], 'bias': []},
            'without_rl': {'mape': [], 'bias': []}
        }
    
    # For each week, compare performance with and without adjustments
    for week, date in enumerate(dates):
        week_of_month = get_week_of_month(date)  # Use utility function
        
        # For each category-band, get metrics
        week_with_mape = []
        week_with_bias = []
        week_without_mape = []
        week_without_bias = []
        
        for category in categories:
            category_with_mape = []
            category_with_bias = []
            category_without_mape = []
            category_without_bias = []
            
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
                
                # Store for category metrics
                category_with_mape.append(rl_mape)
                category_with_bias.append(rl_bias)
                category_without_mape.append(ml_mape)
                category_without_bias.append(ml_bias)
            
            # Store category averages
            category_metrics[category]['with_rl']['mape'].append(np.mean(category_with_mape))
            category_metrics[category]['with_rl']['bias'].append(np.mean(category_with_bias))
            category_metrics[category]['without_rl']['mape'].append(np.mean(category_without_mape))
            category_metrics[category]['without_rl']['bias'].append(np.mean(category_without_bias))
        
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
        
        # Plot Week-of-Month specific comparison
        wom_data = {
            'with_rl': {'mape': [], 'bias': []},
            'without_rl': {'mape': [], 'bias': []}
        }
        
        for i, date in enumerate(dates):
            wom = get_week_of_month(date)
            wom_data['with_rl']['mape'].append((wom, with_rl_metrics['mape'][i]))
            wom_data['with_rl']['bias'].append((wom, with_rl_metrics['bias'][i]))
            wom_data['without_rl']['mape'].append((wom, without_rl_metrics['mape'][i]))
            wom_data['without_rl']['bias'].append((wom, without_rl_metrics['bias'][i]))
        
        # Create WoM chart
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for wom in range(1, 5):
            ml_mapes = [m for w, m in wom_data['without_rl']['mape'] if w == wom]
            rl_mapes = [m for w, m in wom_data['with_rl']['mape'] if w == wom]
            
            if ml_mapes and rl_mapes:
                plt.bar(f"WoM{wom} ML", np.mean(ml_mapes), color='blue', alpha=0.6)
                plt.bar(f"WoM{wom} RL", np.mean(rl_mapes), color='green', alpha=0.6)
        
        plt.title("MAPE by Week of Month")
        plt.ylabel("MAPE")
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for wom in range(1, 5):
            ml_biases = [abs(b) for w, b in wom_data['without_rl']['bias'] if w == wom]
            rl_biases = [abs(b) for w, b in wom_data['with_rl']['bias'] if w == wom]
            
            if ml_biases and rl_biases:
                plt.bar(f"WoM{wom} ML", np.mean(ml_biases), color='blue', alpha=0.6)
                plt.bar(f"WoM{wom} RL", np.mean(rl_biases), color='green', alpha=0.6)
        
        plt.title("Absolute Bias by Week of Month")
        plt.ylabel("Absolute Bias")
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/wom_comparison.png")
        plt.close()
    
    # Print results (continued)
    print("\nRL vs Baseline Comparison:")
    print(f"MAPE: {mape_pct_improvement:.2f}% improvement with RL")
    print(f"Bias: {bias_pct_improvement:.2f}% improvement with RL")
    
    # Print category-specific improvements
    print("\nTop 5 Categories by Improvement:")
    category_improvements = []
    for category in categories:
        cat_metrics = category_metrics[category]
        
        avg_ml_mape = np.mean(cat_metrics['without_rl']['mape'])
        avg_rl_mape = np.mean(cat_metrics['with_rl']['mape'])
        mape_pct_imp = (avg_ml_mape - avg_rl_mape) / avg_ml_mape * 100 if avg_ml_mape > 0 else 0
        
        avg_ml_bias = np.mean([abs(b) for b in cat_metrics['without_rl']['bias']])
        avg_rl_bias = np.mean([abs(b) for b in cat_metrics['with_rl']['bias']])
        bias_pct_imp = (avg_ml_bias - avg_rl_bias) / avg_ml_bias * 100 if avg_ml_bias > 0 else 0
        
        category_improvements.append({
            'category': category,
            'mape_improvement': mape_pct_imp,
            'bias_improvement': bias_pct_imp,
            'overall_improvement': (mape_pct_imp + bias_pct_imp) / 2
        })
    
    # Sort by overall improvement
    category_improvements.sort(key=lambda x: x['overall_improvement'], reverse=True)
    
    # Print top 5
    for i, cat_imp in enumerate(category_improvements[:5]):
        print(f"{i+1}. {cat_imp['category']}: MAPE: {cat_imp['mape_improvement']:.2f}%, " +
              f"Bias: {cat_imp['bias_improvement']:.2f}%")
    
    logger.info(f"Comparison complete. Plots saved to plots/")
    
    return with_rl_metrics, without_rl_metrics, category_metrics

def simulate_wom_effect():
    """Simulate and visualize the Week of Month effect handling."""
    logger.info("Simulating Week of Month effect handling...")
    
    # Create a proper configuration dictionary
    full_config = {
        'AGENT_CONFIG': config.AGENT_CONFIG,
        'ACTION_CONFIG': config.ACTION_CONFIG,
        'REWARD_CONFIG': config.REWARD_CONFIG,
        'STATE_CONFIG': config.STATE_CONFIG,
        'TRAINING_CONFIG': config.TRAINING_CONFIG,
        'DATA_CONFIG': config.DATA_CONFIG,
        'SYSTEM_CONFIG': config.SYSTEM_CONFIG,
        'FEATURE_CONFIG': config.FEATURE_CONFIG
    }
    
    # Initialize components
    data_provider = DataProvider(full_config)
    trainer = ForecastAdjustmentTrainer(full_config)
    
    # Train a simple model
    training_metrics = trainer.train(num_episodes=5)
    
    # Initialize adjuster with the trained model
    adjuster = ForecastAdjuster(full_config)
    
    # Simulate adjustments for 28 days (4 weeks)
    start_date = datetime.now()
    dates = [start_date + timedelta(days=i) for i in range(28)]
    
    # Select multiple categories to monitor for a more comprehensive view
    categories = data_provider.ml_forecasts['category'].unique()[:5]  # Top 5 categories
    bands = ['A', 'B']  # Focus on high volume bands
    
    # Get adjustments for each day, category, and band
    all_adjustments = []
    
    for date in dates:
        week_of_month = get_week_of_month(date)
        
        for category in categories:
            for band in bands:
                # Get adjustment from model
                adjustment_factor, action_idx, action_probs = adjuster.agent.get_adjustment_for_category_band(
                    data_provider, category, band, date, training=False
                )
                
                all_adjustments.append({
                    'date': date,
                    'category': category,
                    'band': band,
                    'week_of_month': week_of_month,
                    'adjustment_factor': adjustment_factor,
                    'action_idx': action_idx,
                    'confidence': np.max(action_probs)
                })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_adjustments)
    
    # Analyze Week of Month patterns across all categories
    wom_analysis = df.groupby('week_of_month').agg({
        'adjustment_factor': ['mean', 'std', 'min', 'max', 'count'],
        'confidence': ['mean', 'min', 'max']
    })
    
    # Plot adjustments by week of month
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Overall WoM adjustment patterns
    plt.subplot(2, 2, 1)
    wom_avg = df.groupby('week_of_month')['adjustment_factor'].mean()
    wom_std = df.groupby('week_of_month')['adjustment_factor'].std()
    
    bars = plt.bar(wom_avg.index, wom_avg.values, 
                  yerr=wom_std.values, 
                  color=['orange', 'blue', 'green', 'purple'],
                  alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('Average Adjustment by Week of Month (All Categories)')
    plt.xlabel('Week of Month')
    plt.ylabel('Avg Adjustment Factor')
    plt.ylim(0.85, 1.2)
    plt.xticks([1, 2, 3, 4], ['WoM 1', 'WoM 2', 'WoM 3', 'WoM 4'])
    plt.grid(axis='y')
    
    # Plot 2: Adjustment by category and WoM
    plt.subplot(2, 2, 2)
    
    # Create a pivot table for category-wom adjustments
    cat_wom_pivot = df.pivot_table(
        index='category', 
        columns='week_of_month', 
        values='adjustment_factor',
        aggfunc='mean'
    )
    
    # Plot as heatmap
    im = plt.imshow(cat_wom_pivot.values, cmap='coolwarm', aspect='auto', vmin=0.9, vmax=1.1)
    plt.colorbar(im, label='Adjustment Factor')
    
    plt.title('Adjustment Patterns by Category and WoM')
    plt.xlabel('Week of Month')
    plt.ylabel('Category')
    plt.xticks(range(4), ['WoM 1', 'WoM 2', 'WoM 3', 'WoM 4'])
    plt.yticks(range(len(cat_wom_pivot.index)), cat_wom_pivot.index)
    
    # Plot 3: Daily adjustment pattern over a month
    plt.subplot(2, 1, 2)
    
    # Aggregate by date and week of month
    daily_adj = df.groupby(['date', 'week_of_month'])['adjustment_factor'].mean().reset_index()
    
    # Plot with different colors for each WoM
    for wom in range(1, 5):
        wom_data = daily_adj[daily_adj['week_of_month'] == wom]
        plt.plot(wom_data['date'], wom_data['adjustment_factor'], 'o-', 
                 label=f'Week {wom}', linewidth=2, markersize=8)
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.title('Daily Adjustment Factors Over Month')
    plt.xlabel('Date')
    plt.ylabel('Adjustment Factor')
    plt.ylim(0.85, 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Use utility function to ensure directory exists
    from utils import ensure_dir
    ensure_dir('plots')
    
    # Save plot
    plt.savefig(f"plots/wom_effect_analysis.png")
    plt.close()
    
    logger.info(f"Saved WoM effect plot to plots/wom_effect_analysis.png")
    
    # Print insights
    print("\nWeek of Month Effect Analysis:")
    print(f"Average adjustment factors by Week of Month:")
    print(wom_analysis)
    
    # Check if WoM1 is highest adjustment (underforecasting correction)
    if wom_avg[1] > wom_avg[2:].max():
        print("\nInsight: Week 1 shows the highest positive adjustment factor,")
        print("indicating the agent has learned to correct for WoM1 underforecasting.")
        
        # Find categories with strongest WoM1 pattern
        wom1_effect = cat_wom_pivot[1] - cat_wom_pivot[2:].mean(axis=1)
        top_wom1_categories = wom1_effect.sort_values(ascending=False).head(3)
        
        print("\nCategories with strongest Week 1 adjustment pattern:")
        for cat, effect in top_wom1_categories.items():
            print(f"  {cat}: +{effect:.4f} (relative to other weeks)")
    else:
        print("\nInsight: The agent has not yet clearly learned the WoM pattern.")
        print("More training episodes may be needed.")
    
    return df

def run_performance_test():
    """Run a comprehensive performance test of the trained model."""
    logger.info("Running performance test...")
    
    # Create a proper configuration dictionary
    full_config = {
        'AGENT_CONFIG': config.AGENT_CONFIG,
        'ACTION_CONFIG': config.ACTION_CONFIG,
        'REWARD_CONFIG': config.REWARD_CONFIG,
        'STATE_CONFIG': config.STATE_CONFIG,
        'TRAINING_CONFIG': config.TRAINING_CONFIG,
        'DATA_CONFIG': config.DATA_CONFIG,
        'SYSTEM_CONFIG': config.SYSTEM_CONFIG,
        'FEATURE_CONFIG': config.FEATURE_CONFIG
    }
    
    # Initialize components
    data_provider = DataProvider(full_config)
    
    # Train model with more episodes for better performance
    trainer = train_model(num_episodes=20)
    
    # Initialize adjuster with the trained model
    adjuster = ForecastAdjuster(full_config)
    
    # Get all categories and bands
    all_categories = sorted(data_provider.ml_forecasts['category'].unique())
    all_bands = ['A', 'B', 'C']
    
    # Define test period (next 28 days)
    start_date = datetime.now()
    test_dates = [start_date + timedelta(days=i) for i in range(28)]
    
    # Define metrics to track
    metrics = {
        'overall': {
            'mape_before': [],
            'mape_after': [],
            'bias_before': [],
            'bias_after': [],
            'adjustment_factors': []
        },
        'by_wom': {1: {}, 2: {}, 3: {}, 4: {}},
        'by_category': {},
        'by_band': {'A': {}, 'B': {}, 'C': {}}
    }
    
    # Initialize category metrics
    for category in all_categories:
        metrics['by_category'][category] = {
            'mape_before': [],
            'mape_after': [],
            'bias_before': [],
            'bias_after': [],
            'adjustment_factors': []
        }
    
    # Initialize WoM metrics
    for wom in range(1, 5):
        metrics['by_wom'][wom] = {
            'mape_before': [],
            'mape_after': [],
            'bias_before': [],
            'bias_after': [],
            'adjustment_factors': []
        }
    
    # Initialize band metrics
    for band in all_bands:
        metrics['by_band'][band] = {
            'mape_before': [],
            'mape_after': [],
            'bias_before': [],
            'bias_after': [],
            'adjustment_factors': []
        }
    
    # Run test for each date, category, and band
    test_results = []
    
    for date in test_dates:
        week_of_month = get_week_of_month(date)
        
        for category in all_categories:
            for band in all_bands:
                # Get metrics before adjustment
                mape_before = data_provider.get_historical_mape(category, band, date, before_adjustment=True)
                bias_before = data_provider.get_historical_bias(category, band, date, before_adjustment=True)
                
                # Get adjustment from model
                adjustment_factor, _, action_probs = adjuster.agent.get_adjustment_for_category_band(
                    data_provider, category, band, date, training=False
                )
                
                # Get metrics after adjustment
                mape_after = data_provider.get_historical_mape(category, band, date, before_adjustment=False)
                bias_after = data_provider.get_historical_bias(category, band, date, before_adjustment=False)
                
                # Store results
                test_results.append({
                    'date': date,
                    'category': category,
                    'band': band,
                    'week_of_month': week_of_month,
                    'mape_before': mape_before,
                    'mape_after': mape_after,
                    'bias_before': bias_before,
                    'bias_after': bias_after,
                    'adjustment_factor': adjustment_factor,
                    'confidence': np.max(action_probs)
                })
                
                # Update metrics
                metrics['overall']['mape_before'].append(mape_before)
                metrics['overall']['mape_after'].append(mape_after)
                metrics['overall']['bias_before'].append(bias_before)
                metrics['overall']['bias_after'].append(bias_after)
                metrics['overall']['adjustment_factors'].append(adjustment_factor)
                
                # Update WoM metrics
                metrics['by_wom'][week_of_month]['mape_before'].append(mape_before)
                metrics['by_wom'][week_of_month]['mape_after'].append(mape_after)
                metrics['by_wom'][week_of_month]['bias_before'].append(bias_before)
                metrics['by_wom'][week_of_month]['bias_after'].append(bias_after)
                metrics['by_wom'][week_of_month]['adjustment_factors'].append(adjustment_factor)
                
                # Update category metrics
                metrics['by_category'][category]['mape_before'].append(mape_before)
                metrics['by_category'][category]['mape_after'].append(mape_after)
                metrics['by_category'][category]['bias_before'].append(bias_before)
                metrics['by_category'][category]['bias_after'].append(bias_after)
                metrics['by_category'][category]['adjustment_factors'].append(adjustment_factor)
                
                # Update band metrics
                metrics['by_band'][band]['mape_before'].append(mape_before)
                metrics['by_band'][band]['mape_after'].append(mape_after)
                metrics['by_band'][band]['bias_before'].append(bias_before)
                metrics['by_band'][band]['bias_after'].append(bias_after)
                metrics['by_band'][band]['adjustment_factors'].append(adjustment_factor)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(test_results)
    
    # Calculate overall performance metrics
    overall_mape_before = np.mean(metrics['overall']['mape_before'])
    overall_mape_after = np.mean(metrics['overall']['mape_after'])
    overall_mape_improvement = (overall_mape_before - overall_mape_after) / overall_mape_before * 100
    
    overall_bias_before = np.mean([abs(b) for b in metrics['overall']['bias_before']])
    overall_bias_after = np.mean([abs(b) for b in metrics['overall']['bias_after']])
    overall_bias_improvement = (overall_bias_before - overall_bias_after) / overall_bias_before * 100
    
    # Calculate WoM-specific metrics
    wom_metrics = {}
    for wom in range(1, 5):
        wom_data = metrics['by_wom'][wom]
        
        mape_before = np.mean(wom_data['mape_before'])
        mape_after = np.mean(wom_data['mape_after'])
        mape_improvement = (mape_before - mape_after) / mape_before * 100 if mape_before > 0 else 0
        
        bias_before = np.mean([abs(b) for b in wom_data['bias_before']])
        bias_after = np.mean([abs(b) for b in wom_data['bias_after']])
        bias_improvement = (bias_before - bias_after) / bias_before * 100 if bias_before > 0 else 0
        
        avg_adjustment = np.mean(wom_data['adjustment_factors'])
        
        wom_metrics[wom] = {
            'mape_before': mape_before,
            'mape_after': mape_after,
            'mape_improvement': mape_improvement,
            'bias_before': bias_before,
            'bias_after': bias_after,
            'bias_improvement': bias_improvement,
            'avg_adjustment': avg_adjustment
        }
    
    # Create visualizations
    try:
        from visualization_utils import plot_comparison_with_without_rl
        
        # Prepare data for WoM comparison
        wom_comparison = {
            'labels': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            'mape': {
                'before': [wom_metrics[w]['mape_before'] for w in range(1, 5)],
                'after': [wom_metrics[w]['mape_after'] for w in range(1, 5)]
            },
            'bias': {
                'before': [wom_metrics[w]['bias_before'] for w in range(1, 5)],
                'after': [wom_metrics[w]['bias_after'] for w in range(1, 5)]
            },
            'adjustments': [wom_metrics[w]['avg_adjustment'] for w in range(1, 5)]
        }
        
        # Create WoM performance chart
        plt.figure(figsize=(15, 10))
        
        # MAPE by WoM
        plt.subplot(2, 2, 1)
        x = np.arange(len(wom_comparison['labels']))
        width = 0.35
        
        plt.bar(x - width/2, wom_comparison['mape']['before'], width, 
                label='Before Adjustment', color='blue', alpha=0.7)
        plt.bar(x + width/2, wom_comparison['mape']['after'], width,
                label='After Adjustment', color='green', alpha=0.7)
        
        plt.title('MAPE by Week of Month')
        plt.xlabel('Week of Month')
        plt.ylabel('MAPE')
        plt.xticks(x, wom_comparison['labels'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Bias by WoM
        plt.subplot(2, 2, 2)
        plt.bar(x - width/2, wom_comparison['bias']['before'], width,
                label='Before Adjustment', color='blue', alpha=0.7)
        plt.bar(x + width/2, wom_comparison['bias']['after'], width,
                label='After Adjustment', color='green', alpha=0.7)
        
        plt.title('Absolute Bias by Week of Month')
        plt.xlabel('Week of Month')
        plt.ylabel('Absolute Bias')
        plt.xticks(x, wom_comparison['labels'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Adjustment factors by WoM
        plt.subplot(2, 2, 3)
        bars = plt.bar(x, wom_comparison['adjustments'], color=['orange', 'blue', 'blue', 'blue'])
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.title('Average Adjustment Factor by Week of Month')
        plt.xlabel('Week of Month')
        plt.ylabel('Adjustment Factor')
        plt.xticks(x, wom_comparison['labels'])
        plt.grid(axis='y', alpha=0.3)
        
        # Improvement percentages
        plt.subplot(2, 2, 4)
        
        improvements = [
            [wom_metrics[w]['mape_improvement'] for w in range(1, 5)],
            [wom_metrics[w]['bias_improvement'] for w in range(1, 5)]
        ]
        
        im = plt.imshow(improvements, cmap='YlGn', aspect='auto')
        plt.colorbar(im, label='% Improvement')
        
        plt.title('Improvement Percentages by Week of Month')
        plt.xlabel('Week of Month')
        plt.ylabel('Metric')
        plt.xticks(range(4), wom_comparison['labels'])
        plt.yticks([0, 1], ['MAPE', 'Bias'])
        
        # Add text annotations
        for i in range(2):
            for j in range(4):
                plt.text(j, i, f'{improvements[i][j]:.1f}%', 
                        ha='center', va='center', color='black', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('plots/performance_test_results.png')
        plt.close()
        
    except ImportError:
        logger.warning("visualization_utils module not found, skipping visualization")
    
    # Print results
    print("\n===== Performance Test Results =====")
    print("\nOverall Metrics:")
    print(f"MAPE: {overall_mape_before:.4f} → {overall_mape_after:.4f} ({overall_mape_improvement:.2f}% improvement)")
    print(f"Bias: {overall_bias_before:.4f} → {overall_bias_after:.4f} ({overall_bias_improvement:.2f}% improvement)")
    
    print("\nPerformance by Week of Month:")
    for wom, wom_data in wom_metrics.items():
        print(f"Week {wom}:")
        print(f"  MAPE: {wom_data['mape_before']:.4f} → {wom_data['mape_after']:.4f} ({wom_data['mape_improvement']:.2f}% improvement)")
        print(f"  Bias: {wom_data['bias_before']:.4f} → {wom_data['bias_after']:.4f} ({wom_data['bias_improvement']:.2f}% improvement)")
        print(f"  Avg Adjustment: {wom_data['avg_adjustment']:.4f}")
    
    # Compute top performers
    category_improvements = []
    for category, cat_data in metrics['by_category'].items():
        mape_before = np.mean(cat_data['mape_before'])
        mape_after = np.mean(cat_data['mape_after'])
        mape_improvement = (mape_before - mape_after) / mape_before * 100 if mape_before > 0 else 0
        
        bias_before = np.mean([abs(b) for b in cat_data['bias_before']])
        bias_after = np.mean([abs(b) for b in cat_data['bias_after']])
        bias_improvement = (bias_before - bias_after) / bias_before * 100 if bias_before > 0 else 0
        
        overall_improvement = (mape_improvement + bias_improvement) / 2
        
        category_improvements.append({
            'category': category,
            'mape_improvement': mape_improvement,
            'bias_improvement': bias_improvement,
            'overall_improvement': overall_improvement
        })
    
    category_improvements.sort(key=lambda x: x['overall_improvement'], reverse=True)
    
    print("\nTop 5 Categories by Improvement:")
    for i, cat in enumerate(category_improvements[:5]):
        print(f"{i+1}. {cat['category']}: MAPE: {cat['mape_improvement']:.2f}%, " +
              f"Bias: {cat['bias_improvement']:.2f}%, Overall: {cat['overall_improvement']:.2f}%")
    
    logger.info("Performance test completed")
    
    return results_df, metrics, wom_metrics

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    
    print("===== Forecast Adjustment RL System Demo =====\n")
    
    try:
        # Enable multiprocessing for PyTorch
        if config.SYSTEM_CONFIG.get('parallel_processing', True):
            mp.set_start_method('spawn', force=True)
        
        # Train the model with more episodes
        train_model()
        
        # Run inference on more categories
        run_inference(top_n=10)
        
        # Run comprehensive performance test
        run_performance_test()
        
        # Compare RL vs baseline
        compare_rl_vs_baseline()
        
        # Simulate WoM effect handling
        simulate_wom_effect()
        
        print("\n===== Demo Complete =====")
    except Exception as e:
        logger.error(f"Error running demo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n===== Demo Failed - See log for details =====")
"""
Visualization utilities for the Forecast Adjustment RL system.
Provides functions for visualizing training progress and performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def plot_training_progress(episode_rewards, save_dir=None, show=True):
    """
    Plot training progress showing episode rewards over time.
    
    Args:
        episode_rewards: List of episode rewards
        save_dir: Directory to save the plot (optional)
        show: Whether to display the plot (default True)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-', markersize=4)
    plt.title('Training Progress: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Mean Episode Reward')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(episode_rewards) > 1:
        z = np.polyfit(range(1, len(episode_rewards) + 1), episode_rewards, 1)
        p = np.poly1d(z)
        plt.plot(range(1, len(episode_rewards) + 1), p(range(1, len(episode_rewards) + 1)), 
                "r--", alpha=0.7, label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
        plt.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'episode_rewards.png')
        plt.savefig(filename)
        logger.info(f"Saved episode rewards plot to {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_forecast_metrics(metrics_df, save_dir=None, show=True):
    """
    Plot forecast metrics (MAPE and Bias) over time.
    
    Args:
        metrics_df: DataFrame with 'episode', 'mape', and 'bias' columns
        save_dir: Directory to save the plots (optional)
        show: Whether to display the plots (default True)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot MAPE
    ax1.plot(metrics_df['episode'], metrics_df['mape'], marker='o', linestyle='-', 
             markersize=4, color='green')
    ax1.set_title('MAPE Over Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('MAPE')
    ax1.grid(True, alpha=0.3)
    
    # Add initial and final annotations for MAPE
    if len(metrics_df) > 1:
        initial_mape = metrics_df['mape'].iloc[0]
        final_mape = metrics_df['mape'].iloc[-1]
        improvement_pct = (initial_mape - final_mape) / initial_mape * 100
        
        ax1.annotate(f'Initial: {initial_mape:.4f}', 
                     xy=(metrics_df['episode'].iloc[0], initial_mape),
                     xytext=(metrics_df['episode'].iloc[0] + 1, initial_mape * 1.05),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        ax1.annotate(f'Final: {final_mape:.4f}\nImprovement: {improvement_pct:.1f}%', 
                     xy=(metrics_df['episode'].iloc[-1], final_mape),
                     xytext=(metrics_df['episode'].iloc[-1] * 0.9, final_mape * 0.9),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    # Plot Bias
    ax2.plot(metrics_df['episode'], metrics_df['bias'], marker='o', linestyle='-',
             markersize=4, color='blue')
    ax2.set_title('Forecast Bias Over Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Bias')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)  # Zero bias reference line
    
    # Add initial and final annotations for Bias
    if len(metrics_df) > 1:
        initial_bias = metrics_df['bias'].iloc[0]
        final_bias = metrics_df['bias'].iloc[-1]
        abs_improvement = abs(initial_bias) - abs(final_bias)
        
        ax2.annotate(f'Initial: {initial_bias:.4f}', 
                    xy=(metrics_df['episode'].iloc[0], initial_bias),
                    xytext=(metrics_df['episode'].iloc[0] + 1, initial_bias * 1.2),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        ax2.annotate(f'Final: {final_bias:.4f}\nAbs Improvement: {abs_improvement:.4f}', 
                    xy=(metrics_df['episode'].iloc[-1], final_bias),
                    xytext=(metrics_df['episode'].iloc[-1] * 0.9, final_bias * 1.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'forecast_metrics.png')
        plt.savefig(filename)
        logger.info(f"Saved forecast metrics plot to {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_wom_adjustments(wom_adjustments, save_dir=None, show=True):
    """
    Plot adjustment factors by Week of Month.
    
    Args:
        wom_adjustments: Dictionary with week numbers as keys and adjustment factors as values
        save_dir: Directory to save the plot (optional)
        show: Whether to display the plot (default True)
    """
    weeks = list(wom_adjustments.keys())
    factors = list(wom_adjustments.values())
    
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(weeks, factors, color=['orange', 'blue', 'blue', 'blue'])
    bars[0].set_color('orange')  # Highlight Week 1
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)  # Reference line for no adjustment
    
    plt.title('Adjustment Factors by Week of Month')
    plt.xlabel('Week of Month')
    plt.ylabel('Adjustment Factor')
    plt.ylim(min(0.9, min(factors) - 0.05), max(1.1, max(factors) + 0.05))
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'wom_adjustments.png')
        plt.savefig(filename)
        logger.info(f"Saved Week of Month adjustments plot to {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_metrics_summary(metrics, categories, bands, save_dir=None, show=True):
    """
    Plot summary of metrics improvement across categories and bands.
    
    Args:
        metrics: Dictionary mapping (category, band) tuples to metric dictionaries
        categories: List of categories to include
        bands: List of bands to include
        save_dir: Directory to save the plot (optional)
        show: Whether to display the plot (default True)
    """
    # Extract initial and final MAPE and bias for each category-band
    summary_data = []
    
    for category in categories:
        for band in bands:
            key = (category, band)
            if key in metrics:
                data = metrics[key]
                if data:  # Check if data exists
                    initial_mape = data[0].get('mape', 0)
                    final_mape = data[-1].get('mape', 0)
                    initial_bias = data[0].get('bias', 0)
                    final_bias = data[-1].get('bias', 0)
                    
                    summary_data.append({
                        'category_band': f"{category}-{band}",
                        'mape_improvement': initial_mape - final_mape,
                        'bias_improvement': abs(initial_bias) - abs(final_bias)
                    })
    
    if not summary_data:
        logger.warning("No data available for metrics summary plot")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(summary_data)
    
    # Sort by MAPE improvement
    df = df.sort_values('mape_improvement', ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # MAPE improvement
    ax1.barh(df['category_band'], df['mape_improvement'], color='green')
    ax1.set_title('MAPE Improvement by Category-Band')
    ax1.set_xlabel('MAPE Improvement')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Bias improvement
    ax2.barh(df['category_band'], df['bias_improvement'], color='blue')
    ax2.set_title('Absolute Bias Improvement by Category-Band')
    ax2.set_xlabel('Absolute Bias Improvement')
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'metrics_summary.png')
        plt.savefig(filename)
        logger.info(f"Saved metrics summary plot to {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_training_report(metrics_path, output_dir):
    """
    Create a comprehensive training report with visualizations.
    
    Args:
        metrics_path: Path to the metrics JSON file
        output_dir: Directory to save the report and visualizations
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        episode_rewards = metrics.get('episode_rewards', [])
        category_band_metrics = metrics.get('category_band_metrics', {})
        
        # Convert string keys back to tuples
        category_band_data = {}
        for key, value in category_band_metrics.items():
            # Parse key from string format "(category, band)"
            try:
                parts = key.strip("()").split(", ")
                if len(parts) == 2:
                    category, band = parts
                    # Remove quotes if present
                    category = category.strip("'")
                    band = band.strip("'")
                    category_band_data[(category, band)] = value
            except Exception as e:
                logger.warning(f"Error parsing key {key}: {str(e)}")
        
        # Get unique categories and bands
        categories = set()
        bands = set()
        for cat, band in category_band_data.keys():
            categories.add(cat)
            bands.add(band)
        
        categories = sorted(list(categories))
        bands = sorted(list(bands))
        
        # Create visualizations
        plot_training_progress(episode_rewards, save_dir=output_dir, show=False)
        
        # Extract MAPE and bias data
        all_metrics = []
        for i, reward in enumerate(episode_rewards):
            episode_data = {'episode': i + 1, 'reward': reward}
            
            # Get average MAPE and bias for this episode (if available)
            mape_values = []
            bias_values = []
            
            for (cat, band), data in category_band_data.items():
                if i < len(data):
                    if 'mape' in data[i]:
                        mape_values.append(data[i]['mape'])
                    if 'bias' in data[i]:
                        bias_values.append(data[i]['bias'])
            
            if mape_values:
                episode_data['mape'] = np.mean(mape_values)
            if bias_values:
                episode_data['bias'] = np.mean(bias_values)
            
            all_metrics.append(episode_data)
        
        metrics_df = pd.DataFrame(all_metrics)
        
        if 'mape' in metrics_df.columns and 'bias' in metrics_df.columns:
            plot_forecast_metrics(metrics_df, save_dir=output_dir, show=False)
        
        # Extract Week of Month adjustments (if available)
        wom_adjustments = {}
        for i in range(1, 5):
            wom_key = f'WoM{i}'
            values = []
            
            for (cat, band), data in category_band_data.items():
                for entry in data:
                    if 'week_of_month' in entry and entry['week_of_month'] == i:
                        if 'adjustment_factor' in entry:
                            values.append(entry['adjustment_factor'])
            
            if values:
                wom_adjustments[wom_key] = np.mean(values)
        
        if wom_adjustments:
            plot_wom_adjustments(wom_adjustments, save_dir=output_dir, show=False)
        
        # Create metrics summary plot
        plot_metrics_summary(category_band_data, categories, bands, 
                           save_dir=output_dir, show=False)
        
        # Generate report text
        report_path = os.path.join(output_dir, 'training_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Forecast Adjustment RL Training Report\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Training Summary\n\n")
            f.write(f"* Episodes completed: {len(episode_rewards)}\n")
            
            if episode_rewards:
                f.write(f"* Initial reward: {episode_rewards[0]:.4f}\n")
                f.write(f"* Final reward: {episode_rewards[-1]:.4f}\n")
                f.write(f"* Reward improvement: {episode_rewards[-1] - episode_rewards[0]:.4f}\n\n")
            
            f.write("## Forecast Metrics Improvement\n\n")
            
            if 'mape' in metrics_df.columns:
                initial_mape = metrics_df['mape'].iloc[0]
                final_mape = metrics_df['mape'].iloc[-1]
                mape_improvement = initial_mape - final_mape
                mape_pct_improvement = (mape_improvement / initial_mape) * 100
                
                f.write(f"### MAPE\n")
                f.write(f"* Initial: {initial_mape:.4f}\n")
                f.write(f"* Final: {final_mape:.4f}\n")
                f.write(f"* Absolute improvement: {mape_improvement:.4f}\n")
                f.write(f"* Percentage improvement: {mape_pct_improvement:.2f}%\n\n")
            
            if 'bias' in metrics_df.columns:
                initial_bias = metrics_df['bias'].iloc[0]
                final_bias = metrics_df['bias'].iloc[-1]
                abs_improvement = abs(initial_bias) - abs(final_bias)
                
                f.write(f"### Bias\n")
                f.write(f"* Initial: {initial_bias:.4f}\n")
                f.write(f"* Final: {final_bias:.4f}\n")
                f.write(f"* Absolute improvement: {abs_improvement:.4f}\n\n")
            
            f.write("## Week of Month Adjustment Patterns\n\n")
            
            if wom_adjustments:
                f.write("| Week | Adjustment Factor |\n")
                f.write("|------|------------------|\n")
                for week, factor in wom_adjustments.items():
                    f.write(f"| {week} | {factor:.4f} |\n")
                
                # Check for WoM1 pattern
                if 'WoM1' in wom_adjustments and wom_adjustments['WoM1'] > 1.0:
                    f.write("\nThe model has learned to increase forecasts in Week 1, ")
                    f.write("suggesting it has identified and is correcting for Week 1 underforecasting bias.\n")
            
            f.write("\n## Visualizations\n\n")
            f.write("* [Episode Rewards](episode_rewards.png)\n")
            f.write("* [Forecast Metrics](forecast_metrics.png)\n")
            f.write("* [Week of Month Adjustments](wom_adjustments.png)\n")
            f.write("* [Metrics Summary by Category-Band](metrics_summary.png)\n")
        
        logger.info(f"Training report generated at {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error creating training report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_comparison_with_without_rl(with_rl_metrics, without_rl_metrics, save_dir=None, show=True):
    """
    Plot comparison of forecast performance with and without RL adjustments.
    
    Args:
        with_rl_metrics: Dictionary with 'mape' and 'bias' lists for RL-adjusted forecasts
        without_rl_metrics: Dictionary with 'mape' and 'bias' lists for unadjusted forecasts
        save_dir: Directory to save the plot (optional)
        show: Whether to display the plot (default True)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Categories for x-axis (weeks or categories)
    categories = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    
    # Ensure data length matches categories
    if len(with_rl_metrics['mape']) != len(categories):
        categories = [f'Category {i+1}' for i in range(len(with_rl_metrics['mape']))]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Plot MAPE comparison
    rects1 = ax1.bar(x - width/2, without_rl_metrics['mape'], width, label='Without RL', color='gray')
    rects2 = ax1.bar(x + width/2, with_rl_metrics['mape'], width, label='With RL', color='green')
    
    ax1.set_title('MAPE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel('MAPE')
    ax1.legend()
    
    # Add percentage improvement labels
    for i, (without, with_rl) in enumerate(zip(without_rl_metrics['mape'], with_rl_metrics['mape'])):
        improvement = (without - with_rl) / without * 100
        ax1.annotate(f"{improvement:.1f}%", 
                     xy=(i, with_rl),
                     xytext=(0, 10),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=8, color='green')
    
    # Plot bias comparison (absolute values)
    bias_without_abs = [abs(b) for b in without_rl_metrics['bias']]
    bias_with_abs = [abs(b) for b in with_rl_metrics['bias']]
    
    rects3 = ax2.bar(x - width/2, bias_without_abs, width, label='Without RL', color='gray')
    rects4 = ax2.bar(x + width/2, bias_with_abs, width, label='With RL', color='blue')
    
    ax2.set_title('Absolute Bias Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Absolute Bias')
    ax2.legend()
    
    # Add percentage improvement labels
    for i, (without, with_rl) in enumerate(zip(bias_without_abs, bias_with_abs)):
        if without > 0:
            improvement = (without - with_rl) / without * 100
            ax2.annotate(f"{improvement:.1f}%", 
                        xy=(i, with_rl),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, color='blue')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'rl_comparison.png')
        plt.savefig(filename)
        logger.info(f"Saved RL comparison plot to {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()
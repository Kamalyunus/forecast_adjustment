"""
Visualization utilities for forecast adjustment system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple
import logging


def visualize_adjustments(
    adjustments_df: pd.DataFrame,
    output_dir: str = "output/visualizations",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Create visualizations of forecast adjustments.
    
    Args:
        adjustments_df: DataFrame of adjustment data
        output_dir: Directory to save visualizations
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger("Visualization")
        
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating forecast adjustment visualizations")
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Adjustment Factor Distribution
    plt.subplot(2, 2, 1)
    adjustment_factors = sorted(adjustments_df['adjustment_factor'].unique())
    plt.hist(adjustments_df['adjustment_factor'], bins=20, alpha=0.75)
    plt.title('Adjustment Factor Distribution')
    plt.xlabel('Adjustment Factor')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Day of Week Patterns
    plt.subplot(2, 2, 2)
    if 'day_of_week' in adjustments_df.columns:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Group by day of week
        dow_factors = {}
        for i, day in enumerate(day_names):
            day_data = adjustments_df[adjustments_df['day_of_week'] == i]
            if not day_data.empty:
                dow_factors[day] = day_data['adjustment_factor'].mean()
        
        plt.bar(dow_factors.keys(), dow_factors.values())
        plt.title('Average Adjustment Factor by Day of Week')
        plt.ylabel('Avg Adjustment Factor')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Context Comparison
    plt.subplot(2, 2, 3)
    contexts = []
    avg_factors = []
    
    # Overall average
    avg_factor = adjustments_df['adjustment_factor'].mean()
    contexts.append('Overall')
    avg_factors.append(avg_factor)
    
    # Holiday average
    if 'is_holiday' in adjustments_df.columns:
        holiday_df = adjustments_df[adjustments_df['is_holiday'] == True]
        if not holiday_df.empty:
            holiday_avg = holiday_df['adjustment_factor'].mean()
            contexts.append('Holiday')
            avg_factors.append(holiday_avg)
    
    # Promotion average
    if 'is_promotion' in adjustments_df.columns:
        promo_df = adjustments_df[adjustments_df['is_promotion'] == True]
        if not promo_df.empty:
            promo_avg = promo_df['adjustment_factor'].mean()
            contexts.append('Promotion')
            avg_factors.append(promo_avg)
    
    # Weekend average
    if 'is_weekend' in adjustments_df.columns:
        weekend_df = adjustments_df[adjustments_df['is_weekend'] == True]
        if not weekend_df.empty:
            weekend_avg = weekend_df['adjustment_factor'].mean()
            contexts.append('Weekend')
            avg_factors.append(weekend_avg)
    
    plt.bar(contexts, avg_factors)
    plt.title('Average Adjustment by Context')
    plt.ylabel('Avg Adjustment Factor')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Pattern Type Comparison
    plt.subplot(2, 2, 4)
    if 'pattern_type' in adjustments_df.columns:
        pattern_avgs = {}
        for pattern in adjustments_df['pattern_type'].unique():
            if pattern != "unknown":
                pattern_df = adjustments_df[adjustments_df['pattern_type'] == pattern]
                if not pattern_df.empty:
                    pattern_avgs[pattern] = pattern_df['adjustment_factor'].mean()
        
        if pattern_avgs:
            plt.bar(pattern_avgs.keys(), pattern_avgs.values())
            plt.title('Average Adjustment by Pattern Type')
            plt.ylabel('Avg Adjustment Factor')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjustment_analysis.png'))
    plt.close()
    
    # Create band-specific visualizations if available
    if 'sku_band' in adjustments_df.columns:
        plot_band_comparison(adjustments_df, output_dir, logger)
    
    logger.info(f"Saved visualizations to {output_dir}")


def plot_training_metrics(
    metrics: Dict,
    output_dir: str = "output/visualizations",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Plot training metrics from a training run.
    
    Args:
        metrics: Dictionary of training metrics
        output_dir: Directory to save visualizations
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger("Visualization")
        
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating training metrics visualizations")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: MAPE Improvement
    plt.subplot(2, 2, 1)
    plt.plot(metrics['mape_improvements'])
    plt.title('MAPE Improvement Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Improvement Ratio')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    if len(metrics['mape_improvements']) > 10:
        window_size = 10
        moving_avg = np.convolve(metrics['mape_improvements'], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(metrics['mape_improvements'])), moving_avg, 'r-', label='Moving Avg')
        plt.legend()
    
    # Plot 2: Bias Improvement
    plt.subplot(2, 2, 2)
    plt.plot(metrics['bias_improvements'])
    plt.title('Bias Improvement Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Improvement Ratio')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    if len(metrics['bias_improvements']) > 10:
        window_size = 10
        moving_avg = np.convolve(metrics['bias_improvements'], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(metrics['bias_improvements'])), moving_avg, 'r-', label='Moving Avg')
        plt.legend()
    
    # Plot 3: Context-specific improvements
    plt.subplot(2, 2, 3)
    contexts = []
    improvements = []
    
    # Overall
    contexts.append('Overall')
    improvements.append(np.mean(metrics['mape_improvements'][-50:]))
    
    # Context-specific
    if 'holiday_metrics' in metrics and metrics['holiday_metrics'].get('mape_improvements'):
        contexts.append('Holiday')
        improvements.append(np.mean(metrics['holiday_metrics']['mape_improvements'][-50:]))
    
    if 'promo_metrics' in metrics and metrics['promo_metrics'].get('mape_improvements'):
        contexts.append('Promo')
        improvements.append(np.mean(metrics['promo_metrics']['mape_improvements'][-50:]))
    
    if 'weekend_metrics' in metrics and metrics['weekend_metrics'].get('mape_improvements'):
        contexts.append('Weekend')
        improvements.append(np.mean(metrics['weekend_metrics']['mape_improvements'][-50:]))
    
    plt.bar(contexts, improvements)
    plt.title('MAPE Improvement by Context (Last 50 Episodes)')
    plt.ylabel('Improvement Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Band-specific improvements
    plt.subplot(2, 2, 4)
    if 'band_metrics' in metrics:
        bands = []
        band_improvements = []
        
        for band in ['A', 'B', 'C', 'D', 'E']:
            if band in metrics['band_metrics'] and metrics['band_metrics'][band].get('mape_improvements'):
                bands.append(band)
                band_improvements.append(np.mean(metrics['band_metrics'][band]['mape_improvements'][-50:]))
        
        if bands:
            plt.bar(bands, band_improvements)
            plt.title('MAPE Improvement by SKU Band (Last 50 Episodes)')
            plt.ylabel('Improvement Ratio')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    logger.info(f"Saved training metrics visualizations to {output_dir}")


def plot_band_comparison(
    adjustments_df: pd.DataFrame,
    output_dir: str = "output/visualizations",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Create band-specific comparison visualizations.
    
    Args:
        adjustments_df: DataFrame of adjustment data with sku_band column
        output_dir: Directory to save visualizations
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger("Visualization")
    
    if 'sku_band' not in adjustments_df.columns:
        logger.warning("Cannot create band comparisons: 'sku_band' column not in data")
        return
    
    logger.info("Generating band comparison visualizations")
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Average adjustment by band
    plt.subplot(2, 2, 1)
    band_avgs = {}
    for band in sorted(adjustments_df['sku_band'].unique()):
        band_df = adjustments_df[adjustments_df['sku_band'] == band]
        if not band_df.empty:
            band_avgs[band] = band_df['adjustment_factor'].mean()
    
    if band_avgs:
        bands = list(band_avgs.keys())
        avgs = list(band_avgs.values())
        
        plt.bar(bands, avgs)
        plt.title('Average Adjustment Factor by SKU Band')
        plt.ylabel('Avg Adjustment Factor')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avgs):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    # Plot 2: Distribution of adjustment direction by band
    plt.subplot(2, 2, 2)
    
    # Setup data
    bands = sorted(adjustments_df['sku_band'].unique())
    upward_pct = []
    downward_pct = []
    neutral_pct = []
    
    for band in bands:
        band_df = adjustments_df[adjustments_df['sku_band'] == band]
        total = len(band_df)
        
        if total > 0:
            # Calculate percentages
            up = len(band_df[band_df['adjustment_factor'] > 1.0]) / total * 100
            neutral = len(band_df[band_df['adjustment_factor'] == 1.0]) / total * 100
            down = len(band_df[band_df['adjustment_factor'] < 1.0]) / total * 100
            
            upward_pct.append(up)
            neutral_pct.append(neutral)
            downward_pct.append(down)
        else:
            upward_pct.append(0)
            neutral_pct.append(0)
            downward_pct.append(0)
    
    # Create stacked bar chart
    x = np.arange(len(bands))
    width = 0.35
    
    p1 = plt.bar(x, upward_pct, width, label='Upward (>1.0)')
    p2 = plt.bar(x, neutral_pct, width, bottom=upward_pct, label='No Change (=1.0)')
    p3 = plt.bar(x, downward_pct, width, bottom=[upward_pct[i] + neutral_pct[i] for i in range(len(upward_pct))], label='Downward (<1.0)')
    
    plt.title('Adjustment Direction by SKU Band')
    plt.ylabel('Percentage')
    plt.xlabel('SKU Band')
    plt.xticks(x, bands)
    plt.legend()
    
    # Plot 3: Context-specific adjustments by band
    plt.subplot(2, 2, 3)
    
    # Setup data - we'll compare bands A and E (if available)
    if 'A' in bands and 'E' in bands and 'is_holiday' in adjustments_df.columns and 'is_promotion' in adjustments_df.columns:
        band_A = adjustments_df[adjustments_df['sku_band'] == 'A']
        band_E = adjustments_df[adjustments_df['sku_band'] == 'E']
        
        contexts = ['Regular', 'Holiday', 'Promotion']
        A_avgs = []
        E_avgs = []
        
        # Regular days
        A_reg = band_A[(~band_A['is_holiday']) & (~band_A['is_promotion'])]
        E_reg = band_E[(~band_E['is_holiday']) & (~band_E['is_promotion'])]
        A_avgs.append(A_reg['adjustment_factor'].mean() if not A_reg.empty else 0)
        E_avgs.append(E_reg['adjustment_factor'].mean() if not E_reg.empty else 0)
        
        # Holidays
        A_holiday = band_A[band_A['is_holiday']]
        E_holiday = band_E[band_E['is_holiday']]
        A_avgs.append(A_holiday['adjustment_factor'].mean() if not A_holiday.empty else 0)
        E_avgs.append(E_holiday['adjustment_factor'].mean() if not E_holiday.empty else 0)
        
        # Promotions
        A_promo = band_A[band_A['is_promotion']]
        E_promo = band_E[band_E['is_promotion']]
        A_avgs.append(A_promo['adjustment_factor'].mean() if not A_promo.empty else 0)
        E_avgs.append(E_promo['adjustment_factor'].mean() if not E_promo.empty else 0)
        
        # Create grouped bar chart
        x = np.arange(len(contexts))
        width = 0.35
        
        plt.bar(x - width/2, A_avgs, width, label='Band A (Fast)')
        plt.bar(x + width/2, E_avgs, width, label='Band E (Slow)')
        
        plt.title('Context-Specific Adjustments: Fast vs. Slow SKUs')
        plt.ylabel('Avg Adjustment Factor')
        plt.xticks(x, contexts)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Histogram comparison of adjustment factors
    plt.subplot(2, 2, 4)
    
    # Setup data
    if len(bands) >= 2:
        # Select the first and last band (usually A and E)
        first_band = bands[0]
        last_band = bands[-1]
        
        first_df = adjustments_df[adjustments_df['sku_band'] == first_band]
        last_df = adjustments_df[adjustments_df['sku_band'] == last_band]
        
        plt.hist(first_df['adjustment_factor'], bins=20, alpha=0.5, label=f'Band {first_band}')
        plt.hist(last_df['adjustment_factor'], bins=20, alpha=0.5, label=f'Band {last_band}')
        
        plt.title(f'Adjustment Factor Distribution: Band {first_band} vs {last_band}')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'band_comparisons.png'))
    plt.close()
    
    logger.info(f"Saved band comparison visualizations to {output_dir}")
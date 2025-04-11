"""
Visualization Utilities - Functions for visualizing forecast adjustments and 
demonstrating learning patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional
import logging


def visualize_sku_improvements(
    original_forecasts: pd.DataFrame,
    adjusted_forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    sku_patterns: Dict[str, str],
    holiday_data: Optional[pd.DataFrame] = None,
    promotion_data: Optional[pd.DataFrame] = None,
    output_dir: str = "output/visualizations",
    logger: Optional[logging.Logger] = None
):
    """
    Generate visualizations showing how the RL agent improved forecasts for different pattern types.
    
    Args:
        original_forecasts: DataFrame with original ML forecasts
        adjusted_forecasts: DataFrame with RL-adjusted forecasts
        actuals: DataFrame with actual values
        sku_patterns: Dictionary mapping SKUs to their pattern types
        holiday_data: Optional DataFrame with holiday information
        promotion_data: Optional DataFrame with promotion information
        output_dir: Directory to save visualizations
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger("Visualization")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating SKU improvement visualizations")
    
    # Get column names for SKU ID (could be 'sku_id' or 'sku')
    sku_col_orig = 'sku_id' if 'sku_id' in original_forecasts.columns else 'sku'
    sku_col_adj = 'sku_id' if 'sku_id' in adjusted_forecasts.columns else 'sku'
    sku_col_act = 'sku_id' if 'sku_id' in actuals.columns else 'sku'
    
    logger.info(f"Using SKU columns - Original: {sku_col_orig}, Adjusted: {sku_col_adj}, Actuals: {sku_col_act}")
    
    # Process holiday data
    holiday_dates = set()
    if holiday_data is not None and not holiday_data.empty:
        holiday_data['date'] = pd.to_datetime(holiday_data['date'])
        holiday_dates = set(holiday_data['date'].dt.strftime('%Y-%m-%d'))
    
    # Process promotion data
    sku_promotion_periods = {}
    if promotion_data is not None and not promotion_data.empty:
        promo_sku_col = 'sku_id' if 'sku_id' in promotion_data.columns else 'sku'
        for _, row in promotion_data.iterrows():
            sku = row[promo_sku_col]
            start = pd.to_datetime(row['start_date'])
            end = pd.to_datetime(row['end_date'])
            
            if sku not in sku_promotion_periods:
                sku_promotion_periods[sku] = []
            
            sku_promotion_periods[sku].append((start, end))
    
    # Group SKUs by pattern type
    pattern_skus = {}
    for sku, pattern in sku_patterns.items():
        if pattern not in pattern_skus:
            pattern_skus[pattern] = []
        pattern_skus[pattern].append(sku)
    
    # Create one visualization per pattern type
    for pattern, skus in pattern_skus.items():
        if not skus:
            continue
        
        # Sort SKUs by data volume for better visualization
        sku_data_counts = {}
        for sku in skus:
            sku_data_counts[sku] = len(actuals[actuals[sku_col_act] == sku])
        
        sorted_skus = sorted(sku_data_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 3 SKUs with most data (or fewer if not enough data)
        selected_skus = [sku for sku, count in sorted_skus[:3] if count > 0]
        
        if not selected_skus:
            logger.warning(f"No SKUs with data found for pattern '{pattern}'")
            continue
        
        # Create a visualization for this pattern type
        plt.figure(figsize=(18, 15))
        plt.suptitle(f"Pattern Type: {pattern.upper()} - Forecast Improvements", fontsize=16)
        
        for i, sku in enumerate(selected_skus):
            # Get data for this SKU
            sku_original = original_forecasts[original_forecasts[sku_col_orig] == sku]
            sku_adjusted = adjusted_forecasts[adjusted_forecasts[sku_col_adj] == sku]
            sku_actuals = actuals[actuals[sku_col_act] == sku]
            
            # Convert dates to datetime
            if 'date' in sku_original.columns:
                sku_original['date'] = pd.to_datetime(sku_original['date'])
            if 'target_date' in sku_original.columns:
                sku_original['target_date'] = pd.to_datetime(sku_original['target_date'])
            
            if 'date' in sku_adjusted.columns:
                sku_adjusted['date'] = pd.to_datetime(sku_adjusted['date'])
            if 'target_date' in sku_adjusted.columns:
                sku_adjusted['target_date'] = pd.to_datetime(sku_adjusted['target_date'])
            
            if 'date' in sku_actuals.columns:
                sku_actuals['date'] = pd.to_datetime(sku_actuals['date'])
            
            # Plot the data
            ax = plt.subplot(3, 1, i + 1)
            plt.title(f"SKU: {sku}")
            
            date_col = 'date' if 'date' in sku_actuals.columns else 'target_date'
            
            # Plot actual values
            if not sku_actuals.empty and date_col in sku_actuals.columns:
                plt.plot(sku_actuals[date_col], sku_actuals['actual_value'], 'k-', label='Actual')
            
            # Plot original forecast
            forecast_col = 'original_forecast' if 'original_forecast' in sku_original.columns else 'forecast'
            if not sku_original.empty and date_col in sku_original.columns:
                plt.plot(sku_original[date_col], sku_original[forecast_col], 'b--', label='Original Forecast')
            
            # Plot adjusted forecast
            if not sku_adjusted.empty and date_col in sku_adjusted.columns:
                plt.plot(sku_adjusted[date_col], sku_adjusted['adjusted_forecast'], 'r-', label='RL Adjusted')
            
            # Add pattern-specific annotations
            if pattern == "underbias":
                plt.annotate("Consistent underbias corrected", 
                            xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center',
                            bbox=dict(boxstyle="round", fc="yellow", alpha=0.3))
                
                # Calculate average improvement
                if not sku_original.empty and not sku_adjusted.empty and not sku_actuals.empty:
                    # Join data on date
                    merged = pd.merge(
                        pd.merge(
                            sku_original[[date_col, forecast_col]], 
                            sku_adjusted[[date_col, 'adjusted_forecast']], 
                            on=date_col
                        ),
                        sku_actuals[[date_col, 'actual_value']],
                        on=date_col
                    )
                    
                    if not merged.empty:
                        # Calculate MAPE
                        merged['original_mape'] = abs(merged[forecast_col] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
                        merged['adjusted_mape'] = abs(merged['adjusted_forecast'] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
                        
                        avg_orig_mape = merged['original_mape'].mean()
                        avg_adj_mape = merged['adjusted_mape'].mean()
                        
                        improvement = (avg_orig_mape - avg_adj_mape) / avg_orig_mape * 100
                        
                        plt.annotate(f"MAPE Improvement: {improvement:.1f}%", 
                                    xy=(0.5, 0.82), xycoords='axes fraction',
                                    ha='center', va='center',
                                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            
            elif pattern == "promo_holiday":
                # Find and mark holidays
                if date_col in sku_original.columns:
                    for date_str in holiday_dates:
                        date = pd.to_datetime(date_str)
                        if date >= sku_original[date_col].min() and date <= sku_original[date_col].max():
                            plt.axvspan(date - pd.Timedelta(days=0.5), 
                                        date + pd.Timedelta(days=0.5), 
                                        alpha=0.2, color='green', label='_Holiday')
                
                # Find and mark promotions
                if sku in sku_promotion_periods and date_col in sku_original.columns:
                    for start, end in sku_promotion_periods[sku]:
                        if end >= sku_original[date_col].min() and start <= sku_original[date_col].max():
                            plt.axvspan(max(start, sku_original[date_col].min()), 
                                        min(end, sku_original[date_col].max()), 
                                        alpha=0.2, color='red', label='_Promotion')
                
                plt.annotate("Holiday/Promo adjustment", 
                            xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center',
                            bbox=dict(boxstyle="round", fc="yellow", alpha=0.3))
                
                # Calculate holiday/promo specific improvements
                if 'is_holiday' in sku_original.columns and 'is_promotion' in sku_original.columns:
                    # Get holiday and promo data points
                    special_days = sku_original[(sku_original['is_holiday'] == True) | 
                                                (sku_original['is_promotion'] == True)]
                    
                    if not special_days.empty:
                        special_dates = set(special_days[date_col])
                        
                        # Merge data for special days
                        merged = pd.merge(
                            pd.merge(
                                sku_original[sku_original[date_col].isin(special_dates)][[date_col, forecast_col]], 
                                sku_adjusted[sku_adjusted[date_col].isin(special_dates)][[date_col, 'adjusted_forecast']], 
                                on=date_col
                            ),
                            sku_actuals[sku_actuals[date_col].isin(special_dates)][[date_col, 'actual_value']],
                            on=date_col
                        )
                        
                        if not merged.empty:
                            # Calculate MAPE
                            merged['original_mape'] = abs(merged[forecast_col] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
                            merged['adjusted_mape'] = abs(merged['adjusted_forecast'] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
                            
                            avg_orig_mape = merged['original_mape'].mean()
                            avg_adj_mape = merged['adjusted_mape'].mean()
                            
                            improvement = (avg_orig_mape - avg_adj_mape) / avg_orig_mape * 100
                            
                            plt.annotate(f"Holiday/Promo MAPE Improvement: {improvement:.1f}%", 
                                        xy=(0.5, 0.82), xycoords='axes fraction',
                                        ha='center', va='center',
                                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            
            elif pattern == "day_pattern":
                # Mark weekends
                if date_col in sku_actuals.columns:
                    for date in pd.date_range(sku_actuals[date_col].min(), sku_actuals[date_col].max()):
                        if date.weekday() >= 5:  # Weekend
                            plt.axvspan(date - pd.Timedelta(days=0.5), 
                                        date + pd.Timedelta(days=0.5), 
                                        alpha=0.1, color='blue', label='_Weekend')
                
                plt.annotate("Day of week pattern learned", 
                            xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center',
                            bbox=dict(boxstyle="round", fc="yellow", alpha=0.3))
                
                # Calculate weekend-specific improvements
                if 'is_weekend' in sku_original.columns:
                    # Get weekend data points
                    weekend_days = sku_original[sku_original['is_weekend'] == True]
                    
                    if not weekend_days.empty:
                        weekend_dates = set(weekend_days[date_col])
                        
                        # Merge data for weekends
                        merged = pd.merge(
                            pd.merge(
                                sku_original[sku_original[date_col].isin(weekend_dates)][[date_col, forecast_col]], 
                                sku_adjusted[sku_adjusted[date_col].isin(weekend_dates)][[date_col, 'adjusted_forecast']], 
                                on=date_col
                            ),
                            sku_actuals[sku_actuals[date_col].isin(weekend_dates)][[date_col, 'actual_value']],
                            on=date_col
                        )
                        
                        if not merged.empty:
                            # Calculate MAPE
                            merged['original_mape'] = abs(merged[forecast_col] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
                            merged['adjusted_mape'] = abs(merged['adjusted_forecast'] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
                            
                            avg_orig_mape = merged['original_mape'].mean()
                            avg_adj_mape = merged['adjusted_mape'].mean()
                            
                            improvement = (avg_orig_mape - avg_adj_mape) / avg_orig_mape * 100
                            
                            plt.annotate(f"Weekend MAPE Improvement: {improvement:.1f}%", 
                                        xy=(0.5, 0.82), xycoords='axes fraction',
                                        ha='center', va='center',
                                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            
            # Remove duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(output_dir, f"{pattern}_learning_showcase.png"))
        plt.close()
    
    logger.info(f"Visualization complete. Images saved to {output_dir}")


def calculate_context_specific_improvements(
    original_forecasts: pd.DataFrame,
    adjusted_forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Calculate improvement metrics for different contexts.
    
    Args:
        original_forecasts: DataFrame with original ML forecasts
        adjusted_forecasts: DataFrame with RL-adjusted forecasts
        actuals: DataFrame with actual values
        logger: Optional logger instance
        
    Returns:
        Dictionary of metrics by context
    """
    if logger is None:
        logger = logging.getLogger("Metrics")
    
    metrics = {
        "overall": {"original_mape": 0, "adjusted_mape": 0, "count": 0},
        "holiday": {"original_mape": 0, "adjusted_mape": 0, "count": 0},
        "promotion": {"original_mape": 0, "adjusted_mape": 0, "count": 0},
        "weekend": {"original_mape": 0, "adjusted_mape": 0, "count": 0},
        "weekday": {"original_mape": 0, "adjusted_mape": 0, "count": 0},
        "by_pattern": {}
    }
    
    # Get column names
    # SKU column
    sku_col_orig = 'sku_id' if 'sku_id' in original_forecasts.columns else 'sku'
    sku_col_adj = 'sku_id' if 'sku_id' in adjusted_forecasts.columns else 'sku'
    sku_col_act = 'sku_id' if 'sku_id' in actuals.columns else 'sku'
    
    # Date column
    date_col_orig = next((col for col in ['date', 'target_date'] if col in original_forecasts.columns), None)
    date_col_adj = next((col for col in ['date', 'target_date'] if col in adjusted_forecasts.columns), None)
    date_col_act = next((col for col in ['date', 'target_date'] if col in actuals.columns), None)
    
    # Forecast column
    forecast_col = 'original_forecast' if 'original_forecast' in original_forecasts.columns else 'forecast'
    if forecast_col not in original_forecasts.columns:
        logger.warning(f"Forecast column not found in {original_forecasts.columns}")
        return metrics
    
    # Check if we have all necessary columns
    if date_col_orig is None or date_col_adj is None or date_col_act is None:
        logger.error("Missing date columns - cannot merge datasets")
        return metrics
    
    logger.info(f"Using columns - SKU: {sku_col_orig}/{sku_col_adj}/{sku_col_act}, "
               f"Date: {date_col_orig}/{date_col_adj}/{date_col_act}, "
               f"Forecast: {forecast_col}")
    
    # Ensure date columns are in datetime format
    original_forecasts[date_col_orig] = pd.to_datetime(original_forecasts[date_col_orig])
    adjusted_forecasts[date_col_adj] = pd.to_datetime(adjusted_forecasts[date_col_adj])
    actuals[date_col_act] = pd.to_datetime(actuals[date_col_act])
    
    # Check for context columns and add them if missing
    context_cols = ['is_holiday', 'is_promotion', 'is_weekend']
    for col in context_cols:
        if col not in original_forecasts.columns:
            logger.warning(f"Adding missing column '{col}' with default value False")
            original_forecasts[col] = False
    
    # Check for pattern_type column
    if 'pattern_type' not in actuals.columns:
        logger.warning("No pattern_type column in actuals data - pattern-specific metrics will be skipped")
        has_pattern_types = False
    else:
        has_pattern_types = True
    
    # Select required columns
    orig_cols = [date_col_orig, sku_col_orig, forecast_col] + context_cols
    adj_cols = [date_col_adj, sku_col_adj, 'adjusted_forecast']
    act_cols = [date_col_act, sku_col_act, 'actual_value']
    if has_pattern_types:
        act_cols.append('pattern_type')
    
    # Merge datasets with flexible column names
    try:
        # First merge original and adjusted
        merged = pd.merge(
            original_forecasts[orig_cols],
            adjusted_forecasts[adj_cols],
            left_on=[date_col_orig, sku_col_orig],
            right_on=[date_col_adj, sku_col_adj]
        )
        
        # Then merge with actuals
        merged = pd.merge(
            merged,
            actuals[act_cols],
            left_on=[date_col_orig, sku_col_orig],
            right_on=[date_col_act, sku_col_act]
        )
    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        return metrics
    
    if merged.empty:
        logger.warning("No matching data found for calculating metrics")
        return metrics
    
    # Calculate MAPE for each row
    merged['original_mape'] = abs(merged[forecast_col] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
    merged['adjusted_mape'] = abs(merged['adjusted_forecast'] - merged['actual_value']) / merged['actual_value'].replace(0, 1)
    
    # Calculate overall metrics
    metrics["overall"]["original_mape"] = merged['original_mape'].mean()
    metrics["overall"]["adjusted_mape"] = merged['adjusted_mape'].mean()
    metrics["overall"]["count"] = len(merged)
    
    # Calculate holiday metrics
    holiday_data = merged[merged['is_holiday'] == True]
    if not holiday_data.empty:
        metrics["holiday"]["original_mape"] = holiday_data['original_mape'].mean()
        metrics["holiday"]["adjusted_mape"] = holiday_data['adjusted_mape'].mean()
        metrics["holiday"]["count"] = len(holiday_data)
    
    # Calculate promotion metrics
    promo_data = merged[merged['is_promotion'] == True]
    if not promo_data.empty:
        metrics["promotion"]["original_mape"] = promo_data['original_mape'].mean()
        metrics["promotion"]["adjusted_mape"] = promo_data['adjusted_mape'].mean()
        metrics["promotion"]["count"] = len(promo_data)
    
    # Calculate weekend metrics
    weekend_data = merged[merged['is_weekend'] == True]
    if not weekend_data.empty:
        metrics["weekend"]["original_mape"] = weekend_data['original_mape'].mean()
        metrics["weekend"]["adjusted_mape"] = weekend_data['adjusted_mape'].mean()
        metrics["weekend"]["count"] = len(weekend_data)
    
    # Calculate weekday metrics
    weekday_data = merged[merged['is_weekend'] == False]
    if not weekday_data.empty:
        metrics["weekday"]["original_mape"] = weekday_data['original_mape'].mean()
        metrics["weekday"]["adjusted_mape"] = weekday_data['adjusted_mape'].mean()
        metrics["weekday"]["count"] = len(weekday_data)
    
    # Calculate metrics by pattern type
    for pattern_type in merged['pattern_type'].unique():
        pattern_data = merged[merged['pattern_type'] == pattern_type]
        metrics["by_pattern"][pattern_type] = {
            "original_mape": pattern_data['original_mape'].mean(),
            "adjusted_mape": pattern_data['adjusted_mape'].mean(),
            "count": len(pattern_data)
        }
    
    # Calculate improvement percentages
    for context in metrics:
        if context != "by_pattern":
            if metrics[context]["count"] > 0:
                metrics[context]["improvement"] = ((metrics[context]["original_mape"] - 
                                                  metrics[context]["adjusted_mape"]) / 
                                                 metrics[context]["original_mape"] * 100)
        else:
            for pattern in metrics["by_pattern"]:
                pattern_metrics = metrics["by_pattern"][pattern]
                if pattern_metrics["count"] > 0:
                    pattern_metrics["improvement"] = ((pattern_metrics["original_mape"] - 
                                                      pattern_metrics["adjusted_mape"]) / 
                                                     pattern_metrics["original_mape"] * 100)
    
    return metrics


def plot_metrics_summary(metrics: Dict, output_dir: str = "output/visualizations", logger: Optional[logging.Logger] = None):
    """
    Create summary plots of forecast adjustment metrics.
    
    Args:
        metrics: Dictionary of metrics by context
        output_dir: Directory to save visualizations
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger("Metrics")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Improvement by context
    plt.figure(figsize=(12, 8))
    
    contexts = ['Overall', 'Holiday', 'Promotion', 'Weekend', 'Weekday']
    improvements = [
        metrics['overall'].get('improvement', 0),
        metrics['holiday'].get('improvement', 0),
        metrics['promotion'].get('improvement', 0),
        metrics['weekend'].get('improvement', 0),
        metrics['weekday'].get('improvement', 0)
    ]
    
    # Filter out contexts with no data
    valid_contexts = []
    valid_improvements = []
    for c, i in zip(contexts, improvements):
        if i != 0:
            valid_contexts.append(c)
            valid_improvements.append(i)
    
    plt.bar(valid_contexts, valid_improvements, color=['blue', 'green', 'red', 'purple', 'orange'][:len(valid_contexts)])
    plt.title('MAPE Improvement by Context (%)', fontsize=14)
    plt.ylabel('Improvement %')
    plt.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(valid_improvements):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_by_context.png'))
    plt.close()
    
    # Plot 2: Improvement by pattern type
    if metrics['by_pattern']:
        plt.figure(figsize=(12, 8))
        
        patterns = list(metrics['by_pattern'].keys())
        pattern_improvements = [metrics['by_pattern'][p].get('improvement', 0) for p in patterns]
        
        plt.bar(patterns, pattern_improvements, color=['blue', 'green', 'red'])
        plt.title('MAPE Improvement by Pattern Type (%)', fontsize=14)
        plt.ylabel('Improvement %')
        plt.grid(True, alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(pattern_improvements):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_by_pattern.png'))
        plt.close()
    
    # Plot 3: Original vs Adjusted MAPE by context
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    orig_mape = [
        metrics['overall'].get('original_mape', 0),
        metrics['holiday'].get('original_mape', 0),
        metrics['promotion'].get('original_mape', 0),
        metrics['weekend'].get('original_mape', 0),
        metrics['weekday'].get('original_mape', 0)
    ]
    
    adj_mape = [
        metrics['overall'].get('adjusted_mape', 0),
        metrics['holiday'].get('adjusted_mape', 0),
        metrics['promotion'].get('adjusted_mape', 0),
        metrics['weekend'].get('adjusted_mape', 0),
        metrics['weekday'].get('adjusted_mape', 0)
    ]
    
    # Filter valid contexts
    valid_contexts = []
    valid_orig = []
    valid_adj = []
    for c, o, a in zip(contexts, orig_mape, adj_mape):
        if o != 0 or a != 0:
            valid_contexts.append(c)
            valid_orig.append(o)
            valid_adj.append(a)
    
    x = np.arange(len(valid_contexts))
    width = 0.35
    
    plt.bar(x - width/2, valid_orig, width, label='Original ML Forecast', color='royalblue')
    plt.bar(x + width/2, valid_adj, width, label='RL Adjusted Forecast', color='lightcoral')
    
    plt.title('MAPE Before vs After Adjustment by Context', fontsize=14)
    plt.ylabel('MAPE')
    plt.xticks(x, valid_contexts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add percentage values on bars
    for i, (o, a) in enumerate(zip(valid_orig, valid_adj)):
        plt.text(i - width/2, o + 0.01, f"{o:.2f}", ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, a + 0.01, f"{a:.2f}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mape_comparison_by_context.png'))
    plt.close()
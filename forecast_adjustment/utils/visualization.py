"""
Simplified utils/visualization.py - keeping only minimal necessary plotting functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional
import logging


def plot_training_metrics(
    metrics: Dict,
    output_dir: str = "output/visualizations",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Plot simplified training metrics from a training run.
    
    Args:
        metrics: Dictionary of training metrics
        output_dir: Directory to save visualizations
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger("Visualization")
        
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating simplified training metrics visualization")
    
    # Create visualization with just the three important plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Overall Training Score
    plt.subplot(1, 3, 1)
    plt.plot(metrics['scores'])
    plt.title('Overall Training Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: MAPE Improvement
    plt.subplot(1, 3, 2)
    plt.plot(metrics['mape_improvements'])
    plt.title('Overall MAPE Improvement')
    plt.xlabel('Episode')
    plt.ylabel('Improvement Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Bias Improvement
    plt.subplot(1, 3, 3)
    plt.plot(metrics['bias_improvements'])
    plt.title('Overall Bias Improvement')
    plt.xlabel('Episode')
    plt.ylabel('Improvement Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    logger.info(f"Saved training metrics visualization to {output_dir}")
"""
Training pipeline for the Forecast Adjustment RL system.
Handles training loop, episode execution, and evaluation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import json
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from models.agent import ForecastAdjustmentAgent
from environment.state import StateBuilder
from environment.actions import ActionHandler
from environment.reward import RewardCalculator
from data.data_loader import DataProvider

logger = logging.getLogger(__name__)

class ForecastAdjustmentTrainer:
    """
    Trainer for the Forecast Adjustment RL agent.
    Handles training loop, episode execution, and evaluation.
    """
    
    def __init__(self, config):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config['TRAINING_CONFIG']
        self.system_config = config['SYSTEM_CONFIG']
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.system_config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Import utils here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from utils import ensure_dir
        
        # Create necessary directories
        ensure_dir(self.training_config['model_dir'])
        ensure_dir('data')
        ensure_dir('plots')
        
        # Initialize components
        self.data_provider = DataProvider(config)
        self.state_builder = StateBuilder(config)
        self.action_handler = ActionHandler(config)
        self.reward_calculator = RewardCalculator(config)
        
        # Initialize RL agent
        self.agent = ForecastAdjustmentAgent(
            self.state_builder.get_state_dim(),
            self.action_handler.get_action_dim(),
            config
        )
        
        # Metrics tracking
        self.episode_rewards = []
        self.category_band_metrics = defaultdict(list)
        self.adjustment_history = []
    
    def run_episode(self, start_date, end_date, categories, bands, training=True):
        """
        Run a single episode from start_date to end_date.
        
        Args:
            start_date: Episode start date
            end_date: Episode end date
            categories: List of categories to include
            bands: List of bands to include
            training: Whether in training mode
            
        Returns:
            Dictionary with episode metrics
        """
        current_date = start_date
        episode_rewards = []
        adjustment_decisions = []
        
        while current_date <= end_date:
            logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
            
            # Process each category and band
            for category in categories:
                for band in bands:
                    # Build state
                    state = self.state_builder.build_state(
                        self.data_provider, category, band, current_date)
                    
                    # Select action
                    action_idx, action_probs = self.agent.select_action(state, training)
                    
                    # Get adjustment factor
                    adjustment_factor = self.action_handler.get_adjustment_factor(action_idx)
                    
                    # Get previous adjustment (if any)
                    previous_adjustment = self.data_provider.get_previous_adjustment(category, band)
                    
                    # Calculate immediate reward
                    immediate_reward = self.reward_calculator.calculate_immediate_reward(
                        self.data_provider, category, band, current_date,
                        adjustment_factor, previous_adjustment
                    )
                    
                    # Apply adjustment
                    adjustment_info = self.action_handler.apply_adjustment(
                        self.data_provider, category, band, current_date, action_idx
                    )
                    
                    # Store adjustment decision for later evaluation
                    adjustment_decisions.append({
                        'category': category,
                        'band': band,
                        'date': current_date,
                        'state': state.tolist(),
                        'action_idx': action_idx,
                        'adjustment_factor': adjustment_factor,
                        'immediate_reward': immediate_reward
                    })
                    
                    # Add immediate reward to episode rewards
                    episode_rewards.append(immediate_reward)
                    
                    # Log adjustment
                    logger.debug(f"Applied {adjustment_factor:.2f} to {category}-{band} " +
                               f"(immediate reward: {immediate_reward:.4f})")
            
            # Evaluate previous adjustments (when actuals are available)
            self._evaluate_previous_adjustments(current_date)
            
            # Update policy periodically
            if training and current_date.day % self.config['AGENT_CONFIG']['update_frequency'] == 0:
                self.agent.update_policy()
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Calculate episode metrics
        episode_metrics = {
            'total_adjustments': len(adjustment_decisions),
            'mean_immediate_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'adjustment_decisions': adjustment_decisions
        }
        
        return episode_metrics
    
    def _evaluate_previous_adjustments(self, evaluation_date):
        """
        Evaluate previous adjustments when actuals become available.
        
        Args:
            evaluation_date: Date when actuals are available
        """
        # In a real system, would look up adjustments that can now be evaluated
        # For POC, simulate with 7-day delayed feedback
        
        # Get adjustments from 7 days ago
        adjustment_date = evaluation_date - timedelta(days=7)
        
        # Check all adjustments from that date
        adj = self.data_provider.adjustments[
            self.data_provider.adjustments['date'] == adjustment_date
        ]
        
        for _, row in adj.iterrows():
            category = row['category']
            band = row['band']
            adjustment_factor = row['adjustment_factor']
            
            # Calculate delayed reward
            delayed_reward = self.reward_calculator.calculate_delayed_reward(
                self.data_provider, category, band,
                adjustment_date, evaluation_date, adjustment_factor
            )
            
            # In a real system, would retrieve the original state and action
            # For POC, create a placeholder state
            state = self.state_builder.build_state(
                self.data_provider, category, band, adjustment_date)
            
            # Get action index
            action_idx = self.action_handler.get_action_idx(adjustment_factor)
            
            # Build next state (current state for this category-band)
            next_state = self.state_builder.build_state(
                self.data_provider, category, band, evaluation_date)
            
            # Store experience for learning
            self.agent.store_experience(
                state, action_idx, delayed_reward, next_state, False)
            
            # Track metrics by category-band
            self.category_band_metrics[(category, band)].append({
                'date': adjustment_date,
                'evaluation_date': evaluation_date,
                'adjustment_factor': adjustment_factor,
                'delayed_reward': delayed_reward
            })
            
            # Log evaluation
            logger.debug(f"Evaluated adjustment for {category}-{band} on " +
                       f"{adjustment_date.strftime('%Y-%m-%d')}: " +
                       f"reward = {delayed_reward:.4f}")
    
    def train(self, num_episodes=None, categories=None, bands=None):
        """
        Train the RL agent.
        
        Args:
            num_episodes: Number of episodes to train (defaults to config value)
            categories: List of categories to include (defaults to first 5)
            bands: List of bands to include (defaults to A, B, C)
            
        Returns:
            Dictionary with training metrics
        """
        if num_episodes is None:
            num_episodes = self.training_config['num_episodes']
        
        if categories is None:
            # Use first 5 categories for POC
            all_categories = self.data_provider.ml_forecasts['category'].unique()
            categories = sorted(all_categories)[:5]
        
        if bands is None:
            bands = ['A', 'B', 'C']
        
        logger.info(f"Starting training for {num_episodes} episodes")
        logger.info(f"Categories: {categories}")
        logger.info(f"Bands: {bands}")
        
        # Get date range for forecasts (all available dates in data)
        all_dates = sorted(self.data_provider.ml_forecasts['date'].unique())
        
        # We need at least 90 days of data for proper training
        if len(all_dates) < 90:
            logger.warning(f"Less than 90 days of data available ({len(all_dates)} days). " 
                        f"Training will be limited.")
        
        # Determine evaluation window (number of days to predict forward)
        eval_window = self.training_config.get('eval_window_days', 35)  # Default to 5 weeks (35 days)
        
        # Calculate number of training days available (total days - evaluation window)
        training_days = max(0, len(all_dates) - eval_window)
        
        logger.info(f"Total days in data: {len(all_dates)}")
        logger.info(f"Evaluation window: {eval_window} days")
        logger.info(f"Training days available: {training_days}")
        
        # Split dates into training periods based on evaluation windows (weeks)
        # Each training sample will consist of historical data and a specific forecast date
        # We'll train the agent to make the right adjustment for each forecast date
        
        # Create forecast date samples for each week (WoM1, WoM2, etc.)
        forecast_dates_by_wom = {1: [], 2: [], 3: [], 4: []}
        
        for i in range(training_days):
            forecast_date = all_dates[i + eval_window]  # Use the date after the evaluation window
            from utils import get_week_of_month
            wom = get_week_of_month(forecast_date)
            forecast_dates_by_wom[wom].append({
                'forecast_date': forecast_date,
                'history_start': all_dates[max(0, i - 90)],  # Use up to 90 days of history
                'history_end': all_dates[i]
            })
        
        # Main training loop
        for episode in range(num_episodes):
            logger.info(f"Episode {episode+1}/{num_episodes}")
            episode_rewards = []
            
            # Process each Week of Month to ensure balanced learning
            for wom, date_samples in forecast_dates_by_wom.items():
                logger.info(f"Processing Week of Month {wom} samples: {len(date_samples)} dates")
                
                # Shuffle date samples to prevent overfitting to time sequence
                random.shuffle(date_samples)
                
                # Limit samples per episode if there are too many
                max_samples_per_wom = self.training_config.get('max_samples_per_wom', 10)
                date_samples_to_use = date_samples[:max_samples_per_wom]
                
                for date_sample in date_samples_to_use:
                    forecast_date = date_sample['forecast_date']
                    
                    # Process each category and band
                    for category in categories:
                        for band in bands:
                            # Build state using historical data up to history_end
                            state = self.state_builder.build_state(
                                self.data_provider, category, band, date_sample['history_end'])
                            
                            # Select action
                            action_idx, action_probs = self.agent.select_action(state, training=True)
                            
                            # Get adjustment factor
                            adjustment_factor = self.action_handler.get_adjustment_factor(action_idx)
                            
                            # Get previous adjustment (if any)
                            previous_adjustment = self.data_provider.get_previous_adjustment(category, band)
                            
                            # Calculate immediate reward
                            immediate_reward = self.reward_calculator.calculate_immediate_reward(
                                self.data_provider, category, band, forecast_date,
                                adjustment_factor, previous_adjustment
                            )
                            
                            # Apply adjustment to simulate its effect
                            adjustment_info = self.action_handler.apply_adjustment(
                                self.data_provider, category, band, forecast_date, action_idx
                            )
                            
                            # Calculate delayed reward based on actual performance
                            # This simulates what would happen after we get actuals
                            delayed_reward = self.reward_calculator.calculate_delayed_reward(
                                self.data_provider, category, band,
                                forecast_date, forecast_date + timedelta(days=7), adjustment_factor
                            )
                            
                            # Calculate total reward
                            total_reward = immediate_reward + delayed_reward
                            
                            # Store experience in the agent's memory
                            # For next state, we use the state after the forecast date
                            next_state = self.state_builder.build_state(
                                self.data_provider, category, band, forecast_date + timedelta(days=1))
                            
                            self.agent.store_experience(
                                state, action_idx, total_reward, next_state, True)
                            
                            # Track reward
                            episode_rewards.append(total_reward)
                            
                            # Store for metrics
                            self.category_band_metrics[(category, band)].append({
                                'date': forecast_date,
                                'wom': wom,
                                'adjustment_factor': adjustment_factor,
                                'immediate_reward': immediate_reward,
                                'delayed_reward': delayed_reward,
                                'total_reward': total_reward,
                                'mape': self.data_provider.get_historical_mape(
                                    category, band, forecast_date, before_adjustment=False),
                                'bias': self.data_provider.get_historical_bias(
                                    category, band, forecast_date, before_adjustment=False)
                            })
                            
                            # Track adjustment decision
                            self.adjustment_history.append({
                                'category': category,
                                'band': band,
                                'date': forecast_date,
                                'wom': wom,
                                'action_idx': action_idx,
                                'adjustment_factor': adjustment_factor,
                                'reward': total_reward
                            })
                            
                    # Update policy after processing each forecast date
                    if len(self.agent.memory) >= self.agent.batch_size:
                        self.agent.update_policy()
            
            # Calculate episode metrics
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            self.episode_rewards.append(mean_reward)
            
            logger.info(f"Episode {episode+1} complete: Mean Reward: {mean_reward:.4f}")
            
            # Evaluate periodically
            if (episode + 1) % self.training_config['eval_frequency'] == 0:
                self._evaluate_model(categories, bands)
            
            # Save model periodically
            if (episode + 1) % self.training_config['save_frequency'] == 0:
                self._save_model(episode + 1)
        
        # Final evaluation and save
        self._evaluate_model(categories, bands)
        self._save_model(num_episodes)
        
        # Plot training progress
        self._plot_training_progress()
        
        return {
            'episodes': num_episodes,
            'final_reward': self.episode_rewards[-1] if self.episode_rewards else 0.0,
            'category_band_metrics': self.category_band_metrics,
            'adjustment_history': self.adjustment_history
        }
    
    def _evaluate_model(self, categories, bands):
        """
        Evaluate the model across different Weeks of Month.
        
        Args:
            categories: List of categories to evaluate
            bands: List of bands to evaluate
        """
        logger.info("Evaluating model performance...")
        
        # Get all dates available for evaluation
        all_dates = sorted(self.data_provider.ml_forecasts['date'].unique())
        
        # Group the last 28 days (4 weeks) by WoM for evaluation
        eval_period = min(28, len(all_dates))
        eval_dates = all_dates[-eval_period:]
        
        # Group dates by Week of Month
        from utils import get_week_of_month
        dates_by_wom = {1: [], 2: [], 3: [], 4: []}
        
        for date in eval_dates:
            wom = get_week_of_month(date)
            dates_by_wom[wom].append(date)
        
        # Track metrics by WoM
        wom_metrics = {1: [], 2: [], 3: [], 4: []}
        overall_metrics = []
        
        # Evaluate each WoM separately
        for wom, dates in dates_by_wom.items():
            if not dates:
                continue
                
            logger.info(f"Evaluating Week of Month {wom}: {len(dates)} dates")
            
            wom_rewards = []
            wom_mape = []
            wom_bias = []
            wom_adjustments = []
            
            for date in dates:
                for category in categories:
                    for band in bands:
                        # Build state
                        state = self.state_builder.build_state(
                            self.data_provider, category, band, date)
                        
                        # Select action without exploration
                        action_idx, action_probs = self.agent.select_action(state, training=False)
                        
                        # Get adjustment factor
                        adjustment_factor = self.action_handler.get_adjustment_factor(action_idx)
                        wom_adjustments.append(adjustment_factor)
                        
                        # Simulate calculating metrics after adjustment
                        mape = self.data_provider.get_historical_mape(
                            category, band, date, before_adjustment=False)
                        bias = self.data_provider.get_historical_bias(
                            category, band, date, before_adjustment=False)
                        
                        # Calculate reward that would be achieved
                        reward = self.reward_calculator.calculate_delayed_reward(
                            self.data_provider, category, band,
                            date, date + timedelta(days=7), adjustment_factor
                        )
                        
                        wom_rewards.append(reward)
                        wom_mape.append(mape)
                        wom_bias.append(bias)
                        
                        # Store for overall metrics
                        overall_metrics.append({
                            'wom': wom,
                            'category': category,
                            'band': band,
                            'date': date,
                            'adjustment_factor': adjustment_factor,
                            'mape': mape,
                            'bias': bias,
                            'reward': reward
                        })
            
            # Calculate average metrics for this WoM
            if wom_rewards:
                avg_reward = np.mean(wom_rewards)
                avg_mape = np.mean(wom_mape)
                avg_bias = np.mean(wom_bias)
                avg_adjustment = np.mean(wom_adjustments)
                
                wom_metrics[wom] = {
                    'reward': avg_reward,
                    'mape': avg_mape,
                    'bias': avg_bias,
                    'adjustment_factor': avg_adjustment,
                    'num_samples': len(wom_rewards)
                }
                
                logger.info(f"WoM {wom} metrics: " +
                        f"Reward={avg_reward:.4f}, MAPE={avg_mape:.4f}, " +
                        f"Bias={avg_bias:.4f}, Avg Adjustment={avg_adjustment:.4f}")
        
        # Log overall evaluation metrics
        if overall_metrics:
            df = pd.DataFrame(overall_metrics)
            
            logger.info("Overall evaluation metrics:")
            logger.info(f"Average Reward: {df['reward'].mean():.4f}")
            logger.info(f"Average MAPE: {df['mape'].mean():.4f}")
            logger.info(f"Average Bias: {df['bias'].mean():.4f}")
            
            # Check for Week 1 pattern
            wom1_adjustments = df[df['wom'] == 1]['adjustment_factor']
            other_adjustments = df[df['wom'] != 1]['adjustment_factor']
            
            if not wom1_adjustments.empty and not other_adjustments.empty:
                wom1_avg = wom1_adjustments.mean()
                other_avg = other_adjustments.mean()
                
                if wom1_avg > other_avg:
                    logger.info(f"Model has learned WoM1 pattern: " +
                            f"WoM1 avg adjustment: {wom1_avg:.4f}, " +
                            f"Other weeks: {other_avg:.4f}")
        
        return wom_metrics
    
    def _save_model(self, episode):
        """
        Save model checkpoint.
        
        Args:
            episode: Current episode number
        """
        model_path = os.path.join(
            self.training_config['model_dir'],
            f"forecast_adjustment_model_ep{episode}.pt"
        )
        
        self.agent.save_model(model_path)
        
        # Save metrics
        metrics_path = os.path.join(
            self.training_config['model_dir'],
            f"metrics_ep{episode}.json"
        )
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'category_band_metrics': {str(k): v for k, v in self.category_band_metrics.items()},
            'episodes_completed': episode
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, default=str)
        
        logger.info(f"Model and metrics saved at episode {episode}")

    def _plot_training_progress(self):
        """Plot and save training progress charts using visualization utilities."""
        # Create plots directory
        import os
        try:
            from visualization_utils import (
                plot_training_progress, 
                plot_forecast_metrics, 
                plot_wom_adjustments,
                plot_metrics_summary,
                create_training_report
            )
        except ImportError:
            logger.warning("visualization_utils module not found. Using basic plotting.")
            # Fall back to basic matplotlib plotting
            self._plot_training_progress_basic()
            return
        
        plots_dir = os.path.join(self.training_config['model_dir'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot episode rewards
        plot_training_progress(self.episode_rewards, save_dir=plots_dir, show=False)
        
        # Prepare data for forecast metrics plot
        if self.category_band_metrics:
            # Extract MAPE and bias data
            all_metrics = []
            for i, reward in enumerate(self.episode_rewards):
                episode_data = {'episode': i + 1, 'reward': reward}
                
                # Get average MAPE and bias for this episode (if available)
                mape_values = []
                bias_values = []
                
                for (cat, band), data in self.category_band_metrics.items():
                    # Simply take data entries that exist - don't try to match with episode dates
                    matching_entries = data[max(0, min(i, len(data)-1)): min(i+1, len(data))]
                    
                    for entry in matching_entries:
                        if 'mape' in entry:
                            mape_values.append(entry['mape'])
                        if 'bias' in entry:
                            bias_values.append(entry['bias'])
                
                if mape_values:
                    episode_data['mape'] = np.mean(mape_values)
                if bias_values:
                    episode_data['bias'] = np.mean(bias_values)
                
                all_metrics.append(episode_data)
            
            import pandas as pd
            metrics_df = pd.DataFrame(all_metrics)
            
            if 'mape' in metrics_df.columns and 'bias' in metrics_df.columns:
                plot_forecast_metrics(metrics_df, save_dir=plots_dir, show=False)
        
            # Extract Week of Month adjustments
            wom_adjustments = {}
            for wom in range(1, 5):
                adjustments = []
                for decision in self.adjustment_history:
                    if 'wom' in decision and decision['wom'] == wom:
                        adjustments.append(decision['adjustment_factor'])
                
                if adjustments:
                    wom_adjustments[f'WoM{wom}'] = np.mean(adjustments)
            
            if wom_adjustments:
                plot_wom_adjustments(wom_adjustments, save_dir=plots_dir, show=False)
            
            # Get unique categories and bands
            categories = set()
            bands = set()
            for cat, band in self.category_band_metrics.keys():
                categories.add(cat)
                bands.add(band)
            
            categories = sorted(list(categories))
            bands = sorted(list(bands))
            
            # Create metrics summary plot
            plot_metrics_summary(self.category_band_metrics, categories, bands, 
                            save_dir=plots_dir, show=False)
            
            # Save metrics for report generation
            metrics_path = os.path.join(self.training_config['model_dir'], f"metrics_final.json")
            metrics = {
                'episode_rewards': self.episode_rewards,
                'category_band_metrics': {str(k): v for k, v in self.category_band_metrics.items()},
                'episodes_completed': len(self.episode_rewards)
            }
            
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, default=str)
            
            # Generate full report
            create_training_report(metrics_path, plots_dir)
        
        logger.info(f"Training progress plots and report saved to {plots_dir}")
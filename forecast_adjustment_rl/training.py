"""
Training pipeline for the Forecast Adjustment RL system.
Handles training loop, episode execution, and evaluation.
"""

import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
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
        
        # Main training loop
        for episode in range(num_episodes):
            # Define episode start and end dates
            episode_length = self.training_config['episode_length_days']
            end_date = datetime.now() + timedelta(days=episode * episode_length)
            start_date = end_date - timedelta(days=episode_length - 1)
            
            logger.info(f"Episode {episode+1}/{num_episodes}: " +
                      f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Run episode
            episode_metrics = self.run_episode(start_date, end_date, categories, bands)
            
            # Track episode rewards
            self.episode_rewards.append(episode_metrics['mean_immediate_reward'])
            
            # Save adjustment history
            self.adjustment_history.extend(episode_metrics['adjustment_decisions'])
            
            # Log episode metrics
            logger.info(f"Episode {episode+1} complete: " +
                      f"Adjustments: {episode_metrics['total_adjustments']}, " +
                      f"Mean Reward: {episode_metrics['mean_immediate_reward']:.4f}")
            
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
        Evaluate the model on a test period.
        
        Args:
            categories: List of categories to evaluate
            bands: List of bands to evaluate
        """
        # Define evaluation period (next week after training)
        episode_length = self.training_config['episode_length_days']
        start_date = datetime.now() + timedelta(days=episode_length * 2)
        end_date = start_date + timedelta(days=episode_length - 1)
        
        logger.info(f"Evaluating model from {start_date.strftime('%Y-%m-%d')} " +
                  f"to {end_date.strftime('%Y-%m-%d')}")
        
        # Run evaluation episode (no training)
        eval_metrics = self.run_episode(start_date, end_date, categories, bands, training=False)
        
        logger.info(f"Evaluation complete: " +
                  f"Adjustments: {eval_metrics['total_adjustments']}, " +
                  f"Mean Reward: {eval_metrics['mean_immediate_reward']:.4f}")
        
        return eval_metrics
    
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
        """Plot and save training progress charts."""
        # Create plots directory
        plots_dir = os.path.join(self.training_config['model_dir'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot episode rewards
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'episode_rewards.png'))
        
        # Plot rewards by category-band
        plt.figure(figsize=(12, 8))
        for (category, band), metrics in self.category_band_metrics.items():
            if len(metrics) > 0:
                rewards = [m['delayed_reward'] for m in metrics]
                plt.plot(rewards, label=f"{category}-{band}")
        
        plt.title('Rewards by Category-Band')
        plt.xlabel('Adjustment')
        plt.ylabel('Delayed Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'category_band_rewards.png'))
        
        logger.info(f"Training progress plots saved to {plots_dir}")
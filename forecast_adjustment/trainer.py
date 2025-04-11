"""
Enhanced Trainer Module - Handles training, evaluation, and forecast generation
with support for calendar effects, holidays, and promotions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm


class ForecastTrainer:
    """
    Enhanced trainer for forecast adjustment with support for calendar effects, 
    holidays, and promotions.
    """
    
    def __init__(self, 
                agent,
                environment, 
                output_dir: str = "output",
                num_episodes: int = 500,
                max_steps: int = 14,
                batch_size: int = 32,
                save_every: int = 50,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced trainer.
        
        Args:
            agent: Enhanced linear agent
            environment: Enhanced forecast environment
            output_dir: Directory for outputs
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            batch_size: Batch size for updates
            save_every: How often to save the model
            optimize_for: Which metric to optimize for ("mape", "bias", or "both")
            logger: Logger instance
        """
        self.agent = agent
        self.env = environment
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.save_every = save_every
        self.optimize_for = optimize_for
        
        # Set up logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ForecastTrainer")
        else:
            self.logger = logger
            
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.model_dir = os.path.join(output_dir, "models")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Metrics
        self.scores = []
        self.mape_improvements = []
        self.bias_improvements = []
        self.training_time = 0
        
        # Context-specific metrics
        self.holiday_metrics = {
            'scores': [],
            'mape_improvements': [],
            'bias_improvements': []
        }
        self.promo_metrics = {
            'scores': [],
            'mape_improvements': [],
            'bias_improvements': []
        }
        self.weekend_metrics = {
            'scores': [],
            'mape_improvements': [],
            'bias_improvements': []
        }
        self.weekday_metrics = {
            'scores': [],
            'mape_improvements': [],
            'bias_improvements': []
        }
    
    def _extract_calendar_features(self, state: np.ndarray, feature_dims: Tuple) -> Dict:
        """
        Extract calendar features from state representation.
        
        Args:
            state: State vector
            feature_dims: Feature dimensions from environment
            
        Returns:
            Dictionary of calendar features
        """
        forecast_dim, error_dim, calendar_dim, holiday_dim, promo_dim, horizon_dim, _ = feature_dims
        
        # Calculate starting index for calendar features
        calendar_start = forecast_dim + error_dim
        
        # Check if we have calendar features
        if len(state) <= calendar_start:
            return {'is_weekend': False}
        
        # Extract is_weekend from the first day's features
        # Calendar features are structured as [day_of_week, day_of_month, is_weekend] for each day
        is_weekend = bool(state[calendar_start + 2] > 0.5)  # Third feature is is_weekend
        
        return {'is_weekend': is_weekend}
        
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the agent for forecast adjustment.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Starting training for {self.num_episodes} episodes")
        start_time = time.time()
        
        # Track best metrics for model saving
        best_score = float('-inf')
        best_mape_improvement = 0.0
        best_bias_improvement = 0.0
        
        # Training loop
        for episode in tqdm(range(1, self.num_episodes + 1), disable=not verbose):
            state = self.env.reset()
            episode_score = 0
            episode_td_errors = []
            
            # Metrics for this episode
            sku_original_mape = {sku: 0 for sku in self.env.skus}
            sku_adjusted_mape = {sku: 0 for sku in self.env.skus}
            sku_original_bias = {sku: 0 for sku in self.env.skus}
            sku_adjusted_bias = {sku: 0 for sku in self.env.skus}
            
            # Context-specific metrics for this episode
            holiday_scores = []
            promo_scores = []
            weekend_scores = []
            weekday_scores = []
            
            holiday_mape_imps = []
            promo_mape_imps = []
            weekend_mape_imps = []
            weekday_mape_imps = []
            
            holiday_bias_imps = []
            promo_bias_imps = []
            weekend_bias_imps = []
            weekday_bias_imps = []
            
            for step in range(self.max_steps):
                adjustments = {}
                
                # Determine adjustments for all SKUs
                for i, sku in enumerate(self.env.skus):
                    # Get feature dimensions
                    feature_dims = self.env.get_feature_dims()
                    forecast_dim = feature_dims[0]
                    
                    # State features for this SKU
                    sku_state = state[i]
                    
                    # Extract calendar information for context-based tracking
                    calendar_features = self._extract_calendar_features(sku_state, feature_dims)
                    is_weekend = calendar_features.get('is_weekend', False)
                    
                    # Get action from agent with context
                    context = {
                        'is_holiday': False,  # This will be updated after the environment step
                        'is_promotion': False,  # This will be updated after the environment step
                        'is_weekend': is_weekend
                    }
                    action_idx = self.agent.act(sku_state, context=context)
                    
                    # Extract forecast from state
                    forecasts = sku_state[:forecast_dim]
                    current_forecast = forecasts[0]  # Current day's forecast
                    
                    # Calculate adjusted forecast based on action
                    adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast)
                    
                    adjustments[sku] = (action_idx, adjusted_forecast)
                
                # Take step in environment
                next_state, rewards, done, info = self.env.step(adjustments)
                
                # Update episode metrics
                episode_score += sum(rewards.values())
                
                # Update per-SKU metrics
                for sku in self.env.skus:
                    sku_original_mape[sku] = info['original_mape'][sku]
                    sku_adjusted_mape[sku] = info['adjusted_mape'][sku]
                    sku_original_bias[sku] = info['original_bias'][sku]
                    sku_adjusted_bias[sku] = info['adjusted_bias'][sku]
                    
                    # Calculate improvements
                    mape_imp = info['original_mape'][sku] - info['adjusted_mape'][sku]
                    bias_imp = abs(info['original_bias'][sku]) - abs(info['adjusted_bias'][sku])
                    
                    # Update context-specific metrics
                    is_holiday = info['is_holiday'].get(sku, False)
                    is_promo = info['is_promotion'].get(sku, False)
                    
                    # Extract day of week from calendar features
                    if (self.env.start_date is not None and hasattr(self.env, 'current_step')):
                        current_date = self.env.start_date + pd.Timedelta(days=self.env.current_step-1)
                        is_weekend = current_date.weekday() >= 5  # 5 = Saturday, 6 = Sunday
                    else:
                        is_weekend = False
                    
                    # Track metrics by context
                    if is_holiday:
                        holiday_scores.append(rewards[sku])
                        holiday_mape_imps.append(mape_imp)
                        holiday_bias_imps.append(bias_imp)
                    
                    if is_promo:
                        promo_scores.append(rewards[sku])
                        promo_mape_imps.append(mape_imp)
                        promo_bias_imps.append(bias_imp)
                    
                    if is_weekend:
                        weekend_scores.append(rewards[sku])
                        weekend_mape_imps.append(mape_imp)
                        weekend_bias_imps.append(bias_imp)
                    else:
                        weekday_scores.append(rewards[sku])
                        weekday_mape_imps.append(mape_imp)
                        weekday_bias_imps.append(bias_imp)
                
                # Update agent for each SKU
                for i, sku in enumerate(self.env.skus):
                    sku_state = state[i]
                    next_sku_state = next_state[i]
                    action_idx, _ = adjustments[sku]
                    
                    # Update agent
                    td_error = self.agent.update(sku_state, action_idx, rewards[sku], next_sku_state, done)
                    episode_td_errors.append(td_error)
                
                # Batch update
                self.agent.batch_update(self.batch_size)
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate MAPE and bias improvements for this episode
            avg_original_mape = np.mean(list(sku_original_mape.values()))
            avg_adjusted_mape = np.mean(list(sku_adjusted_mape.values()))
            avg_original_bias = np.mean([abs(b) for b in sku_original_bias.values()])
            avg_adjusted_bias = np.mean([abs(b) for b in sku_adjusted_bias.values()])
            
            mape_improvement = (avg_original_mape - avg_adjusted_mape) / avg_original_mape if avg_original_mape > 0 else 0
            bias_improvement = (avg_original_bias - avg_adjusted_bias) / avg_original_bias if avg_original_bias > 0 else 0
            
            # Store metrics
            self.scores.append(episode_score)
            self.mape_improvements.append(mape_improvement)
            self.bias_improvements.append(bias_improvement)
            
            # Store context-specific metrics
            # Holiday metrics
            if holiday_scores:
                self.holiday_metrics['scores'].append(np.mean(holiday_scores))
                self.holiday_metrics['mape_improvements'].append(np.mean(holiday_mape_imps))
                self.holiday_metrics['bias_improvements'].append(np.mean(holiday_bias_imps))
            else:
                self.holiday_metrics['scores'].append(0.0)
                self.holiday_metrics['mape_improvements'].append(0.0)
                self.holiday_metrics['bias_improvements'].append(0.0)
            
            # Promotion metrics
            if promo_scores:
                self.promo_metrics['scores'].append(np.mean(promo_scores))
                self.promo_metrics['mape_improvements'].append(np.mean(promo_mape_imps))
                self.promo_metrics['bias_improvements'].append(np.mean(promo_bias_imps))
            else:
                self.promo_metrics['scores'].append(0.0)
                self.promo_metrics['mape_improvements'].append(0.0)
                self.promo_metrics['bias_improvements'].append(0.0)
            
            # Weekend metrics
            if weekend_scores:
                self.weekend_metrics['scores'].append(np.mean(weekend_scores))
                self.weekend_metrics['mape_improvements'].append(np.mean(weekend_mape_imps))
                self.weekend_metrics['bias_improvements'].append(np.mean(weekend_bias_imps))
            else:
                self.weekend_metrics['scores'].append(0.0)
                self.weekend_metrics['mape_improvements'].append(0.0)
                self.weekend_metrics['bias_improvements'].append(0.0)
            
            # Weekday metrics
            if weekday_scores:
                self.weekday_metrics['scores'].append(np.mean(weekday_scores))
                self.weekday_metrics['mape_improvements'].append(np.mean(weekday_mape_imps))
                self.weekday_metrics['bias_improvements'].append(np.mean(weekday_bias_imps))
            else:
                self.weekday_metrics['scores'].append(0.0)
                self.weekday_metrics['mape_improvements'].append(0.0)
                self.weekday_metrics['bias_improvements'].append(0.0)
            
            # Log progress
            if verbose and (episode % 10 == 0 or episode == 1):
                avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
                avg_mape_imp = np.mean(self.mape_improvements[-100:]) if len(self.mape_improvements) >= 100 else np.mean(self.mape_improvements)
                avg_bias_imp = np.mean(self.bias_improvements[-100:]) if len(self.bias_improvements) >= 100 else np.mean(self.bias_improvements)
                avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
                
                self.logger.info(f"Episode {episode}/{self.num_episodes} | "
                              f"Score: {episode_score:.2f} | "
                              f"Avg Score: {avg_score:.2f} | "
                              f"MAPE Imp: {mape_improvement:.4f} | "
                              f"Bias Imp: {bias_improvement:.4f} | "
                              f"TD Error: {avg_td_error:.4f}")
            
            # Save models periodically
            if episode % self.save_every == 0:
                model_path = os.path.join(self.model_dir, f"model_episode_{episode}.pkl")
                self.agent.save(model_path)
                
                # Plot progress with context-specific metrics
                self._plot_training_progress()
            
            # Save best models
            if episode_score > best_score:
                best_score = episode_score
                best_model_path = os.path.join(self.model_dir, "best_score_model.pkl")
                self.agent.save(best_model_path)
                
            if mape_improvement > best_mape_improvement:
                best_mape_improvement = mape_improvement
                best_mape_model_path = os.path.join(self.model_dir, "best_mape_model.pkl")
                self.agent.save(best_mape_model_path)
                
            if bias_improvement > best_bias_improvement:
                best_bias_improvement = bias_improvement
                best_bias_model_path = os.path.join(self.model_dir, "best_bias_model.pkl")
                self.agent.save(best_bias_model_path)
        
        # Final save
        final_model_path = os.path.join(self.model_dir, "final_model.pkl")
        self.agent.save(final_model_path)
        
        # Calculate training time
        self.training_time = time.time() - start_time
        self.logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        # Final plot
        self._plot_training_progress()
        
        # Return metrics
        metrics = {
            'scores': self.scores,
            'mape_improvements': self.mape_improvements,
            'bias_improvements': self.bias_improvements,
            'training_time': self.training_time,
            'best_score': best_score,
            'best_mape_improvement': best_mape_improvement,
            'best_bias_improvement': best_bias_improvement,
            'final_model_path': final_model_path,
            'holiday_metrics': self.holiday_metrics,
            'promo_metrics': self.promo_metrics,
            'weekend_metrics': self.weekend_metrics,
            'weekday_metrics': self.weekday_metrics
        }
        
        return metrics
    
    def _plot_training_progress(self):
        """Create enhanced plot of training progress metrics including context-specific performance."""
        plt.figure(figsize=(20, 15))
        
        # Plot overall scores
        plt.subplot(3, 3, 1)
        plt.plot(self.scores)
        plt.title('Overall Training Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Moving average
        if len(self.scores) > 10:
            moving_avg = np.convolve(self.scores, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(self.scores)), moving_avg, 'r-')
        
        # Plot MAPE improvement
        plt.subplot(3, 3, 2)
        plt.plot(self.mape_improvements)
        plt.title('Overall MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.grid(True, alpha=0.3)
        
        # Plot bias improvement
        plt.subplot(3, 3, 3)
        plt.plot(self.bias_improvements)
        plt.title('Overall Bias Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.grid(True, alpha=0.3)
        
        # Plot context-specific MAPE improvements
        plt.subplot(3, 3, 4)
        plt.plot(self.holiday_metrics['mape_improvements'], 'g-', label='Holidays')
        plt.plot(self.promo_metrics['mape_improvements'], 'r-', label='Promotions')
        plt.title('Context-Specific MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot day-of-week MAPE improvements
        plt.subplot(3, 3, 5)
        plt.plot(self.weekend_metrics['mape_improvements'], 'b-', label='Weekends')
        plt.plot(self.weekday_metrics['mape_improvements'], 'k-', label='Weekdays')
        plt.title('Day-of-Week MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot action distribution
        plt.subplot(3, 3, 6)
        action_stats = self.agent.get_action_statistics()
        
        # Overall action distribution
        action_dist = action_stats['overall']['distribution']
        adj_factors = action_stats['overall']['factors']
        
        factor_labels = [f"{f:.1f}x" for f in adj_factors]
        plt.bar(factor_labels, action_dist)
        plt.title('Overall Adjustment Factor Distribution')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        
        # Plot context-specific action distributions
        plt.subplot(3, 3, 7)
        
        # Set up bar positions
        x = np.arange(len(adj_factors))
        width = 0.35
        
        # Plot holiday vs promotion distributions
        if sum(action_stats['holiday']['counts']) > 0:
            plt.bar(x - width/2, action_stats['holiday']['distribution'], width, label='Holidays')
        if sum(action_stats['promotion']['counts']) > 0:
            plt.bar(x + width/2, action_stats['promotion']['distribution'], width, label='Promotions')
        
        plt.title('Holiday vs Promotion Adjustments')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Frequency')
        plt.xticks(x, factor_labels, rotation=45)
        plt.legend()
        
        # Plot weekend vs weekday distributions
        plt.subplot(3, 3, 8)
        
        if sum(action_stats['weekend']['counts']) > 0:
            plt.bar(x - width/2, action_stats['weekend']['distribution'], width, label='Weekends')
        if sum(action_stats['weekday']['counts']) > 0:
            plt.bar(x + width/2, action_stats['weekday']['distribution'], width, label='Weekdays')
        
        plt.title('Weekend vs Weekday Adjustments')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Frequency')
        plt.xticks(x, factor_labels, rotation=45)
        plt.legend()
        
        # Plot comparative MAPE improvements by context
        plt.subplot(3, 3, 9)
        
        # Calculate average improvements for last 100 episodes (or all if less than 100)
        window = min(100, len(self.mape_improvements))
        avg_overall = np.mean(self.mape_improvements[-window:])
        avg_holiday = np.mean(self.holiday_metrics['mape_improvements'][-window:])
        avg_promo = np.mean(self.promo_metrics['mape_improvements'][-window:])
        avg_weekend = np.mean(self.weekend_metrics['mape_improvements'][-window:])
        avg_weekday = np.mean(self.weekday_metrics['mape_improvements'][-window:])
        
        contexts = ['Overall', 'Holidays', 'Promos', 'Weekends', 'Weekdays']
        improvements = [avg_overall, avg_holiday, avg_promo, avg_weekend, avg_weekday]
        
        plt.bar(contexts, improvements)
        plt.title('Avg MAPE Improvement by Context (Last 100 Episodes)')
        plt.ylabel('Improvement Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
    
    def evaluate(self, num_episodes: int = 10, verbose: bool = True) -> Dict:
        """
        Evaluate the forecast adjustment agent with context-specific metrics.
        
        Args:
            num_episodes: Number of episodes to evaluate
            verbose: Whether to print progress
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Starting evaluation for {num_episodes} episodes")
        
        # Evaluation metrics
        eval_scores = []
        eval_mape_improvements = []
        eval_bias_improvements = []
        
        # Context-specific metrics
        holiday_metrics = {
            'scores': [], 'mape_improvements': [], 'bias_improvements': [],
            'sku_actions': {}, 'sku_improvements': {}
        }
        promo_metrics = {
            'scores': [], 'mape_improvements': [], 'bias_improvements': [],
            'sku_actions': {}, 'sku_improvements': {}
        }
        weekend_metrics = {
            'scores': [], 'mape_improvements': [], 'bias_improvements': [],
            'sku_actions': {}, 'sku_improvements': {}
        }
        weekday_metrics = {
            'scores': [], 'mape_improvements': [], 'bias_improvements': [],
            'sku_actions': {}, 'sku_improvements': {}
        }
        
        # Initialize SKU-level metrics tracking
        sku_level_metrics = {}
        for sku in self.env.skus:
            sku_level_metrics[sku] = {
                "original_mape": [],
                "adjusted_mape": [],
                "original_bias": [],
                "adjusted_bias": [],
                "actions": [],
                "is_holiday": [],
                "is_promotion": [],
                "is_weekend": []
            }
        
        # Evaluation loop
        for episode in tqdm(range(1, num_episodes + 1), disable=not verbose):
            state = self.env.reset()
            episode_score = 0
            
            # Episode-specific context metrics
            episode_holiday_metrics = {'scores': [], 'mape_imps': [], 'bias_imps': []}
            episode_promo_metrics = {'scores': [], 'mape_imps': [], 'bias_imps': []}
            episode_weekend_metrics = {'scores': [], 'mape_imps': [], 'bias_imps': []}
            episode_weekday_metrics = {'scores': [], 'mape_imps': [], 'bias_imps': []}
            
            for step in range(self.max_steps):
                adjustments = {}
                
                # Determine adjustments for all SKUs (no exploration)
                for i, sku in enumerate(self.env.skus):
                    # Get feature dimensions
                    feature_dims = self.env.get_feature_dims()
                    forecast_dim = feature_dims[0]
                    
                    # State features for this SKU
                    sku_state = state[i]
                    
                    # Extract calendar information
                    calendar_features = self._extract_calendar_features(sku_state, feature_dims)
                    is_weekend = calendar_features.get('is_weekend', False)
                    
                    # Get action from agent with context (no exploration)
                    context = {
                        'is_holiday': False,  # Will be updated after environment step
                        'is_promotion': False,  # Will be updated after environment step
                        'is_weekend': is_weekend
                    }
                    action_idx = self.agent.act(sku_state, explore=False, context=context)
                    
                    # Extract forecast from state
                    forecasts = sku_state[:forecast_dim]
                    current_forecast = forecasts[0]  # Current day's forecast
                    
                    # Calculate adjusted forecast based on action
                    adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast)
                    
                    adjustments[sku] = (action_idx, adjusted_forecast)
                    
                    # Track actions for SKU
                    sku_level_metrics[sku]["actions"].append(action_idx)
                    sku_level_metrics[sku]["is_weekend"].append(is_weekend)
                
                # Take step in environment
                next_state, rewards, done, info = self.env.step(adjustments)
                
                # Update episode metrics
                episode_score += sum(rewards.values())
                
                # Update SKU-level metrics
                for sku in self.env.skus:
                    # Basic metrics
                    sku_level_metrics[sku]["original_mape"].append(info['original_mape'][sku])
                    sku_level_metrics[sku]["adjusted_mape"].append(info['adjusted_mape'][sku])
                    sku_level_metrics[sku]["original_bias"].append(info['original_bias'][sku])
                    sku_level_metrics[sku]["adjusted_bias"].append(info['adjusted_bias'][sku])
                    
                    # Context flags
                    is_holiday = info['is_holiday'].get(sku, False)
                    is_promo = info['is_promotion'].get(sku, False)
                    
                    sku_level_metrics[sku]["is_holiday"].append(is_holiday)
                    sku_level_metrics[sku]["is_promotion"].append(is_promo)
                    
                    # Calculate improvements
                    mape_imp = info['original_mape'][sku] - info['adjusted_mape'][sku]
                    bias_imp = abs(info['original_bias'][sku]) - abs(info['adjusted_bias'][sku])
                    
                    # Track context-specific metrics
                    if is_holiday:
                        episode_holiday_metrics['scores'].append(rewards[sku])
                        episode_holiday_metrics['mape_imps'].append(mape_imp)
                        episode_holiday_metrics['bias_imps'].append(bias_imp)
                        
                        # Track per-SKU actions for holidays
                        if sku not in holiday_metrics['sku_actions']:
                            holiday_metrics['sku_actions'][sku] = []
                            holiday_metrics['sku_improvements'][sku] = []
                        
                        action_idx, _ = adjustments[sku]
                        holiday_metrics['sku_actions'][sku].append(action_idx)
                        holiday_metrics['sku_improvements'][sku].append((mape_imp, bias_imp))
                    
                    if is_promo:
                        episode_promo_metrics['scores'].append(rewards[sku])
                        episode_promo_metrics['mape_imps'].append(mape_imp)
                        episode_promo_metrics['bias_imps'].append(bias_imp)
                        
                        # Track per-SKU actions for promotions
                        if sku not in promo_metrics['sku_actions']:
                            promo_metrics['sku_actions'][sku] = []
                            promo_metrics['sku_improvements'][sku] = []
                        
                        action_idx, _ = adjustments[sku]
                        promo_metrics['sku_actions'][sku].append(action_idx)
                        promo_metrics['sku_improvements'][sku].append((mape_imp, bias_imp))
                    
                    # Extract weekend status
                    if (self.env.start_date is not None and hasattr(self.env, 'current_step')):
                        current_date = self.env.start_date + pd.Timedelta(days=self.env.current_step-1)
                        is_weekend = current_date.weekday() >= 5
                    else:
                        is_weekend = False
                    
                    if is_weekend:
                        episode_weekend_metrics['scores'].append(rewards[sku])
                        episode_weekend_metrics['mape_imps'].append(mape_imp)
                        episode_weekend_metrics['bias_imps'].append(bias_imp)
                        
                        # Track per-SKU actions for weekends
                        if sku not in weekend_metrics['sku_actions']:
                            weekend_metrics['sku_actions'][sku] = []
                            weekend_metrics['sku_improvements'][sku] = []
                        
                        action_idx, _ = adjustments[sku]
                        weekend_metrics['sku_actions'][sku].append(action_idx)
                        weekend_metrics['sku_improvements'][sku].append((mape_imp, bias_imp))
                    else:
                        episode_weekday_metrics['scores'].append(rewards[sku])
                        episode_weekday_metrics['mape_imps'].append(mape_imp)
                        episode_weekday_metrics['bias_imps'].append(bias_imp)
                        
                        # Track per-SKU actions for weekdays
                        if sku not in weekday_metrics['sku_actions']:
                            weekday_metrics['sku_actions'][sku] = []
                            weekday_metrics['sku_improvements'][sku] = []
                        
                        action_idx, _ = adjustments[sku]
                        weekday_metrics['sku_actions'][sku].append(action_idx)
                        weekday_metrics['sku_improvements'][sku].append((mape_imp, bias_imp))
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate episode improvements
            original_mape = np.mean([info['original_mape'][sku] for sku in info['original_mape']])
            adjusted_mape = np.mean([info['adjusted_mape'][sku] for sku in info['adjusted_mape']])
            original_bias = np.mean([abs(info['original_bias'][sku]) for sku in info['original_bias']])
            adjusted_bias = np.mean([abs(info['adjusted_bias'][sku]) for sku in info['adjusted_bias']])
            
            mape_improvement = (original_mape - adjusted_mape) / original_mape if original_mape > 0 else 0
            bias_improvement = (original_bias - adjusted_bias) / original_bias if original_bias > 0 else 0
            
            # Store metrics
            eval_scores.append(episode_score)
            eval_mape_improvements.append(mape_improvement)
            eval_bias_improvements.append(bias_improvement)
            
            # Store context-specific metrics for this episode
            if episode_holiday_metrics['scores']:
                holiday_metrics['scores'].append(np.mean(episode_holiday_metrics['scores']))
                holiday_metrics['mape_improvements'].append(np.mean(episode_holiday_metrics['mape_imps']))
                holiday_metrics['bias_improvements'].append(np.mean(episode_holiday_metrics['bias_imps']))
            
            if episode_promo_metrics['scores']:
                promo_metrics['scores'].append(np.mean(episode_promo_metrics['scores']))
                promo_metrics['mape_improvements'].append(np.mean(episode_promo_metrics['mape_imps']))
                promo_metrics['bias_improvements'].append(np.mean(episode_promo_metrics['bias_imps']))
            
            if episode_weekend_metrics['scores']:
                weekend_metrics['scores'].append(np.mean(episode_weekend_metrics['scores']))
                weekend_metrics['mape_improvements'].append(np.mean(episode_weekend_metrics['mape_imps']))
                weekend_metrics['bias_improvements'].append(np.mean(episode_weekend_metrics['bias_imps']))
            
            if episode_weekday_metrics['scores']:
                weekday_metrics['scores'].append(np.mean(episode_weekday_metrics['scores']))
                weekday_metrics['mape_improvements'].append(np.mean(episode_weekday_metrics['mape_imps']))
                weekday_metrics['bias_improvements'].append(np.mean(episode_weekday_metrics['bias_imps']))
            
            if verbose:
                self.logger.info(f"Eval Episode {episode}/{num_episodes} | "
                              f"Score: {episode_score:.2f} | "
                              f"MAPE Imp: {mape_improvement:.4f} | "
                              f"Bias Imp: {bias_improvement:.4f}")
        
        # Calculate aggregate metrics
        avg_score = np.mean(eval_scores)
        avg_mape_improvement = np.mean(eval_mape_improvements)
        avg_bias_improvement = np.mean(eval_bias_improvements)
        
        # Calculate context-specific average metrics
        context_metrics = {
            'holiday': {
                'avg_score': np.mean(holiday_metrics['scores']) if holiday_metrics['scores'] else 0,
                'avg_mape_improvement': np.mean(holiday_metrics['mape_improvements']) if holiday_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(holiday_metrics['bias_improvements']) if holiday_metrics['bias_improvements'] else 0
            },
            'promotion': {
                'avg_score': np.mean(promo_metrics['scores']) if promo_metrics['scores'] else 0,
                'avg_mape_improvement': np.mean(promo_metrics['mape_improvements']) if promo_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(promo_metrics['bias_improvements']) if promo_metrics['bias_improvements'] else 0
            },
            'weekend': {
                'avg_score': np.mean(weekend_metrics['scores']) if weekend_metrics['scores'] else 0,
                'avg_mape_improvement': np.mean(weekend_metrics['mape_improvements']) if weekend_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(weekend_metrics['bias_improvements']) if weekend_metrics['bias_improvements'] else 0
            },
            'weekday': {
                'avg_score': np.mean(weekday_metrics['scores']) if weekday_metrics['scores'] else 0,
                'avg_mape_improvement': np.mean(weekday_metrics['mape_improvements']) if weekday_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(weekday_metrics['bias_improvements']) if weekday_metrics['bias_improvements'] else 0
            }
        }
        
        # Calculate SKU-level summary
        sku_summary = {}
        for sku in self.env.skus:
            # Filter metrics by context
            holiday_indices = [i for i, is_holiday in enumerate(sku_level_metrics[sku]["is_holiday"]) if is_holiday]
            promo_indices = [i for i, is_promo in enumerate(sku_level_metrics[sku]["is_promotion"]) if is_promo]
            weekend_indices = [i for i, is_weekend in enumerate(sku_level_metrics[sku]["is_weekend"]) if is_weekend]
            weekday_indices = [i for i, is_weekend in enumerate(sku_level_metrics[sku]["is_weekend"]) if not is_weekend]
            
            # Overall metrics
            orig_mape = np.mean(sku_level_metrics[sku]["original_mape"])
            adj_mape = np.mean(sku_level_metrics[sku]["adjusted_mape"])
            orig_bias = np.mean([abs(b) for b in sku_level_metrics[sku]["original_bias"]])
            adj_bias = np.mean([abs(b) for b in sku_level_metrics[sku]["adjusted_bias"]])
            
            mape_imp = (orig_mape - adj_mape) / orig_mape if orig_mape > 0 else 0
            bias_imp = (orig_bias - adj_bias) / orig_bias if orig_bias > 0 else 0
            
            # Most common action
            actions = sku_level_metrics[sku]["actions"]
            most_common_action = np.argmax(np.bincount(actions)) if actions else 0
            
            # Context-specific metrics
            holiday_mape_imp = 0
            holiday_bias_imp = 0
            if holiday_indices:
                h_orig_mape = np.mean([sku_level_metrics[sku]["original_mape"][i] for i in holiday_indices])
                h_adj_mape = np.mean([sku_level_metrics[sku]["adjusted_mape"][i] for i in holiday_indices])
                h_orig_bias = np.mean([abs(sku_level_metrics[sku]["original_bias"][i]) for i in holiday_indices])
                h_adj_bias = np.mean([abs(sku_level_metrics[sku]["adjusted_bias"][i]) for i in holiday_indices])
                
                holiday_mape_imp = (h_orig_mape - h_adj_mape) / h_orig_mape if h_orig_mape > 0 else 0
                holiday_bias_imp = (h_orig_bias - h_adj_bias) / h_orig_bias if h_orig_bias > 0 else 0
            
            promo_mape_imp = 0
            promo_bias_imp = 0
            if promo_indices:
                p_orig_mape = np.mean([sku_level_metrics[sku]["original_mape"][i] for i in promo_indices])
                p_adj_mape = np.mean([sku_level_metrics[sku]["adjusted_mape"][i] for i in promo_indices])
                p_orig_bias = np.mean([abs(sku_level_metrics[sku]["original_bias"][i]) for i in promo_indices])
                p_adj_bias = np.mean([abs(sku_level_metrics[sku]["adjusted_bias"][i]) for i in promo_indices])
                
                promo_mape_imp = (p_orig_mape - p_adj_mape) / p_orig_mape if p_orig_mape > 0 else 0
                promo_bias_imp = (p_orig_bias - p_adj_bias) / p_orig_bias if p_orig_bias > 0 else 0
            
            # Store all metrics for this SKU
            sku_summary[sku] = {
                "overall": {
                    "original_mape": orig_mape,
                    "adjusted_mape": adj_mape,
                    "mape_improvement": mape_imp,
                    "original_bias": orig_bias,
                    "adjusted_bias": adj_bias,
                    "bias_improvement": bias_imp,
                    "common_adjustment": self.agent.adjustment_factors[most_common_action]
                },
                "holiday": {
                    "mape_improvement": holiday_mape_imp,
                    "bias_improvement": holiday_bias_imp,
                    "sample_count": len(holiday_indices)
                },
                "promotion": {
                    "mape_improvement": promo_mape_imp,
                    "bias_improvement": promo_bias_imp,
                    "sample_count": len(promo_indices)
                },
                "weekend_weekday": {
                    "weekend_count": len(weekend_indices),
                    "weekday_count": len(weekday_indices)
                }
            }
        
        # Sort SKUs by overall improvement
        if self.optimize_for == "mape":
            top_skus = sorted(sku_summary.items(), key=lambda x: x[1]["overall"]["mape_improvement"], reverse=True)
        elif self.optimize_for == "bias":
            top_skus = sorted(sku_summary.items(), key=lambda x: x[1]["overall"]["bias_improvement"], reverse=True)
        else:  # "both"
            top_skus = sorted(sku_summary.items(), 
                             key=lambda x: x[1]["overall"]["mape_improvement"] + x[1]["overall"]["bias_improvement"], 
                             reverse=True)
        
        # Create summary visualization
        self._plot_evaluation_summary(context_metrics, sku_summary, top_skus)
        
        self.logger.info(f"Evaluation complete | "
                      f"Avg Score: {avg_score:.2f} | "
                      f"Avg MAPE Imp: {avg_mape_improvement:.4f} | "
                      f"Avg Bias Imp: {avg_bias_improvement:.4f}")
        
        # Log context-specific metrics
        self.logger.info(f"Context-specific performance:")
        self.logger.info(f"  Holidays: MAPE Imp = {context_metrics['holiday']['avg_mape_improvement']:.4f}, "
                      f"Bias Imp = {context_metrics['holiday']['avg_bias_improvement']:.4f}")
        self.logger.info(f"  Promotions: MAPE Imp = {context_metrics['promotion']['avg_mape_improvement']:.4f}, "
                      f"Bias Imp = {context_metrics['promotion']['avg_bias_improvement']:.4f}")
        self.logger.info(f"  Weekends: MAPE Imp = {context_metrics['weekend']['avg_mape_improvement']:.4f}, "
                      f"Bias Imp = {context_metrics['weekend']['avg_bias_improvement']:.4f}")
        self.logger.info(f"  Weekdays: MAPE Imp = {context_metrics['weekday']['avg_mape_improvement']:.4f}, "
                      f"Bias Imp = {context_metrics['weekday']['avg_bias_improvement']:.4f}")
        
        # Return metrics
        metrics = {
            'scores': eval_scores,
            'mape_improvements': eval_mape_improvements,
            'bias_improvements': eval_bias_improvements,
            'avg_score': avg_score,
            'avg_mape_improvement': avg_mape_improvement,
            'avg_bias_improvement': avg_bias_improvement,
            'context_metrics': context_metrics,
            'sku_metrics': sku_summary,
            'top_skus': top_skus,
            'holiday_metrics': holiday_metrics,
            'promo_metrics': promo_metrics,
            'weekend_metrics': weekend_metrics,
            'weekday_metrics': weekday_metrics
        }
        
        return metrics
    
    def _plot_evaluation_summary(self, context_metrics, sku_summary, top_skus):
        """
        Create enhanced evaluation summary visualization with context-specific metrics.
        
        Args:
            context_metrics: Dictionary of context-specific metrics
            sku_summary: Dictionary of SKU-level metrics
            top_skus: List of top-performing SKUs
        """
        plt.figure(figsize=(20, 16))
        
        # Plot 1: MAPE improvement by context
        plt.subplot(3, 2, 1)
        contexts = ['Overall', 'Holidays', 'Promos', 'Weekends', 'Weekdays']
        mape_imps = [
            self.avg_mape_improvement if hasattr(self, 'avg_mape_improvement') else context_metrics['holiday']['avg_mape_improvement'],
            context_metrics['holiday']['avg_mape_improvement'],
            context_metrics['promotion']['avg_mape_improvement'],
            context_metrics['weekend']['avg_mape_improvement'],
            context_metrics['weekday']['avg_mape_improvement']
        ]
        
        plt.bar(contexts, mape_imps, color=['blue', 'green', 'red', 'purple', 'orange'])
        plt.title('MAPE Improvement by Context')
        plt.ylabel('Improvement Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: MAPE improvement by SKU (top 10)
        plt.subplot(3, 2, 2)
        if len(top_skus) > 0:
            top_10_skus = [sku for sku, _ in top_skus[:min(10, len(top_skus))]]
            mape_imps = [sku_summary[sku]["overall"]["mape_improvement"] for sku in top_10_skus]
            plt.bar(range(len(top_10_skus)), mape_imps)
            plt.xticks(range(len(top_10_skus)), top_10_skus, rotation=90)
            plt.title('MAPE Improvement by SKU (Top 10)')
            plt.ylabel('Improvement Ratio')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No SKU data available", ha='center', va='center')
            plt.title('MAPE Improvement by SKU')
        
        # Plot 3: Distribution of adjustment factors
        plt.subplot(3, 2, 3)
        action_stats = self.agent.get_action_statistics()
        
        # Overall action distribution
        action_dist = action_stats['overall']['distribution']
        adj_factors = action_stats['overall']['factors']
        
        factor_labels = [f"{f:.1f}x" for f in adj_factors]
        plt.bar(factor_labels, action_dist)
        plt.title('Overall Adjustment Factor Distribution')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Context-specific adjustment factor distributions
        plt.subplot(3, 2, 4)
        
        # Set up bar positions
        x = np.arange(len(adj_factors))
        width = 0.35
        
        # Plot holiday vs promotion distributions
        if sum(action_stats['holiday']['counts']) > 0:
            plt.bar(x - width/2, action_stats['holiday']['distribution'], width, label='Holidays')
        if sum(action_stats['promotion']['counts']) > 0:
            plt.bar(x + width/2, action_stats['promotion']['distribution'], width, label='Promotions')
        
        plt.title('Holiday vs Promotion Adjustments')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Frequency')
        plt.xticks(x, factor_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Top 5 SKU profiles with context-specific improvements
        plt.subplot(3, 2, 5)
        if len(top_skus) >= 5:
            top_5_skus = [sku for sku, _ in top_skus[:5]]
            
            # Get overall, holiday, and promo improvements for top 5 SKUs
            overall_imps = [sku_summary[sku]["overall"]["mape_improvement"] for sku in top_5_skus]
            holiday_imps = [sku_summary[sku]["holiday"]["mape_improvement"] for sku in top_5_skus]
            promo_imps = [sku_summary[sku]["promotion"]["mape_improvement"] for sku in top_5_skus]
            
            x = np.arange(len(top_5_skus))
            width = 0.25
            
            plt.bar(x - width, overall_imps, width, label='Overall')
            plt.bar(x, holiday_imps, width, label='Holidays')
            plt.bar(x + width, promo_imps, width, label='Promotions')
            
            plt.title('Top 5 SKU MAPE Improvements by Context')
            plt.xlabel('SKU')
            plt.ylabel('Improvement Ratio')
            plt.xticks(x, top_5_skus, rotation=90)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Insufficient SKU data", ha='center', va='center')
            plt.title('Top SKU Improvements by Context')
        
        # Plot 6: Comparative Bias improvements by context
        plt.subplot(3, 2, 6)
        bias_imps = [
            context_metrics['holiday']['avg_bias_improvement'],
            context_metrics['promotion']['avg_bias_improvement'],
            context_metrics['weekend']['avg_bias_improvement'],
            context_metrics['weekday']['avg_bias_improvement']
        ]
        bias_contexts = ['Holidays', 'Promos', 'Weekends', 'Weekdays']
        
        plt.bar(bias_contexts, bias_imps, color=['green', 'red', 'purple', 'orange'])
        plt.title('Bias Improvement by Context')
        plt.ylabel('Improvement Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'evaluation_summary.png'))
        plt.close()
    
    def generate_adjusted_forecasts(self, num_days: int = 14) -> pd.DataFrame:
        """
        Generate adjusted forecasts using the trained agent, with context-specific information.
        
        Args:
            num_days: Number of days to forecast
            
        Returns:
            DataFrame of adjusted forecasts with context information
        """
        self.logger.info(f"Generating adjusted forecasts for {num_days} days with context information")
        
        # Reset environment
        state = self.env.reset()
        
        # Predictions storage
        forecast_adjustments = []
        
        for day in range(min(num_days, self.max_steps)):
            # Get date for the current day
            if hasattr(self.env, 'start_date'):
                current_date = self.env.start_date + pd.Timedelta(days=day)
                day_of_week = current_date.weekday()
                is_weekend = day_of_week >= 5
            else:
                current_date = None
                day_of_week = None
                is_weekend = False
            
            # Process each SKU
            for i, sku in enumerate(self.env.skus):
                # Extract state components
                feature_dims = self.env.get_feature_dims()
                forecast_dim = feature_dims[0]
                
                # State features for this SKU
                sku_state = state[i]
                
                # Extract calendar information
                calendar_features = self._extract_calendar_features(sku_state, feature_dims)
                is_weekend_from_features = calendar_features.get('is_weekend', False)
                
                # Check if this is a holiday or promotion
                is_holiday = self.env._check_if_holiday(day) if hasattr(self.env, '_check_if_holiday') else False
                is_promotion = self.env._check_if_promotion(sku, day) if hasattr(self.env, '_check_if_promotion') else False
                
                # Get action from agent with context (no exploration)
                context = {
                    'is_holiday': is_holiday,
                    'is_promotion': is_promotion,
                    'is_weekend': is_weekend or is_weekend_from_features
                }
                action_idx = self.agent.act(sku_state, explore=False, context=context)
                
                # Extract forecasts from state
                forecasts = sku_state[:forecast_dim]
                
                # For each forecast day
                for forecast_day in range(forecast_dim):
                    original_forecast = forecasts[forecast_day]
                    
                    # Apply adjustment factor
                    factor = self.agent.adjustment_factors[action_idx]
                    adjusted_forecast = original_forecast * factor
                    
                    # Add to predictions with context information
                    forecast_adjustments.append({
                        'sku_id': sku,
                        'day': day,
                        'date': current_date.strftime('%Y-%m-%d') if current_date else None,
                        'day_of_week': day_of_week,
                        'is_weekend': is_weekend,
                        'is_holiday': is_holiday,
                        'is_promotion': is_promotion,
                        'forecast_day': forecast_day,
                        'original_forecast': float(original_forecast),
                        'adjustment_factor': float(factor),
                        'adjusted_forecast': float(adjusted_forecast),
                        'action_idx': int(action_idx)
                    })
            
            # Calculate adjustments for environment step
            adjustments = {}
            for i, sku in enumerate(self.env.skus):
                # State features
                sku_state = state[i]
                
                # Extract context for this SKU and day
                is_holiday = self.env._check_if_holiday(day) if hasattr(self.env, '_check_if_holiday') else False
                is_promotion = self.env._check_if_promotion(sku, day) if hasattr(self.env, '_check_if_promotion') else False
                
                # Get action with context
                context = {
                    'is_holiday': is_holiday,
                    'is_promotion': is_promotion,
                    'is_weekend': is_weekend
                }
                action_idx = self.agent.act(sku_state, explore=False, context=context)
                
                # Extract current forecast
                forecast_dim = feature_dims[0]
                current_forecast = sku_state[0]  # First forecast
                
                # Calculate adjusted forecast
                adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast)
                
                adjustments[sku] = (action_idx, adjusted_forecast)
            
            # Take environment step
            next_state, _, done, _ = self.env.step(adjustments)
            state = next_state
            
            if done:
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(forecast_adjustments)
        
        # Log summary statistics
        self.logger.info(f"Generated {len(df)} forecast adjustments")
        self.logger.info(f"Average adjustment factor: {df['adjustment_factor'].mean():.4f}")
        
        # Context-specific summaries
        if 'is_holiday' in df.columns:
            holiday_df = df[df['is_holiday'] == True]
            if len(holiday_df) > 0:
                self.logger.info(f"Holiday adjustments: {len(holiday_df)} rows, avg factor: {holiday_df['adjustment_factor'].mean():.4f}")
        
        if 'is_promotion' in df.columns:
            promo_df = df[df['is_promotion'] == True]
            if len(promo_df) > 0:
                self.logger.info(f"Promotion adjustments: {len(promo_df)} rows, avg factor: {promo_df['adjustment_factor'].mean():.4f}")
        
        if 'is_weekend' in df.columns:
            weekend_df = df[df['is_weekend'] == True]
            weekday_df = df[df['is_weekend'] == False]
            if len(weekend_df) > 0:
                self.logger.info(f"Weekend adjustments: {len(weekend_df)} rows, avg factor: {weekend_df['adjustment_factor'].mean():.4f}")
            if len(weekday_df) > 0:
                self.logger.info(f"Weekday adjustments: {len(weekday_df)} rows, avg factor: {weekday_df['adjustment_factor'].mean():.4f}")
        
        # Save visualizations
        self._visualize_forecast_adjustments(df)
        
        return df
    
    def _visualize_forecast_adjustments(self, adjustments_df: pd.DataFrame):
        """
        Create visualizations for the generated forecast adjustments.
        
        Args:
            adjustments_df: DataFrame of adjustments
        """
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Distribution of adjustment factors
        plt.subplot(3, 2, 1)
        adjustment_counts = adjustments_df['action_idx'].value_counts().sort_index()
        factors = self.agent.adjustment_factors
        
        factor_labels = [f"{f:.1f}x" for f in factors]
        counts = [adjustment_counts.get(i, 0) for i in range(len(factors))]
        plt.bar(factor_labels, counts)
        plt.title('Overall Adjustment Factor Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average adjustment by day
        plt.subplot(3, 2, 2)
        if 'day' in adjustments_df.columns:
            day_factors = adjustments_df.groupby('day')['adjustment_factor'].mean()
            plt.plot(day_factors.index, day_factors.values, 'o-')
            plt.title('Average Adjustment Factor by Day')
            plt.xlabel('Day')
            plt.ylabel('Avg Factor')
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Context-specific adjustments
        plt.subplot(3, 2, 3)
        
        contexts = []
        avg_factors = []
        
        # Add overall average
        contexts.append('Overall')
        avg_factors.append(adjustments_df['adjustment_factor'].mean())
        
        # Add holiday average if present
        if 'is_holiday' in adjustments_df.columns:
            holiday_df = adjustments_df[adjustments_df['is_holiday'] == True]
            if len(holiday_df) > 0:
                contexts.append('Holiday')
                avg_factors.append(holiday_df['adjustment_factor'].mean())
        
        # Add promotion average if present
        if 'is_promotion' in adjustments_df.columns:
            promo_df = adjustments_df[adjustments_df['is_promotion'] == True]
            if len(promo_df) > 0:
                contexts.append('Promo')
                avg_factors.append(promo_df['adjustment_factor'].mean())
        
        # Add weekend/weekday averages if present
        if 'is_weekend' in adjustments_df.columns:
            weekend_df = adjustments_df[adjustments_df['is_weekend'] == True]
            weekday_df = adjustments_df[adjustments_df['is_weekend'] == False]
            
            if len(weekend_df) > 0:
                contexts.append('Weekend')
                avg_factors.append(weekend_df['adjustment_factor'].mean())
            
            if len(weekday_df) > 0:
                contexts.append('Weekday')
                avg_factors.append(weekday_df['adjustment_factor'].mean())
        
        plt.bar(contexts, avg_factors, color=['blue', 'green', 'red', 'purple', 'orange'][:len(contexts)])
        plt.title('Average Adjustment Factor by Context')
        plt.ylabel('Avg Factor')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Day of week adjustments
        plt.subplot(3, 2, 4)
        if 'day_of_week' in adjustments_df.columns:
            dow_mapping = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            adjustments_df['day_name'] = adjustments_df['day_of_week'].map(dow_mapping)
            
            dow_factors = adjustments_df.groupby('day_name')['adjustment_factor'].mean()
            
            # Sort by day of week
            dow_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_factors = dow_factors.reindex(dow_order)
            
            plt.bar(dow_factors.index, dow_factors.values)
            plt.title('Average Adjustment Factor by Day of Week')
            plt.ylabel('Avg Factor')
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Original vs Adjusted Forecast scatter
        plt.subplot(3, 2, 5)
        plt.scatter(adjustments_df['original_forecast'], 
                   adjustments_df['adjusted_forecast'], 
                   alpha=0.3)
        max_val = max(adjustments_df['original_forecast'].max(), 
                      adjustments_df['adjusted_forecast'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')  # Diagonal line
        plt.title('Original vs Adjusted Forecast')
        plt.xlabel('Original Forecast')
        plt.ylabel('Adjusted Forecast')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Adjustment percentage distribution
        plt.subplot(3, 2, 6)
        # Calculate percentage adjustments
        pct_changes = ((adjustments_df['adjusted_forecast'] - adjustments_df['original_forecast']) 
                      / adjustments_df['original_forecast'].clip(lower=1e-8)) * 100
        # Remove extreme values for better visualization
        pct_changes = pct_changes.clip(lower=-50, upper=50)
        plt.hist(pct_changes, bins=20)
        plt.title('Adjustment Percentage Distribution')
        plt.xlabel('Adjustment (%)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "forecast_adjustments_summary.png"))
        plt.close()
                                            
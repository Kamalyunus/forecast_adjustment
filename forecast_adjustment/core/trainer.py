"""
Trainer module for forecast adjustment using reinforcement learning.
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
    Trainer for the forecast adjustment system with support for:
    
    - Curriculum learning (phased training with different exploration rates)
    - Comprehensive metrics tracking
    - Context-specific performance analysis (holidays, promotions, weekends)
    - Band-specific performance analysis (A-E bands)
    - Pattern-specific learning and metrics
    """
    
    def __init__(self, 
                agent,
                environment, 
                output_dir: str = "output",
                num_episodes: int = 500,
                max_steps: int = 14,
                batch_size: int = 64,
                save_every: int = 25,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                training_phases: Optional[List[Dict]] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize trainer with curriculum learning support.
        
        Args:
            agent: Q-learning agent with context-specific learning
            environment: Forecast environment with band support
            output_dir: Directory for outputs
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            batch_size: Batch size for updates
            save_every: How often to save the model
            optimize_for: Which metric to optimize for ("mape", "bias", or "both")
            training_phases: Optional list of training phase configurations
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
        
        # Set up curriculum learning phases
        if training_phases is None:
            # Default training phases if none provided
            self.training_phases = [
                {
                    'episodes': int(num_episodes * 0.2),
                    'epsilon': 0.9,
                    'learning_rate': 0.01,
                    'batch_size': 32,
                    'description': 'High exploration phase'
                },
                {
                    'episodes': int(num_episodes * 0.4),
                    'epsilon': 0.5,
                    'learning_rate': 0.005,
                    'batch_size': 64,
                    'description': 'Pattern-specific learning phase'
                },
                {
                    'episodes': int(num_episodes * 0.4),
                    'epsilon': 0.1,
                    'learning_rate': 0.001,
                    'batch_size': 128,
                    'description': 'Fine-tuning phase'
                }
            ]
        else:
            self.training_phases = training_phases
        
        # Verify that phases sum to total episodes
        total_phase_episodes = sum(phase['episodes'] for phase in self.training_phases)
        if total_phase_episodes != num_episodes:
            if logger:
                logger.warning(f"Training phases sum to {total_phase_episodes} episodes, but num_episodes is {num_episodes}. Adjusting phases.")
            # Adjust the last phase to make up the difference
            self.training_phases[-1]['episodes'] += (num_episodes - total_phase_episodes)
        
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
        
        # Band-specific metrics
        self.band_metrics = {
            'A': {'scores': [], 'mape_improvements': [], 'bias_improvements': []},
            'B': {'scores': [], 'mape_improvements': [], 'bias_improvements': []},
            'C': {'scores': [], 'mape_improvements': [], 'bias_improvements': []},
            'D': {'scores': [], 'mape_improvements': [], 'bias_improvements': []},
            'E': {'scores': [], 'mape_improvements': [], 'bias_improvements': []}
        }
        
        # Pattern-specific metrics
        self.pattern_metrics = {}
        
        # Track overall best metrics
        self.best_score = float('-inf')
        self.best_mape_improvement = 0.0
        self.best_bias_improvement = 0.0
        self.best_epoch = 0
    
    def _extract_context_features(self, state: np.ndarray, feature_dims: Tuple) -> Dict:
        """
        Extract calendar and band features from state representation.
        
        Args:
            state: State vector
            feature_dims: Feature dimensions from environment
            
        Returns:
            Dictionary of context features
        """
        forecast_dim, error_dim, calendar_dim, holiday_dim, promo_dim, horizon_dim, band_dim, _ = feature_dims
        
        # Calculate starting indices for different feature groups
        calendar_start = forecast_dim + error_dim
        band_start = calendar_start + calendar_dim + holiday_dim + promo_dim + horizon_dim
        
        # Extract calendar features (is_weekend from the first day's features)
        is_weekend = False
        if len(state) > calendar_start + 2:  # Ensure we have calendar features
            # Calendar features are structured as [day_of_week, day_of_month, is_weekend]
            is_weekend = bool(state[calendar_start + 2] > 0.5)  # Third feature is is_weekend
        
        # Extract SKU band (one-hot encoding)
        band = 'C'  # Default to 'C' if not found
        if len(state) > band_start + 4:  # Ensure we have band features
            band_values = state[band_start:band_start+5]
            if max(band_values) > 0.5:  # If one-hot encoding is present
                band_idx = np.argmax(band_values)
                band = ['A', 'B', 'C', 'D', 'E'][band_idx]
        
        return {
            'is_weekend': is_weekend,
            'sku_band': band
        }
        
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the agent for forecast adjustment using curriculum learning.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Starting training for {self.num_episodes} episodes with {len(self.training_phases)} phases")
        start_time = time.time()
        
        # Initialize phase counters
        phase_idx = 0
        episode_counter = 0
        
        # Get feature dimensions
        feature_dims = self.env.get_feature_dims()
        
        # Training loop with phases
        for phase in self.training_phases:
            phase_episodes = phase['episodes']
            phase_epsilon = phase.get('epsilon', 0.3)
            phase_lr = phase.get('learning_rate', 0.005)
            phase_batch_size = phase.get('batch_size', self.batch_size)
            
            self.logger.info(f"Starting training phase {phase_idx + 1}: {phase.get('description', f'Phase {phase_idx + 1}')}")
            self.logger.info(f"  Episodes: {phase_episodes}, Epsilon: {phase_epsilon}, Learning Rate: {phase_lr}")
            
            # Update agent parameters for this phase
            self.agent.epsilon = phase_epsilon
            self.agent.learning_rate = phase_lr
            
            # Train for this phase's episodes
            for episode in tqdm(range(1, phase_episodes + 1), disable=not verbose):
                global_episode = episode_counter + episode
                
                state = self.env.reset()
                episode_score = 0
                episode_td_errors = []
                
                # Track metrics for this episode
                context_metrics = {
                    'holiday': {'scores': [], 'mape_imps': [], 'bias_imps': []},
                    'promo': {'scores': [], 'mape_imps': [], 'bias_imps': []},
                    'weekend': {'scores': [], 'mape_imps': [], 'bias_imps': []},
                    'weekday': {'scores': [], 'mape_imps': [], 'bias_imps': []}
                }
                
                # Track band-specific metrics
                band_metrics = {
                    'A': {'scores': [], 'mape_imps': [], 'bias_imps': []},
                    'B': {'scores': [], 'mape_imps': [], 'bias_imps': []},
                    'C': {'scores': [], 'mape_imps': [], 'bias_imps': []},
                    'D': {'scores': [], 'mape_imps': [], 'bias_imps': []},
                    'E': {'scores': [], 'mape_imps': [], 'bias_imps': []}
                }
                
                # Track pattern-specific metrics
                pattern_episode_metrics = {}
                
                # Metrics for this episode
                sku_original_mape = {sku: 0 for sku in self.env.skus}
                sku_adjusted_mape = {sku: 0 for sku in self.env.skus}
                sku_original_bias = {sku: 0 for sku in self.env.skus}
                sku_adjusted_bias = {sku: 0 for sku in self.env.skus}
                
                for step in range(self.max_steps):
                    adjustments = {}
                    
                    # Determine adjustments for all SKUs
                    for i, sku in enumerate(self.env.skus):
                        # State features for this SKU
                        sku_state = state[i]
                        
                        # Extract context information
                        context_features = self._extract_context_features(sku_state, feature_dims)
                        is_weekend = context_features.get('is_weekend', False)
                        sku_band = context_features.get('sku_band', 'C')
                        
                        # Get pattern type if available
                        pattern_type = "unknown"
                        if hasattr(self.env, 'has_pattern_types') and self.env.has_pattern_types:
                            if sku in self.env.sku_patterns:
                                pattern_type = self.env.sku_patterns[sku]
                        
                        # Get action from agent with context
                        context = {
                            'is_holiday': False,  # Will be updated after the environment step
                            'is_promotion': False,  # Will be updated after the environment step
                            'is_weekend': is_weekend,
                            'pattern_type': pattern_type,
                            'sku_band': sku_band
                        }
                        
                        # Apply phase-specific exploration strategies
                        explore = True
                        if phase_idx > 0:
                            # Adjust exploration based on SKU band in later phases
                            if sku_band in ['A', 'B'] and phase_idx == 1:
                                # Explore more for high-volume SKUs in second phase
                                self.agent.epsilon = min(phase_epsilon * 1.2, 0.95)
                            elif sku_band in ['D', 'E'] and phase_idx == 1:
                                # Explore more for low-volume SKUs in second phase
                                self.agent.epsilon = min(phase_epsilon * 1.2, 0.95)
                            # Adjust exploration based on pattern in later phases
                            elif pattern_type == "promo_holiday" and phase_idx == 1:
                                # Explore more for promo/holiday patterns in second phase
                                self.agent.epsilon = min(phase_epsilon * 1.5, 0.9)
                            elif pattern_type == "day_pattern" and phase_idx == 1:
                                # Explore more for day patterns in second phase
                                self.agent.epsilon = min(phase_epsilon * 1.3, 0.8)
                            else:
                                self.agent.epsilon = phase_epsilon
                        
                        action_idx = self.agent.act(sku_state, explore=explore, context=context)
                        
                        # Extract forecast from state
                        forecasts = sku_state[:feature_dims[0]]
                        current_forecast = forecasts[0]  # Current day's forecast
                        
                        # Calculate adjusted forecast based on action
                        adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast, context)
                        
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
                        
                        # Extract pattern type and band if available
                        pattern_type = info.get('pattern_type', {}).get(sku, "unknown")
                        sku_band = info.get('sku_band', {}).get(sku, "C")
                        
                        # Track pattern-specific metrics
                        if pattern_type != "unknown":
                            if pattern_type not in pattern_episode_metrics:
                                pattern_episode_metrics[pattern_type] = {
                                    'scores': [], 'mape_imps': [], 'bias_imps': []
                                }
                            pattern_episode_metrics[pattern_type]['scores'].append(rewards[sku])
                            pattern_episode_metrics[pattern_type]['mape_imps'].append(mape_imp)
                            pattern_episode_metrics[pattern_type]['bias_imps'].append(bias_imp)
                        
                        # Track band-specific metrics
                        if sku_band in band_metrics:
                            band_metrics[sku_band]['scores'].append(rewards[sku])
                            band_metrics[sku_band]['mape_imps'].append(mape_imp)
                            band_metrics[sku_band]['bias_imps'].append(bias_imp)
                        
                        # Get day of week
                        if hasattr(self.env, 'current_step') and hasattr(self.env, 'start_date'):
                            current_date = self.env.start_date + pd.Timedelta(days=self.env.current_step-1)
                            is_weekend = current_date.weekday() >= 5  # 5 = Saturday, 6 = Sunday
                        else:
                            is_weekend = False
                        
                        # Track metrics by context
                        if is_holiday:
                            context_metrics['holiday']['scores'].append(rewards[sku])
                            context_metrics['holiday']['mape_imps'].append(mape_imp)
                            context_metrics['holiday']['bias_imps'].append(bias_imp)
                        
                        if is_promo:
                            context_metrics['promo']['scores'].append(rewards[sku])
                            context_metrics['promo']['mape_imps'].append(mape_imp)
                            context_metrics['promo']['bias_imps'].append(bias_imp)
                        
                        if is_weekend:
                            context_metrics['weekend']['scores'].append(rewards[sku])
                            context_metrics['weekend']['mape_imps'].append(mape_imp)
                            context_metrics['weekend']['bias_imps'].append(bias_imp)
                        else:
                            context_metrics['weekday']['scores'].append(rewards[sku])
                            context_metrics['weekday']['mape_imps'].append(mape_imp)
                            context_metrics['weekday']['bias_imps'].append(bias_imp)
                    
                    # Update agent for each SKU
                    for i, sku in enumerate(self.env.skus):
                        sku_state = state[i]
                        next_sku_state = next_state[i]
                        action_idx, _ = adjustments[sku]
                        
                        # Extract context features
                        context_features = self._extract_context_features(sku_state, feature_dims)
                        is_weekend = context_features.get('is_weekend', False)
                        sku_band = context_features.get('sku_band', 'C')
                        
                        # Get context for this SKU
                        is_holiday = info['is_holiday'].get(sku, False)
                        is_promo = info['is_promotion'].get(sku, False)
                        
                        # Get pattern type
                        pattern_type = info.get('pattern_type', {}).get(sku, "unknown")
                        
                        context = {
                            'is_holiday': is_holiday,
                            'is_promotion': is_promo,
                            'is_weekend': is_weekend,
                            'pattern_type': pattern_type,
                            'sku_band': sku_band
                        }
                        
                        # Update agent with context
                        td_error = self.agent.update(sku_state, action_idx, rewards[sku], next_sku_state, done, context)
                        episode_td_errors.append(td_error)
                    
                    # Batch update with phase-specific batch size
                    self.agent.batch_update(phase_batch_size)
                    
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
                if context_metrics['holiday']['scores']:
                    self.holiday_metrics['scores'].append(np.mean(context_metrics['holiday']['scores']))
                    self.holiday_metrics['mape_improvements'].append(np.mean(context_metrics['holiday']['mape_imps']))
                    self.holiday_metrics['bias_improvements'].append(np.mean(context_metrics['holiday']['bias_imps']))
                else:
                    self.holiday_metrics['scores'].append(0.0)
                    self.holiday_metrics['mape_improvements'].append(0.0)
                    self.holiday_metrics['bias_improvements'].append(0.0)
                
                # Promotion metrics
                if context_metrics['promo']['scores']:
                    self.promo_metrics['scores'].append(np.mean(context_metrics['promo']['scores']))
                    self.promo_metrics['mape_improvements'].append(np.mean(context_metrics['promo']['mape_imps']))
                    self.promo_metrics['bias_improvements'].append(np.mean(context_metrics['promo']['bias_imps']))
                else:
                    self.promo_metrics['scores'].append(0.0)
                    self.promo_metrics['mape_improvements'].append(0.0)
                    self.promo_metrics['bias_improvements'].append(0.0)
                
                # Weekend metrics
                if context_metrics['weekend']['scores']:
                    self.weekend_metrics['scores'].append(np.mean(context_metrics['weekend']['scores']))
                    self.weekend_metrics['mape_improvements'].append(np.mean(context_metrics['weekend']['mape_imps']))
                    self.weekend_metrics['bias_improvements'].append(np.mean(context_metrics['weekend']['bias_imps']))
                else:
                    self.weekend_metrics['scores'].append(0.0)
                    self.weekend_metrics['mape_improvements'].append(0.0)
                    self.weekend_metrics['bias_improvements'].append(0.0)
    
                if context_metrics['weekday']['scores']:
                    self.weekday_metrics['scores'].append(np.mean(context_metrics['weekday']['scores']))
                    self.weekday_metrics['mape_improvements'].append(np.mean(context_metrics['weekday']['mape_imps']))
                    self.weekday_metrics['bias_improvements'].append(np.mean(context_metrics['weekday']['bias_imps']))
                else:
                    self.weekday_metrics['scores'].append(0.0)
                    self.weekday_metrics['mape_improvements'].append(0.0)
                    self.weekday_metrics['bias_improvements'].append(0.0)
                
                # Store band-specific metrics
                for band in self.band_metrics:
                    if band in band_metrics and band_metrics[band]['scores']:
                        self.band_metrics[band]['scores'].append(np.mean(band_metrics[band]['scores']))
                        self.band_metrics[band]['mape_improvements'].append(np.mean(band_metrics[band]['mape_imps']))
                        self.band_metrics[band]['bias_improvements'].append(np.mean(band_metrics[band]['bias_imps']))
                    else:
                        self.band_metrics[band]['scores'].append(0.0)
                        self.band_metrics[band]['mape_improvements'].append(0.0)
                        self.band_metrics[band]['bias_improvements'].append(0.0)
                
                # Store pattern-specific metrics
                for pattern_type, metrics in pattern_episode_metrics.items():
                    if pattern_type not in self.pattern_metrics:
                        self.pattern_metrics[pattern_type] = {
                            'scores': [], 'mape_improvements': [], 'bias_improvements': []
                        }
                    
                    if metrics['scores']:
                        self.pattern_metrics[pattern_type]['scores'].append(np.mean(metrics['scores']))
                        self.pattern_metrics[pattern_type]['mape_improvements'].append(np.mean(metrics['mape_imps']))
                        self.pattern_metrics[pattern_type]['bias_improvements'].append(np.mean(metrics['bias_imps']))
                    else:
                        self.pattern_metrics[pattern_type]['scores'].append(0.0)
                        self.pattern_metrics[pattern_type]['mape_improvements'].append(0.0)
                        self.pattern_metrics[pattern_type]['bias_improvements'].append(0.0)
                
                # Log progress
                if verbose and (global_episode % 10 == 0 or global_episode == 1):
                    avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
                    avg_mape_imp = np.mean(self.mape_improvements[-100:]) if len(self.mape_improvements) >= 100 else np.mean(self.mape_improvements)
                    avg_bias_imp = np.mean(self.bias_improvements[-100:]) if len(self.bias_improvements) >= 100 else np.mean(self.bias_improvements)
                    avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
                    
                    self.logger.info(f"Phase {phase_idx+1} - Episode {episode}/{phase_episodes} (Global: {global_episode}/{self.num_episodes}) | "
                                  f"Score: {episode_score:.2f} | "
                                  f"Avg Score: {avg_score:.2f} | "
                                  f"MAPE Imp: {mape_improvement:.4f} | "
                                  f"Bias Imp: {bias_improvement:.4f} | "
                                  f"Epsilon: {self.agent.epsilon:.4f} | "
                                  f"TD Error: {avg_td_error:.4f}")
                
                # Save models periodically
                if global_episode % self.save_every == 0:
                    model_path = os.path.join(self.model_dir, f"model_episode_{global_episode}.pkl")
                    self.agent.save(model_path)
                    
                    # Plot progress with context-specific metrics
                    self._plot_training_progress()
                
                # Save best models
                if episode_score > self.best_score:
                    self.best_score = episode_score
                    self.best_epoch = global_episode
                    best_model_path = os.path.join(self.model_dir, "best_score_model.pkl")
                    self.agent.save(best_model_path)
                    
                if mape_improvement > self.best_mape_improvement:
                    self.best_mape_improvement = mape_improvement
                    best_mape_model_path = os.path.join(self.model_dir, "best_mape_model.pkl")
                    self.agent.save(best_mape_model_path)
                    
                if bias_improvement > self.best_bias_improvement:
                    self.best_bias_improvement = bias_improvement
                    best_bias_model_path = os.path.join(self.model_dir, "best_bias_model.pkl")
                    self.agent.save(best_bias_model_path)
            
            # Update episode counter for next phase
            episode_counter += phase_episodes
            phase_idx += 1
        
        # Final save
        final_model_path = os.path.join(self.model_dir, "final_model.pkl")
        self.agent.save(final_model_path)
        
        # Calculate training time
        self.training_time = time.time() - start_time
        self.logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        # Final plot
        self._plot_training_progress()
        
        # Plot pattern-specific learning curves
        self._plot_pattern_learning()
        
        # Plot band-specific learning curves
        self._plot_band_learning()
        
        # Return metrics
        metrics = {
            'scores': self.scores,
            'mape_improvements': self.mape_improvements,
            'bias_improvements': self.bias_improvements,
            'training_time': self.training_time,
            'best_score': self.best_score,
            'best_mape_improvement': self.best_mape_improvement,
            'best_bias_improvement': self.best_bias_improvement,
            'best_epoch': self.best_epoch,
            'final_model_path': final_model_path,
            'holiday_metrics': self.holiday_metrics,
            'promo_metrics': self.promo_metrics,
            'weekend_metrics': self.weekend_metrics,
            'weekday_metrics': self.weekday_metrics,
            'band_metrics': self.band_metrics,
            'pattern_metrics': self.pattern_metrics
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
        
        # Plot band-specific MAPE improvements
        plt.subplot(3, 3, 4)
        for band in ['A', 'B', 'D', 'E']:  # Skip 'C' for clarity
            plt.plot(self.band_metrics[band]['mape_improvements'], label=f'Band {band}')
        plt.title('Band-Specific MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot context-specific MAPE improvements
        plt.subplot(3, 3, 5)
        plt.plot(self.holiday_metrics['mape_improvements'], 'g-', label='Holidays')
        plt.plot(self.promo_metrics['mape_improvements'], 'r-', label='Promotions')
        plt.plot(self.weekend_metrics['mape_improvements'], 'b-', label='Weekends')
        plt.title('Context-Specific MAPE Improvement')
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
        
        # Plot band-specific action distributions
        plt.subplot(3, 3, 7)
        
        # Set up bar positions
        x = np.arange(len(adj_factors))
        width = 0.4
        
        # Fast-selling vs. slow-selling bands
        if ('bands' in action_stats and 
            'A' in action_stats['bands'] and 
            'E' in action_stats['bands']):
            
            high_band_dist = action_stats['bands']['A']['distribution']
            low_band_dist = action_stats['bands']['E']['distribution']
            
            plt.bar(x - width/2, high_band_dist, width, label='Band A (Fast)')
            plt.bar(x + width/2, low_band_dist, width, label='Band E (Slow)')
            
            plt.title('Band A vs Band E Adjustments')
            plt.xlabel('Adjustment Factor')
            plt.ylabel('Frequency')
            plt.xticks(x, factor_labels, rotation=45)
            plt.legend()
        
        # Plot holiday vs promotion distributions
        plt.subplot(3, 3, 8)
        
        if sum(action_stats['holiday']['counts']) > 0:
            plt.bar(x - width/2, action_stats['holiday']['distribution'], width, label='Holidays')
        if sum(action_stats['promotion']['counts']) > 0:
            plt.bar(x + width/2, action_stats['promotion']['distribution'], width, label='Promotions')
        
        plt.title('Holiday vs Promotion Adjustments')
        plt.xlabel('Adjustment Factor')
        plt.ylabel('Frequency')
        plt.xticks(x, factor_labels, rotation=45)
        plt.legend()
        
        # Plot comparative MAPE improvements by band and context
        plt.subplot(3, 3, 9)
        
        # Calculate average improvements for last 100 episodes (or all if less than 100)
        window = min(100, len(self.mape_improvements))
        avg_overall = np.mean(self.mape_improvements[-window:])
        avg_band_A = np.mean(self.band_metrics['A']['mape_improvements'][-window:])
        avg_band_B = np.mean(self.band_metrics['B']['mape_improvements'][-window:])
        avg_band_D = np.mean(self.band_metrics['D']['mape_improvements'][-window:])
        avg_band_E = np.mean(self.band_metrics['E']['mape_improvements'][-window:])
        avg_holiday = np.mean(self.holiday_metrics['mape_improvements'][-window:])
        avg_promo = np.mean(self.promo_metrics['mape_improvements'][-window:])
        
        categories = ['Overall', 'Band A', 'Band B', 'Band D', 'Band E', 'Holidays', 'Promos']
        improvements = [avg_overall, avg_band_A, avg_band_B, avg_band_D, avg_band_E, avg_holiday, avg_promo]
        
        plt.bar(categories, improvements)
        plt.title(f'Avg MAPE Improvement (Last {window} Episodes)')
        plt.ylabel('Improvement Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
    
    def generate_adjusted_forecasts(self, num_days: int = 14) -> pd.DataFrame:
        """
        Generate adjusted forecasts using the trained agent.
        
        Args:
            num_days: Number of days to forecast
            
        Returns:
            DataFrame of adjusted forecasts with context information
        """
        self.logger.info(f"Generating adjusted forecasts for {num_days} days with context information")
        
        # Reset environment
        state = self.env.reset()
        
        # Get feature dimensions
        feature_dims = self.env.get_feature_dims()
        
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
                # State features for this SKU
                sku_state = state[i]
                
                # Extract context information
                context_features = self._extract_context_features(sku_state, feature_dims)
                is_weekend_from_features = context_features.get('is_weekend', False)
                sku_band = context_features.get('sku_band', 'C')
                
                # Check if this is a holiday or promotion
                is_holiday = self.env._check_if_holiday(day) if hasattr(self.env, '_check_if_holiday') else False
                is_promotion = self.env._check_if_promotion(sku, day) if hasattr(self.env, '_check_if_promotion') else False
                
                # Get pattern type if available
                pattern_type = "unknown"
                if hasattr(self.env, 'has_pattern_types') and self.env.has_pattern_types:
                    if sku in self.env.sku_patterns:
                        pattern_type = self.env.sku_patterns[sku]
                
                # Build complete context
                context = {
                    'is_holiday': is_holiday,
                    'is_promotion': is_promotion,
                    'is_weekend': is_weekend or is_weekend_from_features,
                    'pattern_type': pattern_type,
                    'sku_band': sku_band
                }
                
                # Get action from agent with context (no exploration)
                action_idx = self.agent.act(sku_state, explore=False, context=context)
                
                # Extract forecasts from state
                forecasts = sku_state[:feature_dims[0]]
                
                # For each forecast day
                for forecast_day in range(min(len(forecasts), self.env.forecast_horizon)):
                    original_forecast = forecasts[forecast_day]
                    
                    # Apply adjustment factor with context
                    adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, original_forecast, context)
                    factor = adjusted_forecast / original_forecast if original_forecast > 0 else 1.0
                    
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
                        'action_idx': int(action_idx),
                        'pattern_type': pattern_type,
                        'sku_band': sku_band
                    })
            
            # Calculate adjustments for environment step
            adjustments = {}
            for i, sku in enumerate(self.env.skus):
                # State features
                sku_state = state[i]
                
                # Extract context for this SKU
                context_features = self._extract_context_features(sku_state, feature_dims)
                sku_band = context_features.get('sku_band', 'C')
                is_weekend = context_features.get('is_weekend', False)
                
                is_holiday = self.env._check_if_holiday(day) if hasattr(self.env, '_check_if_holiday') else False
                is_promotion = self.env._check_if_promotion(sku, day) if hasattr(self.env, '_check_if_promotion') else False
                
                # Get pattern type
                pattern_type = "unknown"
                if hasattr(self.env, 'has_pattern_types') and self.env.has_pattern_types:
                    if sku in self.env.sku_patterns:
                        pattern_type = self.env.sku_patterns[sku]
                
                # Get action with context
                context = {
                    'is_holiday': is_holiday,
                    'is_promotion': is_promotion,
                    'is_weekend': is_weekend,
                    'pattern_type': pattern_type,
                    'sku_band': sku_band
                }
                
                action_idx = self.agent.act(sku_state, explore=False, context=context)
                
                # Extract current forecast
                current_forecast = sku_state[0]  # First forecast
                
                # Calculate adjusted forecast
                adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast, context)
                
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
        
        # Band-specific summaries
        if 'sku_band' in df.columns:
            for band in df['sku_band'].unique():
                band_df = df[df['sku_band'] == band]
                if len(band_df) > 0:
                    self.logger.info(f"Band {band} adjustments: {len(band_df)} rows, avg factor: {band_df['adjustment_factor'].mean():.4f}")
        
        # Pattern-specific summaries
        if 'pattern_type' in df.columns:
            for pattern in df['pattern_type'].unique():
                if pattern != "unknown":
                    pattern_df = df[df['pattern_type'] == pattern]
                    if len(pattern_df) > 0:
                        self.logger.info(f"{pattern} adjustments: {len(pattern_df)} rows, avg factor: {pattern_df['adjustment_factor'].mean():.4f}")
        
        # Save visualizations of adjustments
        self._visualize_forecast_adjustments(df)
        
        return df
        
    def _visualize_forecast_adjustments(self, adjustments_df: pd.DataFrame):
        """
        Create visualizations for the generated forecast adjustments.
        
        Args:
            adjustments_df: DataFrame of adjustments
        """
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Distribution of adjustment factors
        plt.subplot(2, 2, 1)
        adjustment_counts = adjustments_df['action_idx'].value_counts().sort_index()
        factors = self.agent.adjustment_factors
        
        factor_labels = [f"{f:.1f}x" for f in factors]
        counts = [adjustment_counts.get(i, 0) for i in range(len(factors))]
        plt.bar(factor_labels, counts)
        plt.title('Overall Adjustment Factor Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average adjustment by context
        plt.subplot(2, 2, 2)
        
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
        
        # Plot 3: Band-specific adjustments
        plt.subplot(2, 2, 3)
        
        band_avgs = []
        bands = []
        
        if 'sku_band' in adjustments_df.columns:
            for band in sorted(adjustments_df['sku_band'].unique()):
                band_df = adjustments_df[adjustments_df['sku_band'] == band]
                if len(band_df) > 0:
                    bands.append(band)
                    band_avgs.append(band_df['adjustment_factor'].mean())
        
        if bands:
            plt.bar(bands, band_avgs, color=['blue', 'green', 'gray', 'orange', 'red'][:len(bands)])
            plt.title('Average Adjustment Factor by SKU Band')
            plt.ylabel('Avg Factor')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(band_avgs):
                plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        # Plot 4: Pattern-specific adjustments
        plt.subplot(2, 2, 4)
        
        pattern_avgs = []
        patterns = []
        
        if 'pattern_type' in adjustments_df.columns:
            for pattern in adjustments_df['pattern_type'].unique():
                if pattern != "unknown":
                    pattern_df = adjustments_df[adjustments_df['pattern_type'] == pattern]
                    if len(pattern_df) > 0:
                        patterns.append(pattern)
                        pattern_avgs.append(pattern_df['adjustment_factor'].mean())
        
        if patterns:
            plt.bar(patterns, pattern_avgs)
            plt.title('Average Adjustment Factor by Pattern Type')
            plt.ylabel('Avg Factor')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(pattern_avgs):
                plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'forecast_adjustments.png'))
        plt.close()
    
    def _plot_pattern_learning(self):
        """Create plot showing pattern-specific learning curves."""
        if not self.pattern_metrics:
            return  # No pattern data to plot
        
        plt.figure(figsize=(16, 12))
        
        # Plot pattern-specific MAPE improvements
        plt.subplot(2, 2, 1)
        for pattern, metrics in self.pattern_metrics.items():
            plt.plot(metrics['mape_improvements'], label=pattern)
        plt.title('Pattern-Specific MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot pattern-specific scores
        plt.subplot(2, 2, 2)
        for pattern, metrics in self.pattern_metrics.items():
            plt.plot(metrics['scores'], label=pattern)
        plt.title('Pattern-Specific Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot average MAPE improvement by pattern (last 100 episodes)
        plt.subplot(2, 2, 3)
        window = min(100, len(next(iter(self.pattern_metrics.values()))['mape_improvements']))
        pattern_avgs = []
        pattern_names = []
        
        for pattern, metrics in self.pattern_metrics.items():
            avg_imp = np.mean(metrics['mape_improvements'][-window:])
            pattern_avgs.append(avg_imp)
            pattern_names.append(pattern)
        
        plt.bar(pattern_names, pattern_avgs)
        plt.title(f'Avg MAPE Improvement by Pattern (Last {window} Episodes)')
        plt.ylabel('Improvement Ratio')
        plt.xticks(rotation=45)
        
        # Plot moving average of each pattern's MAPE improvement
        plt.subplot(2, 2, 4)
        window_size = 25
        for pattern, metrics in self.pattern_metrics.items():
            if len(metrics['mape_improvements']) >= window_size:
                moving_avg = np.convolve(metrics['mape_improvements'], np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(metrics['mape_improvements'])), moving_avg, label=pattern)
        plt.title(f'Moving Average ({window_size} episodes) MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'pattern_learning.png'))
        plt.close()
    
    def _plot_band_learning(self):
        """Create plot showing band-specific learning curves."""
        plt.figure(figsize=(16, 12))
        
        # Plot band-specific MAPE improvements
        plt.subplot(2, 2, 1)
        for band in ['A', 'B', 'C', 'D', 'E']:
            plt.plot(self.band_metrics[band]['mape_improvements'], label=f'Band {band}')
        plt.title('Band-Specific MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot band-specific bias improvements
        plt.subplot(2, 2, 2)
        for band in ['A', 'B', 'C', 'D', 'E']:
            plt.plot(self.band_metrics[band]['bias_improvements'], label=f'Band {band}')
        plt.title('Band-Specific Bias Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot average MAPE improvement by band (last 100 episodes)
        plt.subplot(2, 2, 3)
        window = min(100, len(self.band_metrics['A']['mape_improvements']))
        bands = ['A', 'B', 'C', 'D', 'E']
        band_avgs = []
        
        for band in bands:
            avg_imp = np.mean(self.band_metrics[band]['mape_improvements'][-window:])
            band_avgs.append(avg_imp)
        
        plt.bar(bands, band_avgs)
        plt.title(f'Avg MAPE Improvement by Band (Last {window} Episodes)')
        plt.ylabel('Improvement Ratio')
        
        # Plot average Bias improvement by band (last 100 episodes)
        plt.subplot(2, 2, 4)
        window = min(100, len(self.band_metrics['A']['bias_improvements']))
        bands = ['A', 'B', 'C', 'D', 'E']
        band_bias_avgs = []
        
        for band in bands:
            avg_imp = np.mean(self.band_metrics[band]['bias_improvements'][-window:])
            band_bias_avgs.append(avg_imp)
        
        plt.bar(bands, band_bias_avgs)
        plt.title(f'Avg Bias Improvement by Band (Last {window} Episodes)')
        plt.ylabel('Improvement Ratio')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'band_learning.png'))
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
        context_metrics = {
            'holiday': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0},
            'promotion': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0},
            'weekend': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0},
            'weekday': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0}
        }
        
        # Band-specific metrics
        band_metrics = {
            'A': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0},
            'B': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0},
            'C': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0},
            'D': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0},
            'E': {'avg_mape_improvement': 0, 'avg_bias_improvement': 0}
        }
        
        # Store current agent state
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # Turn off exploration for evaluation
        
        # Evaluation loop
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            episode_score = 0
            
            # Track metrics for this episode
            original_mapes = []
            adjusted_mapes = []
            original_biases = []
            adjusted_biases = []
            
            for step in range(self.max_steps):
                adjustments = {}
                
                # Get actions for all SKUs (no exploration)
                for i, sku in enumerate(self.env.skus):
                    # Get feature dimensions
                    feature_dims = self.env.get_feature_dims()
                    
                    # Extract context features
                    context_features = self._extract_context_features(state[i], feature_dims)
                    sku_band = context_features.get('sku_band', 'C')
                    
                    # Create context dictionary
                    context = {
                        'is_holiday': False,  # Will be set after env step
                        'is_promotion': False,  # Will be set after env step
                        'is_weekend': context_features.get('is_weekend', False),
                        'sku_band': sku_band
                    }
                    
                    # Get action (no exploration)
                    action_idx = self.agent.act(state[i], explore=False, context=context)
                    
                    # Calculate adjusted forecast
                    forecast = state[i][0]  # First value is the current forecast
                    adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, forecast, context)
                    
                    adjustments[sku] = (action_idx, adjusted_forecast)
                
                # Take step in environment
                next_state, rewards, done, info = self.env.step(adjustments)
                
                # Update metrics
                episode_score += sum(rewards.values())
                
                # Track MAPE and bias
                for sku in self.env.skus:
                    original_mapes.append(info['original_mape'][sku])
                    adjusted_mapes.append(info['adjusted_mape'][sku])
                    original_biases.append(abs(info['original_bias'][sku]))
                    adjusted_biases.append(abs(info['adjusted_bias'][sku]))
                
                state = next_state
                if done:
                    break
            
            # Calculate episode metrics
            avg_original_mape = np.mean(original_mapes)
            avg_adjusted_mape = np.mean(adjusted_mapes)
            avg_original_bias = np.mean(original_biases)
            avg_adjusted_bias = np.mean(adjusted_biases)
            
            # Calculate improvements
            mape_improvement = (avg_original_mape - avg_adjusted_mape) / avg_original_mape if avg_original_mape > 0 else 0
            bias_improvement = (avg_original_bias - avg_adjusted_bias) / avg_original_bias if avg_original_bias > 0 else 0
            
            # Store metrics
            eval_scores.append(episode_score)
            eval_mape_improvements.append(mape_improvement)
            eval_bias_improvements.append(bias_improvement)
            
            if verbose:
                self.logger.info(f"Episode {episode}/{num_episodes} | Score: {episode_score:.2f} | MAPE Imp: {mape_improvement:.4f}")
        
        # Restore agent state
        self.agent.epsilon = original_epsilon
        
        # Calculate aggregate metrics
        avg_score = np.mean(eval_scores)
        avg_mape_improvement = np.mean(eval_mape_improvements)
        avg_bias_improvement = np.mean(eval_bias_improvements)
        
        # Approximate context-specific metrics from training data
        context_metrics = {
            'holiday': {
                'avg_mape_improvement': np.mean(self.holiday_metrics['mape_improvements'][-50:]) if self.holiday_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(self.holiday_metrics['bias_improvements'][-50:]) if self.holiday_metrics['bias_improvements'] else 0
            },
            'promotion': {
                'avg_mape_improvement': np.mean(self.promo_metrics['mape_improvements'][-50:]) if self.promo_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(self.promo_metrics['bias_improvements'][-50:]) if self.promo_metrics['bias_improvements'] else 0
            },
            'weekend': {
                'avg_mape_improvement': np.mean(self.weekend_metrics['mape_improvements'][-50:]) if self.weekend_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(self.weekend_metrics['bias_improvements'][-50:]) if self.weekend_metrics['bias_improvements'] else 0
            },
            'weekday': {
                'avg_mape_improvement': np.mean(self.weekday_metrics['mape_improvements'][-50:]) if self.weekday_metrics['mape_improvements'] else 0,
                'avg_bias_improvement': np.mean(self.weekday_metrics['bias_improvements'][-50:]) if self.weekday_metrics['bias_improvements'] else 0
            }
        }
        
        # Approximate band-specific metrics from training data
        band_metrics = {}
        for band in ['A', 'B', 'C', 'D', 'E']:
            if band in self.band_metrics and self.band_metrics[band]['mape_improvements']:
                band_metrics[band] = {
                    'avg_mape_improvement': np.mean(self.band_metrics[band]['mape_improvements'][-50:]),
                    'avg_bias_improvement': np.mean(self.band_metrics[band]['bias_improvements'][-50:])
                }
            else:
                band_metrics[band] = {
                    'avg_mape_improvement': 0,
                    'avg_bias_improvement': 0
                }
        
        metrics = {
            'scores': eval_scores,
            'mape_improvements': eval_mape_improvements,
            'bias_improvements': eval_bias_improvements,
            'avg_score': avg_score,
            'avg_mape_improvement': avg_mape_improvement,
            'avg_bias_improvement': avg_bias_improvement,
            'context_metrics': context_metrics,
            'band_metrics': band_metrics
        }
        
        return metrics
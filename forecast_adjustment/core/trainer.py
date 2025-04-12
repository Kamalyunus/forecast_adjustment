"""
Simplified trainer module for forecast adjustment using reinforcement learning.
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
    Trainer for the forecast adjustment system with only essential functionality.
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
            
        # Store original agent parameters
        self.original_ucb_constant = agent.ucb_constant
        
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
            phase_ucb = phase.get('ucb_constant', 2.0)
            phase_lr = phase.get('learning_rate', 0.005)
            phase_batch_size = phase.get('batch_size', self.batch_size)
            
            self.logger.info(f"Starting training phase {phase_idx + 1}: {phase.get('description', f'Phase {phase_idx + 1}')}")
            self.logger.info(f"  Episodes: {phase_episodes}, UCB Constant: {phase_ucb}, Learning Rate: {phase_lr}")
            
            # Update agent parameters for this phase
            self.agent.ucb_constant = phase_ucb
            self.agent.learning_rate = phase_lr
            
            # Train for this phase's episodes
            for episode in tqdm(range(1, phase_episodes + 1), disable=not verbose):
                global_episode = episode_counter + episode
                
                state = self.env.reset()
                episode_score = 0
                episode_td_errors = []
                
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
                        
                        # Get context for this SKU
                        context = {
                            'is_holiday': False,  # Will be updated after the environment step
                            'is_promotion': False,  # Will be updated after the environment step
                            'is_weekend': context_features.get('is_weekend', False),
                            'sku_band': context_features.get('sku_band', 'C')
                        }
                        
                        # Get action from agent with context
                        action_idx = self.agent.act(sku_state, explore=True, context=context)
                        
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
                    
                    # Update agent for each SKU
                    for i, sku in enumerate(self.env.skus):
                        sku_state = state[i]
                        next_sku_state = next_state[i]
                        action_idx, _ = adjustments[sku]
                        
                        # Extract context features
                        context_features = self._extract_context_features(sku_state, feature_dims)
                        
                        # Get context for this SKU
                        context = {
                            'is_holiday': info['is_holiday'].get(sku, False),
                            'is_promotion': info['is_promotion'].get(sku, False),
                            'is_weekend': context_features.get('is_weekend', False),
                            'sku_band': context_features.get('sku_band', 'C')
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
                                  f"UCB: {self.agent.ucb_constant:.2f} | "
                                  f"TD Error: {avg_td_error:.4f}")
                
                # Save models periodically
                if global_episode % self.save_every == 0:
                    model_path = os.path.join(self.model_dir, f"model_episode_{global_episode}.pkl")
                    self.agent.save(model_path)
                    
                    # Plot simplified training progress
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
            'final_model_path': final_model_path
        }
        
        return metrics
    
    def _plot_training_progress(self):
        """Create simplified plot of training progress metrics."""
        plt.figure(figsize=(15, 5))
        
        # Plot overall scores
        plt.subplot(1, 3, 1)
        plt.plot(self.scores)
        plt.title('Overall Training Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Plot MAPE improvement
        plt.subplot(1, 3, 2)
        plt.plot(self.mape_improvements)
        plt.title('Overall MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.grid(True, alpha=0.3)
        
        # Plot bias improvement
        plt.subplot(1, 3, 3)
        plt.plot(self.bias_improvements)
        plt.title('Overall Bias Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        plt.grid(True, alpha=0.3)
        
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
                    
                    # Get actual sales value if available
                    target_date = current_date + pd.Timedelta(days=forecast_day) if current_date else None
                    actual_sales = None
                    
                    if target_date and hasattr(self.env, 'actual_values'):
                        if sku in self.env.actual_values and target_date in self.env.actual_values[sku]:
                            actual_sales = float(self.env.actual_values[sku][target_date])
                    
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
                        'forecast_date': (current_date + pd.Timedelta(days=forecast_day)).strftime('%Y-%m-%d') if current_date else None,
                        'original_forecast': float(original_forecast),
                        'adjustment_factor': float(factor),
                        'adjusted_forecast': float(adjusted_forecast),
                        'actual_sales': actual_sales,
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
        
        # Calculate error metrics for rows with actual sales data
        rows_with_sales = df[df['actual_sales'].notna()]
        if len(rows_with_sales) > 0:
            # Calculate MAPE for original and adjusted forecasts
            rows_with_sales['original_mape'] = abs(rows_with_sales['original_forecast'] - rows_with_sales['actual_sales']) / rows_with_sales['actual_sales']
            rows_with_sales['adjusted_mape'] = abs(rows_with_sales['adjusted_forecast'] - rows_with_sales['actual_sales']) / rows_with_sales['actual_sales']
            
            # Calculate bias for original and adjusted forecasts
            rows_with_sales['original_bias'] = (rows_with_sales['original_forecast'] - rows_with_sales['actual_sales']) / rows_with_sales['actual_sales']
            rows_with_sales['adjusted_bias'] = (rows_with_sales['adjusted_forecast'] - rows_with_sales['actual_sales']) / rows_with_sales['actual_sales']
            
            # Log metrics
            self.logger.info(f"Rows with actual sales data: {len(rows_with_sales)} ({len(rows_with_sales)/len(df)*100:.1f}%)")
            self.logger.info(f"Original forecast MAPE: {rows_with_sales['original_mape'].mean():.4f}")
            self.logger.info(f"Adjusted forecast MAPE: {rows_with_sales['adjusted_mape'].mean():.4f}")
            self.logger.info(f"Original forecast bias: {rows_with_sales['original_bias'].mean():.4f}")
            self.logger.info(f"Adjusted forecast bias: {rows_with_sales['adjusted_bias'].mean():.4f}")
            
            # Add error metrics to the dataframe
            df = pd.merge(df, rows_with_sales[['sku_id', 'forecast_date', 'original_mape', 'adjusted_mape', 'original_bias', 'adjusted_bias']], 
                         on=['sku_id', 'forecast_date'], how='left')
        
        # Save adjusted forecasts to CSV
        output_path = os.path.join(self.output_dir, "adjusted_forecasts.csv")
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved adjusted forecasts to {output_path}")
        
        return df
    
    def evaluate(self, num_episodes: int = 10, verbose: bool = True) -> Dict:
        """
        Evaluate the forecast adjustment agent.
        
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
        
        # Store current agent state
        original_ucb_constant = self.agent.ucb_constant
        self.agent.ucb_constant = 0  # Turn off exploration for evaluation
        
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
                    
                    # Create context dictionary
                    context = {
                        'is_holiday': False,  # Will be set after env step
                        'is_promotion': False,  # Will be set after env step
                        'is_weekend': context_features.get('is_weekend', False),
                        'sku_band': context_features.get('sku_band', 'C')
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
        self.agent.ucb_constant = original_ucb_constant
        
        # Calculate aggregate metrics
        avg_score = np.mean(eval_scores)
        avg_mape_improvement = np.mean(eval_mape_improvements)
        avg_bias_improvement = np.mean(eval_bias_improvements)
        
        metrics = {
            'scores': eval_scores,
            'mape_improvements': eval_mape_improvements,
            'bias_improvements': eval_bias_improvements,
            'avg_score': avg_score,
            'avg_mape_improvement': avg_mape_improvement,
            'avg_bias_improvement': avg_bias_improvement
        }
        
        return metrics
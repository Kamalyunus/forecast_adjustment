"""
RL agent for forecast adjustments.
Implements a policy gradient agent with experience replay for delayed rewards.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
import os

from models.networks import PolicyNetwork, ValueNetwork

logger = logging.getLogger(__name__)

# Enhanced agent implementation for models/agent.py

class ForecastAdjustmentAgent:
    """
    Enhanced reinforcement learning agent for forecast adjustments.
    Uses policy gradient with a value function baseline (Actor-Critic).
    Added auxiliary learning and experience prioritization.
    """
    
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize the RL agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary
        """
        self.config = config
        
        # Check if config has the correct structure
        if 'AGENT_CONFIG' in config:
            self.agent_config = config['AGENT_CONFIG']
        else:
            # Fallback if config structure is wrong
            logger.warning("Config doesn't have expected structure. Using defaults.")
            from config import AGENT_CONFIG
            self.agent_config = AGENT_CONFIG
        
        if 'SYSTEM_CONFIG' in config:
            self.system_config = config['SYSTEM_CONFIG']
        else:
            # Fallback if config structure is wrong
            logger.warning("Config doesn't have expected structure. Using defaults.")
            from config import SYSTEM_CONFIG
            self.system_config = SYSTEM_CONFIG
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = self.agent_config.get('learning_rate', 0.001)
        self.gamma = self.agent_config.get('gamma', 0.95)
        self.batch_size = self.agent_config.get('batch_size', 128)  # Larger batch size for more stable learning
        
        # Learning rate schedule
        self.lr_decay = self.agent_config.get('lr_decay', 0.99)
        self.min_lr = self.agent_config.get('min_lr', 0.0001)
        
        # Configure GPU usage with proper error handling
        try:
            import torch
            # Default to CPU if any issue occurs
            device_str = "cpu"
            
            # Check if device is specified in config
            if 'device' in self.system_config:
                # Check if CUDA is available when 'cuda' is requested
                if self.system_config['device'] == 'cuda' and torch.cuda.is_available():
                    device_str = 'cuda'
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                elif self.system_config['device'] == 'cuda':
                    logger.warning("CUDA requested but not available. Using CPU instead.")
            
            self.device = torch.device(device_str)
            
            # Set pytorch to use current device
            if self.device.type == 'cuda':
                torch.cuda.set_device(0)
                # Enable CUDA optimization
                torch.backends.cudnn.benchmark = True
        except Exception as e:
            logger.error(f"Error setting up device: {str(e)}. Using CPU.")
            self.device = torch.device("cpu")
        
        # Set random seed for reproducibility
        import random
        import numpy as np
        torch.manual_seed(self.system_config.get('random_seed', 42))
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.system_config.get('random_seed', 42))
        random.seed(self.system_config.get('random_seed', 42))
        np.random.seed(self.system_config.get('random_seed', 42))
        
        # Get enhanced network architecture configuration
        hidden_size_policy = self.agent_config.get('hidden_size_policy', [256, 256, 128, 64])
        hidden_size_value = self.agent_config.get('hidden_size_value', [256, 256, 128, 64])
        
        # Initialize neural networks with enhanced architectures
        from models.networks import PolicyNetwork, ValueNetwork, BiasPredictor
        
        self.policy_net = PolicyNetwork(
            state_dim, 
            action_dim, 
            hidden_size_policy
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim, 
            hidden_size_value
        ).to(self.device)
        
        # Auxiliary network for bias prediction (helps with representation learning)
        self.bias_predictor = BiasPredictor(
            state_dim,
            hidden_sizes=[128, 64]
        ).to(self.device)
        
        # Initialize optimizers with learning rate schedules
        import torch.optim as optim
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.lr,
            weight_decay=1e-5  # L2 regularization
        )
        
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), 
            lr=self.lr,
            weight_decay=1e-5
        )
        
        self.bias_optimizer = optim.Adam(
            self.bias_predictor.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        # Add learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.ExponentialLR(
            self.policy_optimizer, gamma=self.lr_decay)
        self.value_scheduler = optim.lr_scheduler.ExponentialLR(
            self.value_optimizer, gamma=self.lr_decay)
        self.bias_scheduler = optim.lr_scheduler.ExponentialLR(
            self.bias_optimizer, gamma=self.lr_decay)
        
        # Set mixed precision if enabled and using GPU
        self.mixed_precision = self.system_config.get('mixed_precision', False) and self.device.type == 'cuda'
        if self.mixed_precision:
            try:
                # Check if using PyTorch version with native AMP support
                import torch.cuda.amp
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Using mixed precision training with automatic mixed precision")
            except ImportError:
                self.mixed_precision = False
                logger.warning("Mixed precision requested but not supported in current PyTorch version")
        
        # Enhanced experience replay with prioritization
        self.use_prioritized_replay = self.agent_config.get('use_prioritized_replay', True)
        if self.use_prioritized_replay:
            # Prioritized experience replay parameters
            self.alpha = self.agent_config.get('priority_alpha', 0.6)  # Priority exponent
            self.beta = self.agent_config.get('priority_beta', 0.4)    # Importance sampling exponent
            self.beta_increment = self.agent_config.get('beta_increment', 0.001)  # Increment per update
            self.epsilon = 1e-5  # Small constant to avoid zero priorities
            
            # Memory with priorities
            self.memory = []
            self.priorities = []
            self.max_priority = 1.0
        else:
            # Standard replay buffer
            from collections import deque
            self.memory = deque(maxlen=self.agent_config.get('memory_size', 100000))
        
        # Tracking metrics
        self.policy_losses = []
        self.value_losses = []
        self.bias_losses = []
        self.mean_rewards = []
        self.episode_count = 0
        self.update_count = 0
        
        # Learning step counter
        self.learning_step = 0
        
        # Performance tracking
        self.best_reward = float('-inf')
        self.best_model_path = None
    
    def select_action(self, state, training=True):
        """
        Select an action based on current policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            Tuple of (action_idx, action_probs)
        """
        import torch
        import numpy as np
        
        # Convert state to tensor and move to device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities from policy network
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor).cpu().numpy()[0]
        
        if training:
            # Annealing epsilon-greedy exploration
            epsilon = max(0.01, 0.3 * (0.99 ** self.learning_step))
            
            if np.random.random() < epsilon:
                # Random exploration
                action_idx = np.random.choice(self.action_dim)
            else:
                # Sample from probability distribution (guided exploration)
                action_idx = np.random.choice(self.action_dim, p=action_probs)
        else:
            # Take most probable action (exploitation)
            action_idx = np.argmax(action_probs)
        
        return action_idx, action_probs
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """
        Store experience in replay buffer with prioritization.
        
        Args:
            state: Current state
            action_idx: Action index that was taken
            reward: Reward received
            next_state: Next state observed
            done: Whether episode is done
        """
        if self.use_prioritized_replay:
            # For new experiences, use max priority to ensure they're sampled
            self.memory.append((state, action_idx, reward, next_state, done))
            self.priorities.append(self.max_priority)
            
            # Trim memory and priorities if they exceed memory size
            memory_size = self.agent_config.get('memory_size', 100000)
            if len(self.memory) > memory_size:
                self.memory = self.memory[-memory_size:]
                self.priorities = self.priorities[-memory_size:]
        else:
            # Standard experience storage
            self.memory.append((state, action_idx, reward, next_state, done))
        
        if len(self.memory) % 1000 == 0:
            logger.info(f"Experience buffer size: {len(self.memory)}")
    
    def _get_prioritized_batch(self):
        """
        Sample a batch based on priorities.
        
        Returns:
            Tuple of (batch, indices, weights)
        """
        import numpy as np
        import torch
        
        # Increase beta over time to reduce bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.alpha
        probs /= np.sum(probs)
        
        # Sample batch indices based on priorities
        indices = np.random.choice(
            len(self.memory), 
            min(self.batch_size, len(self.memory)), 
            replace=False, 
            p=probs
        )
        
        # Get the experiences
        batch = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        weights = torch.FloatTensor(weights).to(self.device)
        
        return batch, indices, weights
    
    def _update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences in memory
            td_errors: TD errors for these experiences
        """
        for idx, td_error in zip(indices, td_errors):
            # Convert TD error to priority
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
    
    def update_policy(self):
        """
        Update policy based on collected experiences.
        Uses Actor-Critic method with policy gradient and prioritized replay.
        
        Returns:
            Tuple of (policy_loss, value_loss, bias_loss)
        """
        import torch
        import torch.nn.functional as F
        import torch.nn as nn
        import numpy as np
        
        if self.use_prioritized_replay:
            if len(self.memory) < self.batch_size:
                logger.debug(f"Not enough experiences for training: {len(self.memory)}/{self.batch_size}")
                return None, None, None
        else:
            if len(self.memory) < self.batch_size:
                logger.debug(f"Not enough experiences for training: {len(self.memory)}/{self.batch_size}")
                return None, None, None
        
        # Sample batch of experiences
        if self.use_prioritized_replay:
            batch, indices, weights = self._get_prioritized_batch()
            states, action_idxs, rewards, next_states, dones = zip(*batch)
        else:
            import random
            batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
            states, action_idxs, rewards, next_states, dones = zip(*batch)
            weights = torch.ones(len(batch)).to(self.device)  # Equal weights
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_idxs = torch.LongTensor(action_idxs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Use mixed precision if enabled
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                # Calculate advantage estimates
                current_values = self.value_net(states).squeeze()
                next_values = self.value_net(next_states).squeeze()
                
                # TD error as advantage estimate
                targets = rewards + self.gamma * next_values * (1 - dones)
                advantages = targets - current_values
                td_errors = advantages.detach().cpu().numpy()
                
                # Get action probabilities and log probabilities
                action_probs = self.policy_net(states)
                action_probs_for_actions = action_probs.gather(1, action_idxs.unsqueeze(1)).squeeze()
                log_probs = torch.log(action_probs_for_actions + 1e-10)  # Add small epsilon for numerical stability
                
                # Calculate losses with importance sampling weights
                policy_loss = -((log_probs * advantages.detach()) * weights).mean()
                value_loss = (F.mse_loss(current_values, targets.detach(), reduction='none') * weights).mean()
                
                # Auxiliary bias prediction loss
                bias_preds = self.bias_predictor(states).squeeze()
                # Extract bias values from states (assuming bias is a feature in the state)
                # This is a placeholder - adjust based on your actual state representation
                bias_feature_idx = 3  # Adjust this index based on your state representation
                bias_targets = states[:, bias_feature_idx]
                bias_loss = (F.mse_loss(bias_preds, bias_targets, reduction='none') * weights).mean()
            
            # Update networks with gradient scaling
            
            # Policy network update
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            self.scaler.unscale_(self.policy_optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
            self.scaler.step(self.policy_optimizer)
            
            # Value network update
            self.value_optimizer.zero_grad()
            self.scaler.scale(value_loss).backward()
            self.scaler.unscale_(self.value_optimizer)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.scaler.step(self.value_optimizer)
            
            # Bias predictor update
            self.bias_optimizer.zero_grad()
            self.scaler.scale(bias_loss).backward()
            self.scaler.unscale_(self.bias_optimizer)
            torch.nn.utils.clip_grad_norm_(self.bias_predictor.parameters(), 1.0)
            self.scaler.step(self.bias_optimizer)
            
            # Update scaler
            self.scaler.update()
        else:
            # Regular training without mixed precision
            # Calculate advantage estimates
            current_values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            
            # TD error as advantage estimate
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - current_values
            td_errors = advantages.detach().cpu().numpy()
            
            # Get action probabilities and log probabilities
            action_probs = self.policy_net(states)
            action_probs_for_actions = action_probs.gather(1, action_idxs.unsqueeze(1)).squeeze()
            log_probs = torch.log(action_probs_for_actions + 1e-10)  # Add small epsilon for numerical stability
            
            # Calculate losses with importance sampling weights
            policy_loss = -((log_probs * advantages.detach()) * weights).mean()
            value_loss = (F.mse_loss(current_values, targets.detach(), reduction='none') * weights).mean()
            
            # Auxiliary bias prediction loss
            bias_preds = self.bias_predictor(states).squeeze()
            # Extract bias values from states (assuming bias is a feature in the state)
            # This is a placeholder - adjust based on your actual state representation
            bias_feature_idx = 3  # Adjust this index based on your state representation
            bias_targets = states[:, bias_feature_idx]
            bias_loss = (F.mse_loss(bias_preds, bias_targets, reduction='none') * weights).mean()
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()
            
            # Update bias predictor
            self.bias_optimizer.zero_grad()
            bias_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bias_predictor.parameters(), 1.0)
            self.bias_optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            self._update_priorities(indices, td_errors)
        
        # Update learning rate schedulers periodically
        self.learning_step += 1
        if self.learning_step % 1000 == 0:
            self.policy_scheduler.step()
            self.value_scheduler.step()
            self.bias_scheduler.step()
            
            # Log current learning rates
            policy_lr = self.policy_optimizer.param_groups[0]['lr']
            value_lr = self.value_optimizer.param_groups[0]['lr']
            logger.info(f"Updated learning rates: Policy={policy_lr:.6f}, Value={value_lr:.6f}")
        
        # Track losses
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        bias_loss_val = bias_loss.item()
        
        self.policy_losses.append(policy_loss_val)
        self.value_losses.append(value_loss_val)
        self.bias_losses.append(bias_loss_val)
        self.update_count += 1
        
        logger.debug(f"Updated policy: Policy Loss={policy_loss_val:.4f}, Value Loss={value_loss_val:.4f}, " +
                    f"Bias Loss={bias_loss_val:.4f}")
        
        return policy_loss_val, value_loss_val, bias_loss_val
    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        import torch
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save network parameters and optimizer states
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'bias_predictor_state_dict': self.bias_predictor.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'bias_optimizer': self.bias_optimizer.state_dict(),
            'learning_step': self.learning_step,
            'best_reward': self.best_reward,
        }, path)
        
        # Save metrics separately
        metrics_path = path.replace('.pt', '_metrics.pt')
        torch.save({
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'bias_losses': self.bias_losses,
            'mean_rewards': self.mean_rewards,
            'episode_count': self.episode_count,
            'update_count': self.update_count
        }, metrics_path, pickle_protocol=4)
        
        # Save prioritized replay memory if enabled
        if self.use_prioritized_replay and len(self.memory) > 0:
            memory_path = path.replace('.pt', '_memory.pkl')
            import pickle
            with open(memory_path, 'wb') as f:
                pickle.dump({
                    'memory': self.memory,
                    'priorities': self.priorities,
                    'max_priority': self.max_priority
                }, f, protocol=4)
            
            logger.info(f"Memory saved to {memory_path}")
        # Or save standard memory if it exists
        elif hasattr(self, 'memory') and len(self.memory) > 0:
            memory_path = path.replace('.pt', '_memory.pkl')
            import pickle
            with open(memory_path, 'wb') as f:
                pickle.dump(self.memory, f, protocol=4)
            
            logger.info(f"Memory saved to {memory_path}")
        
        logger.info(f"Model saved to {path}")
        logger.info(f"Metrics saved to {metrics_path}")

    def load_model(self, path):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        import torch
        import os
        
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        
        try:
            # Add deque to safe globals for PyTorch 2.6+ compatibility
            import torch.serialization
            from collections import deque
            try:
                torch.serialization.add_safe_globals([deque])
            except AttributeError:
                logger.warning("torch.serialization.add_safe_globals not available in this PyTorch version")
            
            # Load checkpoint with error handling
            try:
                checkpoint = torch.load(path, map_location=self.device)
            except RuntimeError as e:
                logger.warning(f"Error loading model with standard method: {str(e)}")
                logger.info("Trying alternative loading method...")
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            
            # Load network parameters
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            
            # Load bias predictor if available
            if 'bias_predictor_state_dict' in checkpoint:
                self.bias_predictor.load_state_dict(checkpoint['bias_predictor_state_dict'])
            
            # Load optimizer states
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            
            # Load bias optimizer if available
            if 'bias_optimizer' in checkpoint:
                self.bias_optimizer.load_state_dict(checkpoint['bias_optimizer'])
            
            # Load other attributes if available
            if 'learning_step' in checkpoint:
                self.learning_step = checkpoint['learning_step']
            if 'best_reward' in checkpoint:
                self.best_reward = checkpoint['best_reward']
            
            # Try to load metrics
            metrics_path = path.replace('.pt', '_metrics.pt')
            if os.path.exists(metrics_path):
                try:
                    metrics = torch.load(metrics_path, map_location=self.device)
                    if 'policy_losses' in metrics:
                        self.policy_losses = metrics['policy_losses']
                    if 'value_losses' in metrics:
                        self.value_losses = metrics['value_losses']
                    if 'bias_losses' in metrics:
                        self.bias_losses = metrics['bias_losses']
                    if 'mean_rewards' in metrics:
                        self.mean_rewards = metrics['mean_rewards']
                    if 'episode_count' in metrics:
                        self.episode_count = metrics['episode_count']
                    if 'update_count' in metrics:
                        self.update_count = metrics['update_count']
                    
                    logger.info(f"Metrics loaded from {metrics_path}")
                except Exception as e:
                    logger.warning(f"Error loading metrics: {str(e)}")
            
            # Try to load memory
            memory_path = path.replace('.pt', '_memory.pkl')
            if os.path.exists(memory_path):
                try:
                    import pickle
                    with open(memory_path, 'rb') as f:
                        memory_data = pickle.load(f)
                    
                    if isinstance(memory_data, dict) and 'memory' in memory_data:
                        # Prioritized replay memory
                        self.memory = memory_data['memory']
                        self.priorities = memory_data['priorities']
                        self.max_priority = memory_data['max_priority']
                    else:
                        # Standard memory
                        self.memory = memory_data
                    
                    logger.info(f"Memory loaded from {memory_path} ({len(self.memory)} experiences)")
                except Exception as e:
                    logger.warning(f"Error loading memory: {str(e)}")
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    def get_adjustment_for_category_band(self, data_provider, category, band, date, training=False):
        """
        Get adjustment factor for a specific category-band combination.
        
        Args:
            data_provider: Object providing data for state construction
            category: Category identifier
            band: Band identifier (A, B, C)
            date: Date for getting the adjustment
            training: Whether in training mode
            
        Returns:
            Tuple of (adjustment_factor, action_idx, action_probs)
        """
        from environment.state import StateBuilder
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Create state builder if not provided
        state_builder = StateBuilder(self.config)
        
        # Build state for this category-band
        state = state_builder.build_state(data_provider, category, band, date)
        
        # Select action based on state
        action_idx, action_probs = self.select_action(state, training)
        
        # Convert action to adjustment factor
        if 'ACTION_CONFIG' in self.config:
            adjustment_factors = self.config['ACTION_CONFIG']['adjustment_factors']
        else:
            from config import ACTION_CONFIG
            adjustment_factors = ACTION_CONFIG['adjustment_factors']
        
        adjustment_factor = adjustment_factors[action_idx]
        
        return adjustment_factor, action_idx, action_probs
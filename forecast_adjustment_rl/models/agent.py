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

class ForecastAdjustmentAgent:
    """
    Reinforcement learning agent for forecast adjustments.
    Uses policy gradient with a value function baseline (Actor-Critic).
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
        self.agent_config = config['AGENT_CONFIG']
        self.system_config = config['SYSTEM_CONFIG']
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = self.agent_config['learning_rate']
        self.gamma = self.agent_config['gamma']
        self.batch_size = self.agent_config['batch_size']
        self.device = torch.device(self.system_config['device'])
        
        # Set random seed for reproducibility
        torch.manual_seed(self.system_config['random_seed'])
        random.seed(self.system_config['random_seed'])
        np.random.seed(self.system_config['random_seed'])
        
        # Initialize neural networks
        self.policy_net = PolicyNetwork(
            state_dim, 
            action_dim, 
            self.agent_config['hidden_size_policy']
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim, 
            self.agent_config['hidden_size_value']
        ).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.lr
        )
        
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), 
            lr=self.lr
        )
        
        # Initialize experience replay buffer
        self.memory = deque(maxlen=self.agent_config['memory_size'])
        
        # Tracking metrics
        self.policy_losses = []
        self.value_losses = []
        self.mean_rewards = []
        self.episode_count = 0
        self.update_count = 0
    
    def select_action(self, state, training=True):
        """
        Select an action based on current policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            Tuple of (action_idx, action_probs)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities from policy network
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor).cpu().numpy()[0]
        
        if training:
            # Sample from probability distribution (exploration)
            action_idx = np.random.choice(self.action_dim, p=action_probs)
        else:
            # Take most probable action (exploitation)
            action_idx = np.argmax(action_probs)
        
        return action_idx, action_probs
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action_idx: Action index that was taken
            reward: Reward received
            next_state: Next state observed
            done: Whether episode is done
        """
        self.memory.append((state, action_idx, reward, next_state, done))
        
        if len(self.memory) % 1000 == 0:
            logger.info(f"Experience buffer size: {len(self.memory)}")
    
    def update_policy(self):
        """
        Update policy based on collected experiences.
        Uses Actor-Critic method with policy gradient.
        
        Returns:
            Tuple of (policy_loss, value_loss)
        """
        if len(self.memory) < self.batch_size:
            logger.debug(f"Not enough experiences for training: {len(self.memory)}/{self.batch_size}")
            return None, None
        
        # Sample batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, action_idxs, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_idxs = torch.LongTensor(action_idxs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate advantage estimates
        current_values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        
        # TD error as advantage estimate
        targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = targets - current_values
        
        # Get action probabilities and log probabilities
        action_probs = self.policy_net(states)
        action_probs_for_actions = action_probs.gather(1, action_idxs.unsqueeze(1)).squeeze()
        log_probs = torch.log(action_probs_for_actions + 1e-10)  # Add small epsilon for numerical stability
        
        # Calculate losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = nn.MSELoss()(current_values, targets.detach())
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Track losses
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        
        self.policy_losses.append(policy_loss_val)
        self.value_losses.append(value_loss_val)
        self.update_count += 1
        
        logger.debug(f"Updated policy: Policy Loss={policy_loss_val:.4f}, Value Loss={value_loss_val:.4f}")
        
        return policy_loss_val, value_loss_val
    
    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save only network parameters and optimizer states
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)
        
        # Save other data separately (metrics, etc.)
        metrics_path = path.replace('.pt', '_metrics.pt')
        torch.save({
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'mean_rewards': self.mean_rewards,
            'episode_count': self.episode_count,
            'update_count': self.update_count
        }, metrics_path, pickle_protocol=4)
        
        # Save memory separately if needed
        if hasattr(self, 'memory') and len(self.memory) > 0:
            memory_path = path.replace('.pt', '_memory.pkl')
            import pickle
            with open(memory_path, 'wb') as f:
                pickle.dump(self.memory, f, protocol=4)
        
        logger.info(f"Model saved to {path}")
        logger.info(f"Metrics saved to {metrics_path}")
        if hasattr(self, 'memory') and len(self.memory) > 0:
            memory_path = path.replace('.pt', '_memory.pkl')
            logger.info(f"Memory saved to {memory_path}")
    
    def load_model(self, path):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        
        try:
            # Add deque to safe globals for PyTorch 2.6+ compatibility
            import torch.serialization
            from collections import deque
            torch.serialization.add_safe_globals([deque])
            
            # Load checkpoint
            checkpoint = torch.load(path)
            
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            
            # Load memory and metrics if they exist in the checkpoint
            if 'memory' in checkpoint:
                self.memory = checkpoint['memory']
            if 'policy_losses' in checkpoint:
                self.policy_losses = checkpoint['policy_losses']
            if 'value_losses' in checkpoint:
                self.value_losses = checkpoint['value_losses']
            if 'mean_rewards' in checkpoint:
                self.mean_rewards = checkpoint['mean_rewards']
            if 'episode_count' in checkpoint:
                self.episode_count = checkpoint['episode_count']
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            # Fallback to loading with weights_only=False
            logger.warning(f"Error loading model with standard method, trying alternative approach: {str(e)}")
            try:
                checkpoint = torch.load(path, weights_only=False)
                
                self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                self.value_net.load_state_dict(checkpoint['value_state_dict'])
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
                
                logger.info(f"Model loaded from {path} using alternative method")
                return True
                
            except Exception as e2:
                logger.error(f"Failed to load model: {str(e2)}")
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
        from data.feature_engineering import FeatureEngineer
        
        # Create state builder
        state_builder = StateBuilder(self.config)
        
        # Build state for this category-band
        state = state_builder.build_state(data_provider, category, band, date)
        
        # Select action based on state
        action_idx, action_probs = self.select_action(state, training)
        
        # Convert action to adjustment factor
        adjustment_factor = self.config['ACTION_CONFIG']['adjustment_factors'][action_idx]
        
        return adjustment_factor, action_idx, action_probs
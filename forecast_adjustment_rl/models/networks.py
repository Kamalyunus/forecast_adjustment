"""
Neural network models for the Forecast Adjustment RL agent.
Includes Policy and Value networks for the Actor-Critic architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Policy network for the Actor-Critic RL agent.
    Maps state to action probabilities.
    """
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 64]):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_sizes: List of hidden layer sizes
        """
        super(PolicyNetwork, self).__init__()
        
        # Input layer
        layers = [nn.Linear(state_dim, hidden_sizes[0])]
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Output layer
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: State input tensor
            
        Returns:
            Action probability distribution
        """
        logits = self.model(x)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    Value network for the Actor-Critic RL agent.
    Estimates the value of a state.
    """
    
    def __init__(self, state_dim, hidden_sizes=[128, 64]):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_sizes: List of hidden layer sizes
        """
        super(ValueNetwork, self).__init__()
        
        # Input layer
        layers = [nn.Linear(state_dim, hidden_sizes[0])]
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Output layer
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: State input tensor
            
        Returns:
            State value estimation
        """
        return self.model(x)
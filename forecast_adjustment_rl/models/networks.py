
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Enhanced policy network for the Actor-Critic RL agent.
    Maps state to action probabilities with a deeper architecture.
    """
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256, 128, 64]):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_sizes: List of hidden layer sizes
        """
        super(PolicyNetwork, self).__init__()
        
        # Input layer with batch normalization for improved training
        self.input_bn = nn.BatchNorm1d(state_dim)
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # Skip connection layers
        self.skip_layers = nn.ModuleList()
        
        # Build network layers with skip connections
        for i in range(len(hidden_sizes) - 1):
            # Regular layer
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
            # Skip connection layer (if dimensions don't match)
            if hidden_sizes[i] != hidden_sizes[i+1]:
                self.skip_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            else:
                self.skip_layers.append(nn.Identity())
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], action_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: State input tensor
            
        Returns:
            Action probability distribution
        """
        # Handle batch normalization - only apply when batch size > 1
        # This avoids the "Expected more than 1 value per channel when training" error
        if x.size(0) > 1:
            x = self.input_bn(x)
        
        # Input layer
        x = F.relu(self.input_layer(x))
        
        # Hidden layers with skip connections
        for i, (layer, skip) in enumerate(zip(self.hidden_layers, self.skip_layers)):
            residual = x
            x = layer(x)
            x = F.relu(x)
            
            # Apply skip connection if possible
            skip_x = skip(residual)
            if x.size() == skip_x.size():
                x = x + skip_x
                
            # Apply dropout (only in training)
            if self.training and x.size(0) > 1:
                x = self.dropout(x)
        
        # Output layer
        logits = self.output(x)
        
        # Return softmax probabilities
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    Enhanced value network for the Actor-Critic RL agent.
    Estimates the value of a state with a deeper architecture.
    """
    
    def __init__(self, state_dim, hidden_sizes=[256, 256, 128, 64]):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_sizes: List of hidden layer sizes
        """
        super(ValueNetwork, self).__init__()
        
        # Input layer with batch normalization
        self.input_bn = nn.BatchNorm1d(state_dim)
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # Skip connection layers
        self.skip_layers = nn.ModuleList()
        
        # Build network layers with skip connections
        for i in range(len(hidden_sizes) - 1):
            # Regular layer
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
            # Skip connection layer (if dimensions don't match)
            if hidden_sizes[i] != hidden_sizes[i+1]:
                self.skip_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            else:
                self.skip_layers.append(nn.Identity())
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: State input tensor
            
        Returns:
            State value estimation
        """
        # Handle batch normalization - only apply when batch size > 1
        if x.size(0) > 1:
            x = self.input_bn(x)
        
        # Input layer
        x = F.relu(self.input_layer(x))
        
        # Hidden layers with skip connections
        for i, (layer, skip) in enumerate(zip(self.hidden_layers, self.skip_layers)):
            residual = x
            x = layer(x)
            x = F.relu(x)
            
            # Apply skip connection if possible
            skip_x = skip(residual)
            if x.size() == skip_x.size():
                x = x + skip_x
                
            # Apply dropout (only in training)
            if self.training and x.size(0) > 1:
                x = self.dropout(x)
        
        # Output layer
        return self.output(x)

# Add a new network model for predicting actual forecast bias
class BiasPredictor(nn.Module):
    """
    Network for predicting actual forecast bias.
    Can be used for auxiliary training to improve representations.
    """
    def __init__(self, state_dim, hidden_sizes=[128, 64]):
        super(BiasPredictor, self).__init__()
        
        # Input batch norm - only used with batch size > 1
        self.input_bn = nn.BatchNorm1d(state_dim)
        
        # Create sequential layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            # Only add batch norm for batch size > 1 (handled in forward)
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Final output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Tanh())  # Bias is typically between -1 and 1
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # Apply batch norm only for batch size > 1
        if x.size(0) > 1:
            x = self.input_bn(x)
        
        # Apply all layers
        for layer in self.layers:
            # Skip batch norm layers when batch size is 1
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                continue
            x = layer(x)
            
        return x
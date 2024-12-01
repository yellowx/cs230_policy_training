# policy_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_states, action_size, embedding_dim=64, hidden_sizes=[256, 256], state_representation='one_hot'):
        super(PolicyNetwork, self).__init__()
        self.state_representation = state_representation
        if self.state_representation == 'index':
            self.embedding = nn.Embedding(num_states, embedding_dim)
            input_size = embedding_dim
        elif self.state_representation == 'one_hot':
            input_size = num_states
        else:
            raise ValueError("state_representation must be 'one_hot' or 'index'")
        
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, mask=None):
        if self.state_representation == 'index':
            embedded = self.embedding(state)  # Shape: (batch_size, embedding_dim)
        else:
            embedded = state  # Shape: (batch_size, num_states)
        
        logits = self.network(embedded)  # Shape: (batch_size, action_size)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)  # Prohibit certain actions
        # Removed softmax here
        return logits  # Shape: (batch_size, action_size)

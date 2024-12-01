# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DemonstrationDataset(Dataset):
    def __init__(self, state_seqs, action_seqs, num_states, state_representation='one_hot'):
        """
        Initialize the dataset with state and action sequences.

        Args:
            state_seqs (list of np.ndarray): List where each element is an array of states for a trajectory.
            action_seqs (list of np.ndarray): List where each element is an array of actions for a trajectory.
            num_states (int): Total number of unique states in the MDP.
            state_representation (str): 'one_hot' or 'index' indicating how states are represented.
        """
        # Flatten all state-action pairs
        self.states = []
        self.actions = []
        for s_seq, a_seq in zip(state_seqs, action_seqs):
            self.states.extend(s_seq)
            self.actions.extend(a_seq)
        
        # Convert to numpy array for processing
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        
        # Process states based on representation
        if state_representation == 'one_hot':
            # One-hot encode states
            self.states = np.eye(num_states)[self.states]  # Shape: (N, num_states)
            self.states = torch.tensor(self.states, dtype=torch.float32)
        elif state_representation == 'index':
            # Keep states as indices for embedding
            self.states = torch.tensor(self.states, dtype=torch.long)  # Shape: (N,)
        else:
            raise ValueError("state_representation must be 'one_hot' or 'index'")
        
        # Convert actions to torch tensors
        self.actions = torch.tensor(self.actions, dtype=torch.long)  # Shape: (N,)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

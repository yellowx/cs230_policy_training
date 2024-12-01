# simulate.py

from mdp import GridMDP
from policy_network import PolicyNetwork
import torch
import pickle
import numpy as np

def simulate_with_constraints(mdp, constraints=None):
    """
    Simulate a single trajectory using the neural network-based policy with constraints.

    Args:
        mdp (GridMDP): The MDP instance.
        constraints (list or None): List of action indices to mask out.

    Returns:
        tuple: (state_seq, action_seq)
    """
    state_seq, action_seq = [], []
    state = np.random.choice(mdp.nS, p=mdp.D0.flatten())
    state_seq.append(state)
    
    while True:
        # Get action probabilities with constraints
        action_probs = mdp.get_action_probabilities(state, constraints=constraints)
        
        # Sample an action based on the probabilities
        action = np.random.choice(mdp.nA, p=action_probs)
        action_seq.append(action)
        
        # Determine the next state
        next_state_probs = mdp.actions[state, action, :].flatten()
        if next_state_probs.sum() == 0:
            # Terminal state or no transition
            action_seq.append(mdp.terminal_action)
            break
        next_state = np.random.choice(mdp.nS, p=next_state_probs)
        state = next_state
        state_seq.append(state)
        
        # Check for terminal condition
        if state in mdp.goal:
            action_seq.append(mdp.terminal_action)
            break
    
    return state_seq, action_seq

def main():
    # Load the nominal MDP
    with open('nominal_mdp.pkl', 'rb') as f:
        nominal_mdp = pickle.load(f)

    # Initialize GridMDP
    estimated_mdp = GridMDP(nominal_mdp)

    # Load the trained policy network
    policy_net = PolicyNetwork(
        num_states=estimated_mdp.nS,
        action_size=estimated_mdp.nA,
        embedding_dim=64,
        hidden_sizes=[256, 256],
        state_representation='index'  # Change if you used 'one_hot'
    )
    policy_net.load_state_dict(torch.load('trained_policy.pth'))
    policy_net.eval()

    # Set the policy network in the MDP
    estimated_mdp.set_policy_network(policy_net)

    # Define constraints if any
    constraints = []  # Example: [action_index1, action_index2]

    # Simulate a single trajectory
    state_seq, action_seq = simulate_with_constraints(estimated_mdp, constraints=constraints)
    
    print("Simulated State Sequence:", state_seq)
    print("Simulated Action Sequence:", action_seq)

if __name__ == "__main__":
    main()

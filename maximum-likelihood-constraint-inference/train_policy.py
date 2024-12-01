# train_policy.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from policy_network import PolicyNetwork
from dataset import DemonstrationDataset
from mdp import GridMDP  # Ensure this imports your existing MDP classes
import numpy as np
from tqdm import tqdm
import pickle
import os, argparse, shutil

def train_behavioral_cloning(nominal_mdp, demonstrations, epochs=10, batch_size=64, learning_rate=1e-3, hidden_sizes=[128, 128], state_representation='one-hot'):
    """
    Train a neural network policy using Behavioral Cloning.

    Args:
        nominal_mdp (GridMDP): The nominal MDP instance.
        demonstrations (list): A list containing state and action sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Optimizer learning rate.
        hidden_sizes (list): List of hidden layer sizes for the network.
        state_representation (str): 'one_hot' or 'index' indicating how states are represented.

    Returns:
        PolicyNetwork: The trained policy network.
    """
    state_seqs, action_seqs = demonstrations  # Assuming demonstrations is [[state_seq1, state_seq2, ...], [action_seq1, action_seq2, ...]]
    
    num_states = nominal_mdp.nS
    action_size = nominal_mdp.nA
    
    # Create dataset and dataloader
    dataset = DemonstrationDataset(state_seqs, action_seqs, num_states=num_states, state_representation=state_representation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the policy network
    policy_net = PolicyNetwork(
        num_states=num_states,
        action_size=action_size,
        embedding_dim=64,
        hidden_sizes=hidden_sizes,
        state_representation=state_representation
    )
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            action_probs = policy_net(batch_states)  # Shape: (batch_size, action_size)
            action_probs = action_probs.squeeze(1)
            batch_actions = batch_actions.argmax(dim=1)
            loss = criterion(action_probs, batch_actions)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_states.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return policy_net


def main():
    try:
        if not os.path.exists("pickles"):
            os.mkdir("pickles")
        with open('pickles/trajectories.pickle', 'rb') as handle:
            trajs = pickle.load(handle)
    except:
        print("Cannot find trajectory data! Make sure pickles/trajectories.pickle exists.")
        exit(0)

    n = 35 # 20 # dimensionality of state-space
    allowed_end_state = [945,946,947,948,980,981,982,983,1015,1016,1017,1018,1050,1051,1052,1053] # [320]
    banned_start_state = [1087] # [361]
    num_actions = 9  
    one_hot_action_trajs = []

    try:
        if not os.path.exists("pickles"):
            os.mkdir("pickles")
        with open('pickles/trajectories.pickle', 'rb') as handle:
            trajs = pickle.load(handle)
    except:
        print("Cannot find trajectory data! Make sure pickles/trajectories.pickle exists.")
        exit(0)

    new_trajs = []
    action_trajs = []
    stationary = []
    D0 = {}
    Dn = {}
    lens = []
    for traj in trajs:
        new_traj = []
        action_traj = []
        for item in traj:
            new_traj += [item[0]+item[1]*n]
        for i in range(len(traj)-1):
            dx = traj[i+1][0] - traj[i][0]
            dy = traj[i+1][1] - traj[i][1]
            if dx == -1 and dy == 0:
                action_traj += [0]
            if dx == 1 and dy == 0:
                action_traj += [1]
            if dx == 0 and dy == -1:
                action_traj += [2]
            if dx == 0 and dy == 1:
                action_traj += [3]
            if dx == -1 and dy == -1:
                action_traj += [4]
            if dx == 1 and dy == -1:
                action_traj += [5]
            if dx == -1 and dy == 1:
                action_traj += [6]
            if dx == 1 and dy == 1:
                action_traj += [7]
        action_traj += [8]
        if len(new_traj) in [1]:
            stationary += [new_traj]
        elif False and (new_traj[-1] not in allowed_end_state) or (new_traj[0] in banned_start_state):
            pass
        else:
            lens += [len(new_traj)]
            if new_traj[0] not in D0.keys():
                D0[new_traj[0]] = 0
            D0[new_traj[0]] += 1
            if new_traj[-1] not in Dn.keys():
                Dn[new_traj[-1]] = 0
            Dn[new_traj[-1]] += 1
            new_trajs += [np.array(new_traj).reshape(-1, 1)]
            # print(new_trajs[-1].reshape(-1))
            action_trajs += [np.array(action_traj).reshape(-1, 1)]
    trajs = new_trajs
    # Load or initialize your nominal MDP
    with open('nominal_mdp.pkl', 'rb') as f:
        nominal_mdp = pickle.load(f)
    for action_traj in action_trajs:
        # Flatten action_traj to ensure it's a 1D array of action indices
        action_traj = action_traj.flatten()
        
        # Create one-hot encoding using NumPy
        one_hot_encoded = np.eye(num_actions)[action_traj]
        
    # Append to the result list
    one_hot_action_trajs.append(one_hot_encoded)
    # Generate or load demonstrations
    state_seqs, action_seqs = trajs, one_hot_action_trajs
    demonstrations = [state_seqs, action_seqs]
    state_representation = 'one_hot'  # Change to 'one_hot' if preferred

    # Train the policy network
    trained_policy = train_behavioral_cloning(
        nominal_mdp=nominal_mdp,
        demonstrations=demonstrations,
        epochs=1000,
        batch_size=16,
        learning_rate=1e-4,
        hidden_sizes=[256, 256],
        state_representation=state_representation
    )

    # Save the trained policy
    torch.save(trained_policy.state_dict(), 'trained_policy.pth')
    print("Policy network trained and saved as 'trained_policy.pth'.")

if __name__ == "__main__":
    main()
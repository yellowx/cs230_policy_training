# main_integration.py

from mdp import GridMDP
from policy_network import PolicyNetwork
import torch
import pickle

def main():
    # Load the nominal MDP
    with open('nominal_mdp.pkl', 'rb') as f:
        nominal_mdp = pickle.load(f)

    # Initialize GridMDP
    estimated_mdp = GridMDP(nominal_mdp)

    # Initialize the policy network and load trained weights
    policy_net = PolicyNetwork(num_states=estimated_mdp.nS, action_size=estimated_mdp.nA)
    policy_net.load_state_dict(torch.load('trained_policy.pth'))
    policy_net.eval()  # Set to evaluation mode

    # Set the policy network in the MDP
    estimated_mdp.set_policy_network(policy_net)

    # Now, estimated_mdp uses the neural network policy
    # You can proceed to generate demonstrations or perform other operations
    demonstrations = estimated_mdp.produce_demonstrations_nn(N=100, num_demos=500)
    with open("pickles/nn_demonstrations.pickle", 'wb') as handle:
        pickle.dump(demonstrations[0], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

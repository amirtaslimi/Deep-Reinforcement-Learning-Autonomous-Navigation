import torch
from torch import nn


class PolicyNetwork(torch.nn.Module):
    """Neural network model representing the policy network."""

    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """Performs the forward pass through the network and computes action probabilities."""

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.softmax(x, dim=0)
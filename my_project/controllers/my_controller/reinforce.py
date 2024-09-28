import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import torch
from matplotlib import pyplot as plt
from torch import optim

from environment import Environment
from policy_network import PolicyNetwork
from utils import plot_rewards, plot_rewards1


class AgentREINFORCE():
    """Agent implementing the REINFORCE algorithm."""

    def __init__(self, save_path, load_path, num_episodes, max_steps,
                 learning_rate, gamma, hidden_size, clip_grad_norm, baseline, device, robot):

        self.save_path = save_path
        self.load_path = load_path
        os.makedirs(self.save_path, exist_ok=True)
        self.device = device
        # Hyper-parameters Attributes
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learing_rate = learning_rate
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.clip_grad_norm = clip_grad_norm
        self.baseline = baseline

        # Initialize Network (Model)
        self.network = PolicyNetwork(input_size=17, hidden_size=self.hidden_size, output_size=3).to(self.device)

        # Create the self.optimizers
        self.optimizer = optim.Adam(self.network.parameters(), self.learing_rate)

        # instance of env
        self.env = Environment(robot)

    def save(self, path):
        """Save the trained model parameters after final episode and after receiving the best reward."""
        torch.save(self.network.state_dict(), self.save_path + path)

    def load(self):
        """Load pre-trained model parameters."""
        # self.network.load_state_dict(torch.load(self.load_path))
        if os.path.exists(self.load_path):
            print(f"Loading weights from {self.load_path}...")
            self.network.load_state_dict(torch.load(self.save_path + '/best_weights.pt'))

        else:
            print("No previous weights found, starting from scratch.")

    def compute_returns(self, rewards):
        """
        Compute the discounted returns.

        Parameters:
        - rewards (list): List of rewards obtained during an episode.

        Returns:
        - torch.Tensor: Computed returns.
        """

        # Generate time steps and calculate discount factors
        t_steps = torch.arange(len(rewards))
        discount_factors = torch.pow(self.gamma, t_steps).to(self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Calculate returns using discounted sum
        returns = rewards * discount_factors
        returns = returns.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,)) / discount_factors

        if self.baseline:
            mean_reward = torch.mean(rewards)
            returns -= mean_reward

        return returns

    def compute_loss(self, log_probs, returns):
        """
        Compute the REINFORCE loss.

        Parameters:
        - log_probs (list): List of log probabilities of actions taken during an episode.
        - returns (torch.Tensor): Computed returns for the episode.

        Returns:
        - torch.Tensor: Computed loss.
        """

        # Calculate loss for each time step
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)

        # Sum the individual losses to get the total loss
        return torch.stack(loss).sum()

    def learn(self, rewards, log_probs):
        self.optimizer.zero_grad()
        returns = self.compute_returns(rewards)
        loss = self.compute_loss(log_probs, returns)
        loss.backward()
        # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.network.parameters(), float('inf'))
        # print("Gradient norm before clipping:", grad_norm_before_clip)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def train(self):
        """
        Train the agent using the REINFORCE algorithm.

        This method performs the training of the agent using the REINFORCE algorithm. It iterates
        over episodes, collects experiences, computes returns, and updates the policy network.
        """
        self.load()
        self.network.train()
        start_time = time.time()
        reward_history = []
        best_score = -np.inf
        for episode in range(1, self.num_episodes + 1):
            done = False
            state = self.env.reset()

            log_probs = []
            rewards = []
            ep_reward = 0
            while True:
                action_probs = self.network(torch.as_tensor(state, device=self.device))  # action probabilities
                dist = torch.distributions.Categorical(action_probs)  # Make categorical distrubation
                action = dist.sample()  # Sample action
                log_prob = dist.log_prob(
                    action)  # The log probability of the action under the current policy distribution.
                log_probs.append(log_prob)
                next_state, reward, done = self.env.step(action.item(), self.max_steps)

                rewards.append(reward)
                ep_reward += reward

                if done:
                    self.learn(rewards, log_probs)
                    reward_history.append(ep_reward)
                    if ep_reward > best_score:
                        self.save(path='/best_weights.pt')
                        best_score = ep_reward
                    if len(reward_history) == 1000 :
                        plot_rewards1(reward_history, self.save_path)
                    self.save(path='/last_weights.pt')
                    print(f"Episode {episode}: Score = {ep_reward:.3f}")
                    break

                state = next_state

        # Save final weights and plot reward history
        self.save(path='/final_weights.pt')
        plot_rewards(reward_history, self.save_path)

        # Print total training time
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')

    def test(self):
        """
        Test the trained agent.
        This method evaluates the performance of the trained agent.
        """

        start_time = time.time()
        rewards = []
        self.load()
        self.network.eval()

        for episode in range(1, self.num_episodes + 1):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action_probs = self.network(torch.as_tensor(state, device=self.device))
                dist = torch.distributions.Categorical(action_probs)  # Make categorical distrubation
                # action = dist.sample()  # Sample action
                action = torch.argmax(action_probs, dim=0)
                state, reward, done = self.env.step(action.item(), self.max_steps)
                ep_reward += reward
            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
        print(f"Mean Score = {np.mean(rewards):.3f}")

        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')




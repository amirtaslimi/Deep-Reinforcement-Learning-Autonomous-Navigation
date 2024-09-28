# Add the controller Webots Python library path
import argparse
import sys

from environment import Environment
from reinforce import AgentREINFORCE

webots_path = 'C:\Program Files\Webots\lib\controller\python'
sys.path.append(webots_path)

# Add Webots controlling libraries
from controller import Robot

# Some general libraries
import os
import numpy as np

# PyTorch
import torch

# Create an instance of robot
robot = Robot()

# Seed Everything
seed = 20
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Agent using REINFORCE Algorithm")

    # Adding Arguments
    parser.add_argument('--save_path', type=str, default='./results', help='Directory to save the results.')
    parser.add_argument('--load_path', type=str, default='./results/final_weights.pt',
                        help='Path to load the model weights.')
    parser.add_argument('--train_mode', type=bool, default=True, help='Flag to enable training mode.')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes to run.')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum number of steps per episode.')
    parser.add_argument('--learning_rate', type=float, default=25e-4, help='Learning rate for the agent.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards.')
    parser.add_argument('--hidden_size', type=int, default=34, help='Size of the hidden layer in the network.')
    parser.add_argument('--clip_grad_norm', type=float, default=5, help='Maximum gradient norm for clipping.')
    parser.add_argument('--baseline', type=bool, default=True, help='Use a baseline to reduce variance.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., cpu or cuda).')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Adjust parameters based on mode
    if args.train_mode:
        args.num_episodes = 2000
        args.max_steps = 100
    else:
        args.num_episodes = 10
        args.max_steps = 500

    # Agent Instance
    agent = AgentREINFORCE(save_path=args.save_path,
                           load_path=args.load_path,
                           num_episodes=args.num_episodes,
                           max_steps=args.max_steps,
                           learning_rate=args.learning_rate,
                           gamma=args.gamma,
                           hidden_size=args.hidden_size,
                           clip_grad_norm=args.clip_grad_norm,
                           baseline=args.baseline,
                           device=args.device,
                           robot=robot)

    # Training or Testing
    if args.train_mode:
        agent.train()
    else:
        agent.test()

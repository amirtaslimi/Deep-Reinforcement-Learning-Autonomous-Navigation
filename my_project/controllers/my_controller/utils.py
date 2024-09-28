import numpy as np
from matplotlib import pyplot as plt


def normalizer(value, min_value, max_value):
    """
    Performs min-max normalization on the given value.

    Returns:
    - float: Normalized value.
    """
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value


def get_distance_to_goal(gps, destination_coordinate, floor_size):
    """
    Calculates and returns the normalized distance from the robot's current position to the goal.

    Returns:
    - numpy.ndarray: Normalized distance vector.
    """

    gps_value = gps.getValues()[0:2]
    current_coordinate = np.array(gps_value)
    distance_to_goal = np.linalg.norm(destination_coordinate - current_coordinate)
    normalized_coordinate_vector = normalizer(distance_to_goal, min_value=0, max_value=floor_size)

    return normalized_coordinate_vector


def plot_rewards(rewards, save_path):
    # Calculate the Simple Moving Average (SMA) with a window size of 25
    sma = np.convolve(rewards, np.ones(25) / 25, mode='valid')

    plt.figure()
    plt.title("Episode Rewards")
    plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
    plt.plot(sma, label='SMA 25', color='#f0c52b')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()

    plt.savefig(save_path + '/reward_plot.png', format='png', dpi=1000, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()

def plot_rewards1(rewards, save_path):
    # Calculate the Simple Moving Average (SMA) with a window size of 25
    sma = np.convolve(rewards, np.ones(25) / 25, mode='valid')

    plt.figure()
    plt.title("Episode Rewards")
    plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
    plt.plot(sma, label='SMA 25', color='#f0c52b')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()

    plt.savefig(save_path + '/reward_plot_1000.png', format='png', dpi=1000, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()
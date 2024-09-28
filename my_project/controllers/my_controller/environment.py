from math import acos, pi

import numpy as np

from controller import Supervisor
from utils import normalizer, get_distance_to_goal


class Environment(Supervisor):
    """The robot's environment in Webots."""

    def __init__(self, robot):
        super().__init__()

        self.robot_node = self.getFromDef("pioneer3")
        self.robot = robot
        # General environment parameters
        self.max_speed = 3  # Maximum Angular speed in rad/s
        self.destination_coordinate = np.array([3.5, 0])  # Target (Goal) position
        self.reach_threshold = 0.06  # Distance threshold for considering the destination reached.
        obstacle_threshold = 0.1  # Threshold for considering proximity to obstacles.
        self.obstacle_threshold = 1 - obstacle_threshold
        self.floor_size = np.linalg.norm([10, 8])
        self.previous_distance_to_goal = 40

        # Activate Devices
        # ~~ 1) Wheel Sensors
        self.left_motor = robot.getDevice('left wheel')
        self.right_motor = robot.getDevice('right wheel')

        # Set the motors to rotate for ever
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Zero out starting velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ~~ 2) GPS Sensor
        sampling_period = 1  # in ms
        self.gps = robot.getDevice("gps")
        self.gps.enable(sampling_period)
        gps_value = 0
        # ~~ 3) Enable Touch Sensor
        self.touch = robot.getDevice("touch sensor")
        self.touch.enable(sampling_period)

        # List of all available sensors
        available_devices = list(robot.devices.keys())
        # Filter sensors name that contain 'so'
        filtered_list = [item for item in available_devices if
                         'so' in item and any(char.isdigit() for char in item)]
        filtered_list = sorted(filtered_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        robot.step(200)  # take some dummy steps in environment for initialization

        # Create dictionary from all available distance sensors and keep min and max of from total values
        self.max_sensor = 0
        self.min_sensor = 0
        self.dist_sensors = {}
        for i in filtered_list:
            self.dist_sensors[i] = robot.getDevice(i)
            self.dist_sensors[i].enable(sampling_period)
            self.max_sensor = max(self.dist_sensors[i].max_value, self.max_sensor)
            self.min_sensor = min(self.dist_sensors[i].min_value, self.min_sensor)

    def get_sensor_data(self):
        """
        Retrieves and normalizes data from distance sensors.

        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """

        # Gather values of distance sensors.
        sensor_data = []
        for z in self.dist_sensors:
            sensor_data.append(self.dist_sensors[z].value)

        sensor_data = np.array(sensor_data)
        normalized_sensor_data = normalizer(sensor_data, self.min_sensor, self.max_sensor)

        return normalized_sensor_data

    def get_observations(self):
        """
        Obtains and returns the normalized sensor data and current distance to the goal.

        Returns:
        - numpy.ndarray: State vector representing distance to goal and distance sensors value.
        """

        normalized_sensor_data = np.array(self.get_sensor_data(), dtype=np.float32)
        normalizied_current_coordinate = np.array([get_distance_to_goal(self.gps, self.destination_coordinate, self.floor_size)],
                                                  dtype=np.float32)
        state_vector = np.concatenate([normalizied_current_coordinate, normalized_sensor_data], dtype=np.float32)
        return state_vector

    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observations.

        Returns:
        - numpy.ndarray: Initial state vector.
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_observations()

    def step(self, action, max_steps):
        """
        Takes a step in the environment based on the given action.

        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """

        self.apply_action(action)
        step_reward, done = self.get_reward()

        state = self.get_observations()  # New state

        # Time-based termination condition
        if (int(self.getTime()) + 1) % max_steps == 0:
            done = True

        return state, step_reward, done

    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.

        Returns:
        - The reward and done flag.
        """

        done = False
        reward = 0

        normalized_sensor_data = self.get_sensor_data()
        normalized_current_distance = get_distance_to_goal(self.gps, self.destination_coordinate, self.floor_size)

        normalized_current_distance *= 100  # The value is between 0 and 1. Multiply by 100 will make the function work better
        reach_threshold = self.reach_threshold * 100

        # (1) Reward based on distance to goal (positive reward for progress)
        if normalized_current_distance < self.previous_distance_to_goal:
            reward += (self.previous_distance_to_goal - normalized_current_distance) * 1  # Reward for moving closer
        else:
            reward -= (normalized_current_distance - self.previous_distance_to_goal) * .5  # Penalty for moving away

        if normalized_current_distance < 2.5:
            reward = 10  # Very close to the goal
        elif normalized_current_distance < 5:
            reward = 5  # Close to the goal
        elif normalized_current_distance < 10:
            reward = 2  # Medium distance from the goal

        # Update previous distance for the next step
        self.previous_distance_to_goal = normalized_current_distance

        # (2) Reward for reaching the goal
        if normalized_current_distance < reach_threshold:
            done = True
            reward += 100  # Large reward for reaching the goal
            print('+++ SOLVED +++')

        # (3) Penalty for collision
        if self.touch.value:
            done = True
            reward -= 50  # Large penalty for collision
            print('--- COLLISION ---')

        # (4) Penalty for getting too close to obstacles
        obstacle_penalty_threshold = self.obstacle_threshold
        min_distance_to_obstacle = np.min(normalized_sensor_data)  # Get the minimum distance to any obstacle
        if min_distance_to_obstacle < obstacle_penalty_threshold:
            reward -= 0.1  # Penalize being too close to an obstacle

        current_coordinate = np.array(self.gps.getValues()[0:2])

        direction_to_goal = np.array([self.destination_coordinate[0] - current_coordinate[0],
                                      self.destination_coordinate[1] - current_coordinate[
                                          1]])  # Only use x, y for 2D plane
        robot_orientation_matrix = self.robot_node.getOrientation()
        # (4) Get the robot's heading direction (assuming 2D on the ground plane)
        robot_heading = np.array(
            [robot_orientation_matrix[0], robot_orientation_matrix[6]])  # Heading vector in x-z plane

        # Normalize both vectors to unit length
        direction_to_goal /= np.linalg.norm(direction_to_goal)
        robot_heading /= np.linalg.norm(robot_heading)

        # (5) Calculate the dot product between the direction vectors
        dot_product = np.dot(robot_heading, direction_to_goal)

        # Clamp the value to avoid floating point issues outside the domain of acos
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # (6) Calculate the angle between the robot's heading and the direction to the goal
        angle_to_goal = acos(dot_product)  # In radians

        # Convert angle to degrees for better understanding
        angle_to_goal_deg = angle_to_goal * (180 / pi)

        # (7) Reward based on how aligned the robot is with the goal direction
        if angle_to_goal_deg < 20:  # Small angle means the robot is heading toward the goal
            reward += 0.1  # Positive reward for being aligned with the goal
        else:
            reward -= 0.05  # Small penalty for being off course

        return reward, done

    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.

        Returns:
        - None
        """
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        if action == 0:  # move forward
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        elif action == 1:  # turn right
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(-self.max_speed)
        elif action == 2:  # turn left
            self.left_motor.setVelocity(-self.max_speed)
            self.right_motor.setVelocity(self.max_speed)

        self.robot.step(500)

        self.left_motor.setPosition(0)
        self.right_motor.setPosition(0)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

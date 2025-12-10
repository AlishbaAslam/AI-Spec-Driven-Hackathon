#!/usr/bin/env python3

"""
Control Publisher Agent for ROS 2
This agent demonstrates how to create a Python-based agent that publishes
control commands to robot actuators based on sensor input and decision making.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32, String, Bool
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Duration
import numpy as np
import math


class ControlPublisherAgent(Node):
    """
    A Python-based agent that publishes control commands to robot actuators.
    This agent can implement various control strategies based on sensor input.
    """

    def __init__(self):
        super().__init__('control_publisher_agent')

        # Create publisher for robot velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create publisher for robot status
        self.status_publisher = self.create_publisher(String, '/robot_status', 10)

        # Create publisher for control mode
        self.mode_publisher = self.create_publisher(String, '/control_mode', 10)

        # Create subscriber for sensor data (for reactive control)
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)

        # Create subscriber for high-level commands
        self.command_subscription = self.create_subscription(
            String,
            '/high_level_command',
            self.command_callback,
            10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # Agent state
        self.laser_data = None
        self.current_command = "IDLE"  # IDLE, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP, AVOID_OBSTACLE
        self.robot_status = "READY"
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Control parameters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.safe_distance = 0.8  # meters
        self.turn_speed = 0.5  # rad/s for turning

        # Declare parameters
        self.declare_parameter('max_linear_speed', self.max_linear_speed)
        self.declare_parameter('max_angular_speed', self.max_angular_speed)
        self.declare_parameter('safe_distance', self.safe_distance)

        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.safe_distance = self.get_parameter('safe_distance').value

        self.get_logger().info(f'Control Publisher Agent initialized with max_linear_speed={self.max_linear_speed}, '
                              f'max_angular_speed={self.max_angular_speed}, safe_distance={self.safe_distance}')

    def laser_callback(self, msg):
        """Callback function for handling laser scan data."""
        self.laser_data = msg

    def command_callback(self, msg):
        """Callback function for handling high-level commands."""
        command = msg.data.upper()
        if command in ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "AVOID_OBSTACLE", "IDLE"]:
            self.current_command = command
            self.get_logger().info(f'Received command: {command}')
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def get_min_obstacle_distance(self):
        """
        Get the minimum obstacle distance from laser data.
        Returns the minimum distance in front of the robot.
        """
        if self.laser_data is None or not self.laser_data.ranges:
            return float('inf')

        ranges = self.laser_data.ranges
        # Get the front 60 degrees (30 degrees left and right of center)
        center_idx = len(ranges) // 2
        front_range_start = max(0, center_idx - 30)
        front_range_end = min(len(ranges), center_idx + 30)

        front_ranges = ranges[front_range_start:front_range_end]
        valid_ranges = [r for r in front_ranges if r > 0 and r < float('inf')]

        if not valid_ranges:
            return float('inf')

        return min(valid_ranges)

    def execute_control_strategy(self):
        """
        Execute the current control strategy based on the command and sensor data.
        """
        obstacle_distance = self.get_min_obstacle_distance()

        if self.current_command == "IDLE":
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            self.robot_status = "IDLE"

        elif self.current_command == "STOP":
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            self.robot_status = "STOPPED"

        elif self.current_command == "MOVE_FORWARD":
            if obstacle_distance < self.safe_distance:
                # Obstacle detected, stop
                self.linear_velocity = 0.0
                self.angular_velocity = 0.0
                self.current_command = "AVOID_OBSTACLE"  # Switch to obstacle avoidance
                self.robot_status = "OBSTACLE_DETECTED"
            else:
                # Path is clear, move forward
                self.linear_velocity = self.max_linear_speed
                self.angular_velocity = 0.0
                self.robot_status = "MOVING_FORWARD"

        elif self.current_command == "TURN_LEFT":
            self.linear_velocity = 0.0
            self.angular_velocity = self.turn_speed
            self.robot_status = "TURNING_LEFT"

        elif self.current_command == "TURN_RIGHT":
            self.linear_velocity = 0.0
            self.angular_velocity = -self.turn_speed
            self.robot_status = "TURNING_RIGHT"

        elif self.current_command == "AVOID_OBSTACLE":
            if obstacle_distance > self.safe_distance + 0.3:  # Add buffer
                # Path is clear, switch back to move forward
                self.current_command = "MOVE_FORWARD"
                self.linear_velocity = self.max_linear_speed
                self.angular_velocity = 0.0
                self.robot_status = "MOVING_FORWARD"
            else:
                # Still need to avoid obstacle, turn left
                self.linear_velocity = 0.0
                self.angular_velocity = self.turn_speed
                self.robot_status = "AVOIDING_OBSTACLE"

    def execute_path_following_control(self):
        """
        Execute a more sophisticated path following control strategy.
        This is a placeholder for more advanced control algorithms.
        """
        # In a real implementation, this would include:
        # - Path planning and following algorithms
        # - PID controllers
        # - Trajectory generation
        # - Model predictive control

        # For now, implement a simple proportional controller for obstacle avoidance
        if self.laser_data is not None and self.laser_data.ranges:
            ranges = self.laser_data.ranges
            n = len(ranges)

            # Get left and right side distances for steering
            left_ranges = ranges[n//2 : 3*n//4]
            right_ranges = ranges[n//4 : n//2]

            left_distances = [r for r in left_ranges if r > 0 and r < float('inf')]
            right_distances = [r for r in right_ranges if r > 0 and r < float('inf')]

            left_avg = np.mean(left_distances) if left_distances else 1.0
            right_avg = np.mean(right_distances) if right_distances else 1.0

            # Simple proportional controller for centering
            if left_avg > 0 and right_avg > 0:
                error = left_avg - right_avg
                # Adjust angular velocity based on error
                self.angular_velocity = max(-self.max_angular_speed,
                                          min(self.max_angular_speed, error * 0.1))

    def control_loop(self):
        """
        Main control loop that runs periodically.
        """
        # Execute the appropriate control strategy
        if self.current_command == "PATH_FOLLOW":
            self.execute_path_following_control()
        else:
            self.execute_control_strategy()

        # Create and publish the control command
        cmd_msg = Twist()
        cmd_msg.linear = Vector3(x=self.linear_velocity, y=0.0, z=0.0)
        cmd_msg.angular = Vector3(x=0.0, y=0.0, z=self.angular_velocity)

        self.cmd_vel_publisher.publish(cmd_msg)

        # Publish robot status
        status_msg = String()
        status_msg.data = f"{self.robot_status}: linear={self.linear_velocity:.2f}, angular={self.angular_velocity:.2f}"
        self.status_publisher.publish(status_msg)

        # Publish control mode
        mode_msg = String()
        mode_msg.data = self.current_command
        self.mode_publisher.publish(mode_msg)

        # Log control command
        self.get_logger().info(f'Command: {self.current_command}, '
                              f'Velocity: linear={self.linear_velocity:.2f}, angular={self.angular_velocity:.2f}, '
                              f'Status: {self.robot_status}')

    def send_command(self, command):
        """
        Method to send a command to the robot from within the agent.
        """
        self.current_command = command.upper()

    def set_velocity(self, linear, angular):
        """
        Method to directly set the robot's velocity.
        """
        self.linear_velocity = max(-self.max_linear_speed,
                                 min(self.max_linear_speed, linear))
        self.angular_velocity = max(-self.max_angular_speed,
                                  min(self.max_angular_speed, angular))


def main(args=None):
    rclpy.init(args=args)

    control_agent = ControlPublisherAgent()

    # Example: Start with moving forward
    control_agent.send_command("MOVE_FORWARD")

    try:
        rclpy.spin(control_agent)
    except KeyboardInterrupt:
        control_agent.get_logger().info('Control agent interrupted by user')
    finally:
        # Stop the robot before shutting down
        stop_msg = Twist()
        stop_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
        control_agent.cmd_vel_publisher.publish(stop_msg)

        # Destroy the node explicitly
        control_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
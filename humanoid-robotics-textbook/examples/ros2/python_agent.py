#!/usr/bin/env python3

"""
Advanced Python Agent for ROS 2
This agent demonstrates how to integrate Python-based AI agents with ROS 2 systems.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32
from std_msgs.msg import String
from collections import deque
import numpy as np


class PythonAgent(Node):
    """
    An advanced Python-based AI agent that can interact with ROS 2 systems.
    This agent subscribes to multiple sensor data streams and publishes control commands.
    """

    def __init__(self):
        super().__init__('python_agent')

        # Create subscribers for different sensor data
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)

        self.camera_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10)

        # Create publisher for control commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create publisher for agent status
        self.status_publisher = self.create_publisher(Float32, '/agent_status', 10)

        # Create publisher for agent decisions (for debugging/monitoring)
        self.decision_publisher = self.create_publisher(String, '/agent_decision', 10)

        # Timer for agent behavior
        timer_period = 0.1  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.agent_behavior)

        # Agent state
        self.laser_data = None
        self.camera_data = None
        self.laser_buffer = deque(maxlen=5)  # Keep last 5 laser scans for smoothing
        self.safe_distance = 1.0  # meters
        self.linear_speed = 0.3
        self.angular_speed = 0.5

        # Agent parameters that could be configured via ROS parameters
        self.declare_parameter('safe_distance', self.safe_distance)
        self.declare_parameter('linear_speed', self.linear_speed)
        self.declare_parameter('angular_speed', self.angular_speed)

        self.safe_distance = self.get_parameter('safe_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value

        self.get_logger().info(f'Python Agent initialized with safe_distance={self.safe_distance}, '
                              f'linear_speed={self.linear_speed}, angular_speed={self.angular_speed}')

    def laser_callback(self, msg):
        """Callback function for handling laser scan data."""
        self.laser_data = msg
        self.laser_buffer.append(msg)
        self.get_logger().info(f'Received laser scan with {len(msg.ranges)} ranges')

    def camera_callback(self, msg):
        """Callback function for handling camera image data."""
        self.camera_data = msg
        self.get_logger().info(f'Received camera image: {msg.width}x{msg.height}')

    def process_laser_data(self):
        """
        Process laser data to detect obstacles and extract features for decision making.
        Returns minimum distance in front of the robot.
        """
        if self.laser_data is None or not self.laser_data.ranges:
            return float('inf')

        # Get distances in front of the robot (narrow field of view)
        ranges = self.laser_data.ranges
        # Get the front 30 degrees (15 degrees left and right of center)
        center_idx = len(ranges) // 2
        front_range_start = max(0, center_idx - 15)
        front_range_end = min(len(ranges), center_idx + 15)

        front_ranges = ranges[front_range_start:front_range_end]
        front_ranges = [r for r in front_ranges if r > 0 and r < float('inf')]

        if not front_ranges:
            return float('inf')

        min_front_distance = min(front_ranges) if front_ranges else float('inf')
        return min_front_distance

    def agent_behavior(self):
        """Main agent behavior function that runs periodically."""
        cmd_msg = Twist()
        decision_msg = String()

        # Get sensor data
        obstacle_distance = self.process_laser_data()

        # Make decisions based on sensor data
        if obstacle_distance == float('inf'):
            # No valid sensor data, stop
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            decision_msg.data = "NO_SENSOR_DATA"
            self.get_logger().warn('No valid sensor data available')
        elif obstacle_distance < self.safe_distance:
            # Obstacle detected, implement obstacle avoidance
            cmd_msg.linear.x = 0.0  # Stop forward motion
            # Turn away from obstacle - for now, simple turn left
            # In a more advanced agent, we might check left vs right distances
            cmd_msg.angular.z = self.angular_speed
            decision_msg.data = f"OBSTACLE_AVOIDANCE: distance={obstacle_distance:.2f}m"
            self.get_logger().info(f'Obstacle detected! Distance: {obstacle_distance:.2f}m, turning')
        else:
            # Path is clear, move forward
            cmd_msg.linear.x = self.linear_speed
            cmd_msg.angular.z = 0.0  # No rotation
            decision_msg.data = f"MOVING_FORWARD: distance={obstacle_distance:.2f}m"
            self.get_logger().info(f'Path clear. Distance: {obstacle_distance:.2f}m, moving forward')

        # Publish control command
        self.cmd_publisher.publish(cmd_msg)

        # Publish agent decision for monitoring/debugging
        self.decision_publisher.publish(decision_msg)

        # Publish status (0 = stopped, 1 = moving forward, 2 = turning)
        status_msg = Float32()
        if cmd_msg.linear.x > 0:
            status_msg.data = 1.0  # Moving forward
        elif cmd_msg.angular.z != 0:
            status_msg.data = 2.0  # Turning
        else:
            status_msg.data = 0.0  # Stopped
        self.status_publisher.publish(status_msg)

    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        self.get_logger().info('Shutting down Python Agent...')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    python_agent = PythonAgent()

    try:
        rclpy.spin(python_agent)
    except KeyboardInterrupt:
        python_agent.get_logger().info('Agent interrupted by user')
    finally:
        # Destroy the node explicitly
        python_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
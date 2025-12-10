#!/usr/bin/env python3

"""
Sensor Subscriber Agent for ROS 2
This agent demonstrates how to create a Python-based agent that subscribes to
sensor data and processes it for use by other components.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, JointState
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
from collections import deque
import numpy as np
import time


class SensorSubscriberAgent(Node):
    """
    A Python-based agent that subscribes to multiple sensor data streams
    and processes them for use by other components in the system.
    """

    def __init__(self):
        super().__init__('sensor_subscriber_agent')

        # Create subscribers for different sensor types
        self.laser_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)

        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)

        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)

        # Publisher for processed sensor data
        self.processed_data_publisher = self.create_publisher(String, '/processed_sensor_data', 10)

        # Publisher for obstacle detection
        self.obstacle_publisher = self.create_publisher(Float32, '/obstacle_distance', 10)

        # Timer for sensor processing
        timer_period = 0.05  # seconds (20 Hz)
        self.timer = self.create_timer(timer_period, self.process_sensors)

        # Agent state
        self.laser_data = None
        self.imu_data = None
        self.joint_data = None

        # Data buffers for temporal processing
        self.laser_buffer = deque(maxlen=10)
        self.imu_buffer = deque(maxlen=10)

        # Sensor processing parameters
        self.declare_parameter('obstacle_threshold', 1.0)
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value

        self.get_logger().info('Sensor Subscriber Agent initialized')

    def laser_callback(self, msg):
        """Callback function for handling laser scan data."""
        self.laser_data = msg
        self.laser_buffer.append(msg)
        self.get_logger().info(f'Received laser scan with {len(msg.ranges)} ranges')

    def imu_callback(self, msg):
        """Callback function for handling IMU data."""
        self.imu_data = msg
        self.imu_buffer.append(msg)
        self.get_logger().info(f'Received IMU data: orientation=({msg.orientation.x:.2f}, {msg.orientation.y:.2f}, {msg.orientation.z:.2f}, {msg.orientation.w:.2f})')

    def joint_callback(self, msg):
        """Callback function for handling joint state data."""
        self.joint_data = msg
        self.get_logger().info(f'Received joint states for {len(msg.name)} joints')

    def process_laser_data(self):
        """
        Process laser data to extract meaningful information.
        Returns a dictionary with processed laser information.
        """
        if self.laser_data is None or not self.laser_data.ranges:
            return None

        ranges = np.array(self.laser_data.ranges)
        valid_ranges = ranges[(ranges > 0) & (ranges < float('inf'))]

        if len(valid_ranges) == 0:
            return {
                'min_distance': float('inf'),
                'max_distance': 0,
                'avg_distance': float('inf'),
                'obstacle_detected': False,
                'obstacle_distance': float('inf')
            }

        # Calculate statistics
        min_distance = float(np.min(valid_ranges))
        max_distance = float(np.max(valid_ranges))
        avg_distance = float(np.mean(valid_ranges))

        # Check for obstacles
        obstacle_distance = min_distance
        obstacle_detected = min_distance < self.obstacle_threshold

        return {
            'min_distance': min_distance,
            'max_distance': max_distance,
            'avg_distance': avg_distance,
            'obstacle_detected': obstacle_detected,
            'obstacle_distance': obstacle_distance
        }

    def process_imu_data(self):
        """
        Process IMU data to extract meaningful information.
        Returns a dictionary with processed IMU information.
        """
        if self.imu_data is None:
            return None

        # Extract orientation
        orientation = {
            'x': self.imu_data.orientation.x,
            'y': self.imu_data.orientation.y,
            'z': self.imu_data.orientation.z,
            'w': self.imu_data.orientation.w
        }

        # Extract angular velocity
        angular_velocity = {
            'x': self.imu_data.angular_velocity.x,
            'y': self.imu_data.angular_velocity.y,
            'z': self.imu_data.angular_velocity.z
        }

        # Extract linear acceleration
        linear_acceleration = {
            'x': self.imu_data.linear_acceleration.x,
            'y': self.imu_data.linear_acceleration.y,
            'z': self.imu_data.linear_acceleration.z
        }

        return {
            'orientation': orientation,
            'angular_velocity': angular_velocity,
            'linear_acceleration': linear_acceleration
        }

    def process_joint_data(self):
        """
        Process joint state data to extract meaningful information.
        Returns a dictionary with processed joint information.
        """
        if self.joint_data is None:
            return None

        joint_info = {}
        for i, name in enumerate(self.joint_data.name):
            if i < len(self.joint_data.position) and i < len(self.joint_data.velocity):
                joint_info[name] = {
                    'position': self.joint_data.position[i] if i < len(self.joint_data.position) else 0.0,
                    'velocity': self.joint_data.velocity[i] if i < len(self.joint_data.velocity) else 0.0,
                    'effort': self.joint_data.effort[i] if i < len(self.joint_data.effort) else 0.0
                }

        return joint_info

    def process_sensors(self):
        """
        Main sensor processing function that runs periodically.
        """
        # Process laser data
        laser_info = self.process_laser_data()

        # Process IMU data
        imu_info = self.process_imu_data()

        # Process joint data
        joint_info = self.process_joint_data()

        # Create a summary of processed sensor data
        processed_data = {
            'timestamp': time.time(),
            'laser': laser_info,
            'imu': imu_info,
            'joints': joint_info
        }

        # Publish obstacle information if detected
        if laser_info and laser_info['obstacle_detected']:
            obstacle_msg = Float32()
            obstacle_msg.data = laser_info['obstacle_distance']
            self.obstacle_publisher.publish(obstacle_msg)

        # Publish processed data as a string message
        # In a real system, you might want to create a custom message type
        processed_str = f"Sensor Data: Laser min={laser_info['min_distance'] if laser_info else 'N/A'}m, "
        processed_str += f"Obstacle detected: {laser_info['obstacle_detected'] if laser_info else 'N/A'}, "
        processed_str += f"Joints: {len(joint_info) if joint_info else 0} active"

        processed_msg = String()
        processed_msg.data = processed_str
        self.processed_data_publisher.publish(processed_msg)

        # Log the processed data
        self.get_logger().info(processed_str)

    def get_sensor_status(self):
        """
        Return a summary of sensor status for monitoring purposes.
        """
        status = {
            'laser_received': self.laser_data is not None,
            'imu_received': self.imu_data is not None,
            'joints_received': self.joint_data is not None,
            'timestamp': self.get_clock().now().to_msg()
        }
        return status


def main(args=None):
    rclpy.init(args=args)

    sensor_agent = SensorSubscriberAgent()

    try:
        rclpy.spin(sensor_agent)
    except KeyboardInterrupt:
        sensor_agent.get_logger().info('Sensor agent interrupted by user')
    finally:
        # Destroy the node explicitly
        sensor_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
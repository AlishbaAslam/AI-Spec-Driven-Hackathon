---
title: Chapter 2 - Integrating Python Agents with ROS 2 via rclpy
sidebar_position: 1
---

# Chapter 2 - Integrating Python Agents with ROS 2 via rclpy

This chapter focuses on integrating Python-based AI agents with ROS 2 systems using the rclpy library. You'll learn how to bridge the gap between AI software and robotics hardware.

## Learning Objectives

By the end of this chapter, you will be able to:
- Use rclpy to create Python nodes that interact with ROS 2
- Subscribe to sensor data and publish control commands from Python agents
- Implement proper error handling in Python ROS 2 nodes
- Design agent behaviors that respond to robot sensor inputs

## Table of Contents
1. [Introduction to rclpy](#introduction-to-rclpy)
2. [Python Agents in ROS 2](#python-agents-in-ros-2)
3. [Sensor Data Integration](#sensor-data-integration)
4. [Control Command Implementation](#control-command-implementation)
5. [Error Handling and Robustness](#error-handling-and-robustness)
6. [Hands-on Exercises](#hands-on-exercises)

## Introduction to rclpy

rclpy is the Python client library for ROS 2. It provides Python APIs that are conceptually similar to rcl (the C client library) and the underlying ROS graph.

### Key Features of rclpy

- **Node Creation**: Simple API for creating ROS 2 nodes in Python
- **Topic Communication**: Publisher and subscriber interfaces
- **Service Communication**: Client and server interfaces
- **Parameter Management**: Dynamic parameter handling
- **Lifecycle Management**: Node lifecycle support
- **Time and Timers**: Time-based operations and periodic callbacks

### Basic rclpy Structure

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)

    # Create node instance
    node = YourNodeClass()

    # Spin to process callbacks
    rclpy.spin(node)

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
```

## Python Agents in ROS 2

Python agents in the ROS 2 context are software components written in Python that perform intelligent behaviors and communicate with other ROS 2 nodes. These agents can process sensor data, make decisions, and send control commands to robots.

### Agent Architecture

A typical Python agent in ROS 2 consists of:

1. **ROS Interface Layer**: Handles communication with ROS 2
2. **Processing Layer**: Implements the agent's logic and decision-making
3. **Control Layer**: Translates decisions into ROS 2 messages

### Example Agent Structure

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class PythonAgent(Node):
    def __init__(self):
        super().__init__('python_agent')

        # Create subscriber for sensor data
        self.sensor_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.sensor_callback,
            10
        )

        # Create publisher for control commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Agent state and parameters
        self.agent_state = "idle"
        self.safety_distance = 0.5  # meters

    def sensor_callback(self, msg):
        # Process sensor data and make decisions
        min_distance = min(msg.ranges)

        if min_distance < self.safety_distance:
            # Obstacle detected, stop the robot
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
        else:
            # Safe to move, continue with behavior
            self.execute_behavior()

    def execute_behavior(self):
        # Implement agent's main behavior
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd.angular.z = 0.0  # No rotation
        self.cmd_pub.publish(cmd)
```

## Sensor Data Integration

Python agents need to effectively process various types of sensor data from the robot. Common sensor types include:

- **Laser Scanners**: For obstacle detection and mapping
- **Cameras**: For visual processing and object recognition
- **IMU**: For orientation and acceleration data
- **Encoders**: For position and velocity feedback
- **Force/Torque Sensors**: For interaction with the environment

### Handling Different Sensor Types

```python
# Laser scanner integration
from sensor_msgs.msg import LaserScan

def laser_callback(self, msg):
    # Process laser scan data
    ranges = msg.ranges
    # Implement obstacle detection logic
    pass

# Camera integration
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def camera_callback(self, msg):
    # Convert ROS Image to OpenCV format
    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    # Process image with computer vision
    pass

# IMU integration
from sensor_msgs.msg import Imu

def imu_callback(self, msg):
    # Process orientation and acceleration data
    orientation = msg.orientation
    angular_velocity = msg.angular_velocity
    linear_acceleration = msg.linear_acceleration
    pass
```

## Control Command Implementation

Python agents send control commands to actuators through ROS 2 topics. Common control interfaces include:

- **Velocity Control**: Sending Twist messages for differential drive robots
- **Joint Position Control**: Sending JointState messages for manipulators
- **Effort Control**: Sending force/torque commands for compliant control

### Velocity Control Example

```python
from geometry_msgs.msg import Twist

def send_velocity_command(self, linear_vel, angular_vel):
    cmd = Twist()
    cmd.linear.x = linear_vel
    cmd.angular.z = angular_vel
    self.cmd_pub.publish(cmd)
```

### Path Following Behavior

```python
def follow_path(self, path):
    """Follow a predefined path using pure pursuit algorithm"""
    for waypoint in path:
        # Calculate required velocity to reach waypoint
        cmd = self.calculate_pure_pursuit_command(waypoint)
        self.cmd_pub.publish(cmd)
        # Wait for robot to reach waypoint or timeout
        rclpy.spin_once(self, timeout_sec=0.1)
```

## Error Handling and Robustness

Robust Python agents must handle various error conditions gracefully:

- **Communication failures**: Network issues, node disconnections
- **Sensor failures**: Invalid data, sensor timeouts
- **Actuator failures**: Command rejection, hardware faults
- **Environmental changes**: Unexpected obstacles, dynamic environments

### Error Handling Patterns

```python
def safe_sensor_callback(self, msg):
    try:
        # Validate sensor data
        if not self.validate_sensor_data(msg):
            self.get_logger().warning("Invalid sensor data received")
            return

        # Process valid data
        self.process_sensor_data(msg)

    except Exception as e:
        self.get_logger().error(f"Error processing sensor data: {e}")
        # Implement safe behavior
        self.emergency_stop()

def validate_sensor_data(self, msg):
    """Validate sensor message before processing"""
    if len(msg.ranges) == 0:
        return False
    if any(r < 0 for r in msg.ranges if r != float('inf')):
        return False
    return True

def emergency_stop(self):
    """Send emergency stop command"""
    cmd = Twist()
    cmd.linear.x = 0.0
    cmd.angular.z = 0.0
    self.cmd_pub.publish(cmd)
    self.agent_state = "emergency_stopped"
```

## Hands-on Exercises

Complete these exercises to practice Python agent integration with ROS 2:

1. Create a Python agent that subscribes to laser scan data and implements obstacle avoidance
2. Implement a Python agent that follows a simple navigation goal
3. Add error handling to your agent to handle sensor failures gracefully

Continue to the next sections for detailed instructions on each exercise.
---
title: Exercise 2 - Python Agent Integration with ROS 2
sidebar_position: 2
---

# Exercise 2 - Python Agent Integration with ROS 2

In this exercise, you'll create a Python-based AI agent that integrates with ROS 2 systems using the rclpy library. You'll implement sensor data processing and control command publishing to bridge the gap between AI software and robotics hardware.

## Prerequisites

Before starting this exercise, make sure you have:

1. Completed Chapter 1 exercises on basic ROS 2 concepts
2. A working ROS 2 workspace with Python development environment
3. Basic understanding of Python programming

## Exercise Objectives

By the end of this exercise, you will:

1. Create a Python agent that subscribes to sensor data
2. Implement control command publishing from your agent
3. Add proper error handling to your agent
4. Test your agent in a simulated environment

## Part 1: Advanced Python Agent

Building on the basic concepts, create a more sophisticated Python agent that implements advanced behaviors.

### Step 1: Create the Enhanced Python Agent

Create a file named `advanced_python_agent.py`:

```python
#!/usr/bin/env python3
# Advanced Python Agent for ROS 2
# This agent demonstrates advanced integration with ROS 2 systems using rclpy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
import numpy as np


class AdvancedPythonAgent(Node):
    """
    An advanced Python-based AI agent that integrates with ROS 2 systems.
    This agent subscribes to multiple sensor types and publishes complex control commands.
    """

    def __init__(self):
        super().__init__('advanced_python_agent')

        # Create subscribers for various sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Create publishers for control and status
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/agent_status',
            10
        )

        self.goal_reached_pub = self.create_publisher(
            Bool,
            '/goal_reached',
            10
        )

        # Initialize TF listener for localization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Agent state and parameters
        self.agent_state = "idle"
        self.safety_distance = 0.5  # meters
        self.forward_speed = 0.3    # m/s
        self.rotation_speed = 0.5   # rad/s
        self.last_scan_msg = None
        self.current_pose = None
        self.emergency_stop_active = False
        self.goal_position = None
        self.navigation_active = False

        # Timer for behavior execution
        self.timer = self.create_timer(0.1, self.behavior_timer_callback)  # 10Hz

        # Obstacle detection parameters
        self.obstacle_threshold = 0.8  # meters
        self.clear_threshold = 1.2     # meters

        self.get_logger().info('Advanced Python Agent initialized')

    def scan_callback(self, msg):
        """
        Callback function for laser scan data.
        Processes sensor data and makes decisions based on the environment.
        """
        self.last_scan_msg = msg

        if not self.emergency_stop_active:
            self.process_scan_data(msg)

    def odom_callback(self, msg):
        """
        Callback function for odometry data.
        Updates the agent's current position and orientation.
        """
        self.current_pose = msg.pose.pose

    def process_scan_data(self, scan_msg):
        """
        Process laser scan data to detect obstacles and make navigation decisions.
        """
        try:
            # Get valid range measurements
            valid_ranges = [r for r in scan_msg.ranges
                           if r != float('inf') and not math.isnan(r) and r > 0]

            if not valid_ranges:
                self.get_logger().warn('No valid range measurements in scan data')
                return

            # Find minimum distance in front of the robot (approximate)
            front_ranges_idx = len(valid_ranges) // 2 - 10 : len(valid_ranges) // 2 + 10
            if front_ranges_idx[0] < 0:
                front_ranges_idx = (0, min(len(valid_ranges), 20))
            elif front_ranges_idx[1] > len(valid_ranges):
                front_ranges_idx = (max(0, len(valid_ranges) - 20), len(valid_ranges))

            front_ranges = valid_ranges[front_ranges_idx[0]:front_ranges_idx[1]]
            if front_ranges:
                min_front_distance = min(front_ranges) if front_ranges else float('inf')
            else:
                min_front_distance = float('inf')

            # Update agent state based on sensor data
            if min_front_distance < self.safety_distance:
                self.agent_state = "obstacle_close"
                self.get_logger().warn(f'Close obstacle detected: {min_front_distance:.2f}m')
            elif min_front_distance < self.obstacle_threshold:
                self.agent_state = "obstacle_near"
                self.get_logger().info(f'Near obstacle detected: {min_front_distance:.2f}m')
            else:
                self.agent_state = "navigation_clear"

        except Exception as e:
            self.get_logger().error(f'Error processing scan data: {e}')

    def behavior_timer_callback(self):
        """
        Timer callback that executes the agent's behavior logic periodically.
        """
        if self.emergency_stop_active:
            self.send_stop_command()
            return

        if self.last_scan_msg is None:
            self.send_stop_command()
            return

        # Execute behavior based on current state
        if self.navigation_active and self.goal_position:
            self.execute_navigation_to_goal()
        elif self.agent_state == "obstacle_close":
            self.execute_emergency_avoidance()
        elif self.agent_state == "obstacle_near":
            self.execute_caution_behavior()
        elif self.agent_state == "navigation_clear":
            self.execute_normal_behavior()
        else:
            self.send_stop_command()

    def execute_navigation_to_goal(self):
        """
        Execute navigation behavior to reach a specific goal.
        """
        if not self.current_pose:
            self.get_logger().warn('No current pose available for navigation')
            return

        # Calculate direction to goal (simplified - would need real localization)
        cmd = Twist()

        # For this example, we'll just move forward if path is clear
        # In a real implementation, you'd calculate the actual direction to goal
        cmd.linear.x = min(self.forward_speed, 0.2)  # Slower for safety
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

        # Check if goal is reached (simplified)
        # In real implementation, you'd compare current pose to goal pose
        goal_reached_msg = Bool()
        goal_reached_msg.data = False  # Placeholder - implement actual logic
        self.goal_reached_pub.publish(goal_reached_msg)

    def execute_emergency_avoidance(self):
        """
        Execute emergency obstacle avoidance when obstacle is very close.
        """
        cmd = Twist()
        # Stop forward motion and rotate away from obstacle
        cmd.linear.x = 0.0
        cmd.angular.z = self.rotation_speed  # Rotate away

        self.cmd_pub.publish(cmd)

    def execute_caution_behavior(self):
        """
        Execute cautious navigation when obstacles are nearby.
        """
        cmd = Twist()
        # Move slowly forward with potential to turn
        cmd.linear.x = self.forward_speed * 0.3  # 30% speed
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def execute_normal_behavior(self):
        """
        Execute normal navigation behavior when environment is clear.
        """
        cmd = Twist()
        cmd.linear.x = self.forward_speed
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def send_stop_command(self):
        """
        Send a stop command to halt the robot.
        """
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def set_goal(self, x, y):
        """
        Set a navigation goal for the agent.
        """
        self.goal_position = (x, y)
        self.navigation_active = True
        self.get_logger().info(f'Navigation goal set to: ({x}, {y})')

    def cancel_goal(self):
        """
        Cancel the current navigation goal.
        """
        self.navigation_active = False
        self.goal_position = None
        self.get_logger().info('Navigation goal cancelled')

    def emergency_stop(self):
        """
        Activate emergency stop to halt the robot immediately.
        """
        self.emergency_stop_active = True
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        self.send_stop_command()

    def reset_emergency_stop(self):
        """
        Reset emergency stop to resume normal operation.
        """
        self.emergency_stop_active = False
        self.get_logger().info('Emergency stop reset')


def main(args=None):
    """
    Main function to run the advanced Python agent.
    """
    rclpy.init(args=args)

    agent = AdvancedPythonAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Interrupted, shutting down agent')
    except Exception as e:
        agent.get_logger().error(f'Unexpected error: {e}')
    finally:
        agent.send_stop_command()  # Ensure robot stops before shutting down
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 2: Run the Advanced Agent

1. Save the file as `advanced_python_agent.py`
2. Make it executable: `chmod +x advanced_python_agent.py`
3. Run it in your ROS 2 workspace: `python3 advanced_python_agent.py`

## Part 2: Agent with Multiple Sensor Integration

Create a more comprehensive agent that integrates multiple sensor types:

### Step 1: Create Multi-Sensor Agent

Create a file named `multi_sensor_agent.py`:

```python
#!/usr/bin/env python3
# Multi-Sensor Python Agent for ROS 2
# This agent integrates data from multiple sensor types

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, BatteryState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import math


class MultiSensorAgent(Node):
    """
    A Python agent that integrates data from multiple sensor types.
    Demonstrates how to handle different sensor modalities in a single agent.
    """

    def __init__(self):
        super().__init__('multi_sensor_agent')

        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Create subscribers for different sensor types
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.battery_sub = self.create_subscription(
            BatteryState,
            '/battery_state',
            self.battery_callback,
            10
        )

        # Create publisher for control commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/agent_status',
            10
        )

        # Initialize sensor data storage
        self.laser_data = None
        self.camera_data = None
        self.imu_data = None
        self.battery_data = None

        # Agent state variables
        self.battery_level = 100.0
        self.low_battery_threshold = 20.0
        self.orientation = None
        self.linear_acceleration = None

        # Timer for behavior execution
        self.timer = self.create_timer(0.1, self.behavior_timer_callback)

        self.get_logger().info('Multi-Sensor Agent initialized')

    def laser_callback(self, msg):
        """Handle laser scan data."""
        self.laser_data = msg
        self.get_logger().debug(f'Laser scan received: {len(msg.ranges)} points')

    def camera_callback(self, msg):
        """Handle camera image data."""
        try:
            # Convert ROS Image to OpenCV format (if needed)
            # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.camera_data = msg
            self.get_logger().debug(f'Camera image received: {msg.width}x{msg.height}')
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def imu_callback(self, msg):
        """Handle IMU data."""
        self.imu_data = msg
        self.orientation = msg.orientation
        self.linear_acceleration = msg.linear_acceleration
        self.get_logger().debug('IMU data received')

    def battery_callback(self, msg):
        """Handle battery state data."""
        self.battery_data = msg
        self.battery_level = msg.percentage * 100 if msg.percentage else 0.0
        self.get_logger().info(f'Battery level: {self.battery_level:.1f}%')

    def behavior_timer_callback(self):
        """Main behavior logic based on all sensor inputs."""
        # Check battery level first
        if self.battery_level < self.low_battery_threshold:
            self.get_logger().warn(f'Low battery: {self.battery_level:.1f}%, returning to base')
            self.execute_battery_saver_behavior()
            return

        # Process laser data for navigation
        if self.laser_data:
            min_distance = min([r for r in self.laser_data.ranges
                              if r != float('inf') and not math.isnan(r)], default=float('inf'))

            if min_distance < 0.5:
                # Obstacle detected, handle appropriately
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn away from obstacle
                self.cmd_pub.publish(cmd)
                self.get_logger().warn(f'Obstacle at {min_distance:.2f}m, turning')
            else:
                # Clear path, move forward
                cmd = Twist()
                cmd.linear.x = 0.3
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)

        # Publish status
        status_msg = String()
        status_msg.data = f'Battery: {self.battery_level:.1f}%, Laser: {bool(self.laser_data)}, Cam: {bool(self.camera_data)}'
        self.status_pub.publish(status_msg)

    def execute_battery_saver_behavior(self):
        """Execute behavior when battery is low."""
        cmd = Twist()
        cmd.linear.x = -0.2  # Move backward slowly to return to base
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        status_msg = String()
        status_msg.data = f'LOW BATTERY: {self.battery_level:.1f}%, returning to base'
        self.status_pub.publish(status_msg)

    def get_sensor_fusion_data(self):
        """Combine data from multiple sensors for better decision making."""
        fused_data = {
            'obstacle_distance': float('inf'),
            'battery_level': self.battery_level,
            'orientation': self.orientation,
            'acceleration': self.linear_acceleration,
            'has_camera_data': self.camera_data is not None
        }

        if self.laser_data:
            valid_ranges = [r for r in self.laser_data.ranges
                           if r != float('inf') and not math.isnan(r)]
            if valid_ranges:
                fused_data['obstacle_distance'] = min(valid_ranges)

        return fused_data


def main(args=None):
    """Main function for multi-sensor agent."""
    rclpy.init(args=args)

    agent = MultiSensorAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Interrupted, shutting down multi-sensor agent')
    except Exception as e:
        agent.get_logger().error(f'Unexpected error: {e}')
    finally:
        # Send stop command before shutting down
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        agent.cmd_pub.publish(cmd)

        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 2: Run the Multi-Sensor Agent

1. Save the file as `multi_sensor_agent.py`
2. Make it executable: `chmod +x multi_sensor_agent.py`
3. Run it in your ROS 2 workspace: `python3 multi_sensor_agent.py`

## Part 3: Error Handling and Robustness

Create a version with comprehensive error handling:

### Step 1: Create Robust Agent

Create a file named `robust_agent.py`:

```python
#!/usr/bin/env python3
# Robust Python Agent for ROS 2
# This agent demonstrates comprehensive error handling and robustness

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math
import traceback
from enum import Enum


class AgentState(Enum):
    """Enumeration for agent states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"


class RobustAgent(Node):
    """
    A robust Python agent with comprehensive error handling and state management.
    """

    def __init__(self):
        super().__init__('robust_agent')

        # Create QoS profiles for different reliability requirements
        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        best_effort_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Create subscribers with appropriate QoS
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.safe_scan_callback,
            reliable_qos
        )

        # Create publishers
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            reliable_qos
        )

        self.status_pub = self.create_publisher(
            String,
            '/agent_status',
            best_effort_qos
        )

        self.error_pub = self.create_publisher(
            String,
            '/agent_errors',
            reliable_qos
        )

        self.emergency_pub = self.create_publisher(
            Bool,
            '/emergency_stop_signal',
            reliable_qos
        )

        # Initialize state variables
        self.state = AgentState.IDLE
        self.last_scan_msg = None
        self.safety_distance = 0.5
        self.max_recovery_attempts = 3
        self.recovery_count = 0
        self.error_count = 0
        self.last_error_time = None

        # Timer for behavior execution
        self.timer = self.create_timer(0.1, self.safe_behavior_timer_callback)

        self.get_logger().info('Robust Agent initialized with comprehensive error handling')

    def safe_scan_callback(self, msg):
        """Safe callback that handles errors in scan processing."""
        try:
            # Validate the message before processing
            if not self.validate_scan_message(msg):
                self.get_logger().warn('Invalid scan message received')
                return

            self.last_scan_msg = msg
            self.process_scan_data_safely(msg)

        except Exception as e:
            self.handle_error(f'Scan callback error: {e}', traceback.format_exc())
            self.state = AgentState.ERROR

    def validate_scan_message(self, msg):
        """Validate laser scan message integrity."""
        try:
            if not msg.ranges:
                return False

            # Check for reasonable values
            for r in msg.ranges:
                if not (math.isnan(r) or r == float('inf') or (0.01 <= r <= 50.0)):
                    return False

            return True
        except Exception:
            return False

    def process_scan_data_safely(self, scan_msg):
        """Safely process scan data with error handling."""
        try:
            # Process scan data to detect obstacles
            valid_ranges = [r for r in scan_msg.ranges
                           if r != float('inf') and not math.isnan(r) and r > 0]

            if not valid_ranges:
                self.get_logger().warn('No valid ranges in scan data')
                return

            min_distance = min(valid_ranges) if valid_ranges else float('inf')

            # Update state based on obstacle detection
            if min_distance < self.safety_distance:
                if self.state != AgentState.AVOIDING:
                    self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')
                self.state = AgentState.AVOIDING
            else:
                if self.state == AgentState.AVOIDING:
                    self.get_logger().info('Path clear, resuming navigation')
                if self.state != AgentState.EMERGENCY_STOP:
                    self.state = AgentState.NAVIGATING

        except Exception as e:
            self.handle_error(f'Scan processing error: {e}', traceback.format_exc())

    def safe_behavior_timer_callback(self):
        """Safe timer callback with comprehensive error handling."""
        try:
            # Reset error count if we're in a good state
            if self.state not in [AgentState.ERROR, AgentState.EMERGENCY_STOP]:
                self.error_count = 0

            # Execute behavior based on current state
            if self.state == AgentState.NAVIGATING:
                self.execute_navigation_safely()
            elif self.state == AgentState.AVOIDING:
                self.execute_avoidance_safely()
            elif self.state == AgentState.EMERGENCY_STOP:
                self.execute_emergency_stop_safely()
            elif self.state == AgentState.ERROR:
                self.execute_error_recovery_safely()
            else:
                self.execute_idle_behavior_safely()

            # Publish status
            self.publish_status_safely()

        except Exception as e:
            self.handle_error(f'Behavior timer error: {e}', traceback.format_exc())

    def execute_navigation_safely(self):
        """Safely execute navigation behavior."""
        try:
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
        except Exception as e:
            self.handle_error(f'Navigation execution error: {e}', traceback.format_exc())

    def execute_avoidance_safely(self):
        """Safely execute obstacle avoidance."""
        try:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn to avoid
            self.cmd_pub.publish(cmd)
        except Exception as e:
            self.handle_error(f'Obstacle avoidance error: {e}', traceback.format_exc())

    def execute_emergency_stop_safely(self):
        """Safely execute emergency stop."""
        try:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            # Signal emergency stop
            emergency_msg = Bool()
            emergency_msg.data = True
            self.emergency_pub.publish(emergency_msg)
        except Exception as e:
            self.handle_error(f'Emergency stop error: {e}', traceback.format_exc())

    def execute_error_recovery_safely(self):
        """Safely execute error recovery."""
        try:
            # Stop the robot
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            # Try to recover
            if self.recovery_count < self.max_recovery_attempts:
                self.get_logger().info(f'Attempting recovery ({self.recovery_count + 1}/{self.max_recovery_attempts})')
                self.recovery_count += 1
                # Reset to idle after a delay
                self.get_clock().sleep_for(rclpy.duration.Duration(seconds=2))
                self.state = AgentState.IDLE
            else:
                self.get_logger().error('Max recovery attempts reached, remaining in error state')
        except Exception as e:
            self.handle_error(f'Error recovery error: {e}', traceback.format_exc())

    def execute_idle_behavior_safely(self):
        """Safely execute idle behavior."""
        try:
            # Just stop the robot in idle state
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
        except Exception as e:
            self.handle_error(f'Idle behavior error: {e}', traceback.format_exc())

    def publish_status_safely(self):
        """Safely publish agent status."""
        try:
            status_msg = String()
            status_msg.data = f'State: {self.state.value}, Errors: {self.error_count}, Recovery: {self.recovery_count}'
            self.status_pub.publish(status_msg)
        except Exception as e:
            self.get_logger().error(f'Status publishing error: {e}')

    def handle_error(self, error_msg, traceback_info=None):
        """Handle errors with logging and state management."""
        self.error_count += 1
        self.get_logger().error(f'{error_msg} (Error #{self.error_count})')

        if traceback_info:
            self.get_logger().debug(f'Traceback: {traceback_info}')

        # Publish error message
        try:
            error_status = String()
            error_status.data = f'Error: {error_msg}'
            self.error_pub.publish(error_status)
        except Exception as e:
            self.get_logger().error(f'Could not publish error status: {e}')

        # If too many errors, go to emergency stop
        if self.error_count > 10:  # Arbitrary threshold
            self.get_logger().error('Too many errors, activating emergency stop')
            self.state = AgentState.EMERGENCY_STOP

    def activate_emergency_stop(self):
        """Public method to activate emergency stop from outside."""
        self.state = AgentState.EMERGENCY_STOP
        self.get_logger().warn('Emergency stop activated from external command')

    def reset_agent(self):
        """Reset agent to initial state."""
        self.state = AgentState.IDLE
        self.error_count = 0
        self.recovery_count = 0
        self.get_logger().info('Agent reset to initial state')


def main(args=None):
    """Main function for robust agent."""
    rclpy.init(args=args)

    agent = RobustAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Interrupted by user, shutting down')
    except Exception as e:
        agent.get_logger().error(f'Unexpected error in main: {e}')
    finally:
        # Ensure robot stops before shutting down
        try:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            agent.cmd_pub.publish(cmd)
        except Exception as e:
            agent.get_logger().error(f'Error sending final stop command: {e}')

        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 2: Run the Robust Agent

1. Save the file as `robust_agent.py`
2. Make it executable: `chmod +x robust_agent.py`
3. Run it in your ROS 2 workspace: `python3 robust_agent.py`

## Verification Steps

1. **Check that your agents run without errors**:
   ```bash
   python3 advanced_python_agent.py
   ```

2. **Monitor the topics your agent publishes to**:
   ```bash
   ros2 topic echo /agent_status std_msgs/msg/String
   ros2 topic echo /cmd_vel geometry_msgs/msg/Twist
   ```

3. **Verify your agent subscribes to the expected topics**:
   ```bash
   ros2 topic info /scan
   ```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you have the required packages installed:
   ```bash
   pip3 install opencv-python cv-bridge
   ```

2. **Permission errors**: Make sure your Python files are executable:
   ```bash
   chmod +x *.py
   ```

3. **Topic connection issues**: Ensure the required topics exist in your simulation environment.

4. **Memory issues**: If running multiple agents, make sure to stop them properly to prevent memory leaks.

## Summary

In this exercise, you've:

1. Created an advanced Python agent with multiple behaviors
2. Implemented a multi-sensor integration agent
3. Developed a robust agent with comprehensive error handling
4. Learned how to properly structure Python agents for ROS 2

These skills are essential for creating sophisticated AI agents that can interact reliably with robotic systems in real-world environments.
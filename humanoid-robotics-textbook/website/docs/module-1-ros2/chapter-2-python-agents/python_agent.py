#!/usr/bin/env python3
# Python Agent for ROS 2
# This agent demonstrates how to integrate Python-based AI agents with ROS 2 systems using rclpy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math


class PythonAgent(Node):
    """
    A Python-based AI agent that integrates with ROS 2 systems.
    This agent subscribes to sensor data and publishes control commands.
    """

    def __init__(self):
        super().__init__('python_agent')

        # Create subscriber for laser scan data (sensor input)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Create publisher for velocity commands (control output)
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Create publisher for status updates
        self.status_pub = self.create_publisher(
            String,
            '/agent_status',
            10
        )

        # Agent state and parameters
        self.agent_state = "idle"
        self.safety_distance = 0.5  # meters
        self.forward_speed = 0.3    # m/s
        self.rotation_speed = 0.5   # rad/s
        self.last_scan_msg = None
        self.emergency_stop_active = False

        # Timer for periodic behavior execution
        self.timer = self.create_timer(0.1, self.behavior_timer_callback)  # 10Hz

        self.get_logger().info('Python Agent initialized and ready to operate')

    def scan_callback(self, msg):
        """
        Callback function for laser scan data.
        Processes sensor data and makes decisions based on the environment.
        """
        self.last_scan_msg = msg
        self.get_logger().debug(f'Received scan with {len(msg.ranges)} range measurements')

        # Process the scan data for obstacle detection
        if not self.emergency_stop_active:
            self.process_scan_data(msg)

    def process_scan_data(self, scan_msg):
        """
        Process laser scan data to detect obstacles and make navigation decisions.
        """
        try:
            # Get valid range measurements (not infinite and not invalid)
            valid_ranges = [r for r in scan_msg.ranges if r != float('inf') and not math.isnan(r) and r > 0]

            if not valid_ranges:
                self.get_logger().warn('No valid range measurements in scan data')
                return

            # Find minimum distance
            min_distance = min(valid_ranges) if valid_ranges else float('inf')

            # Update agent state based on sensor data
            if min_distance < self.safety_distance:
                self.agent_state = "obstacle_detected"
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, triggering avoidance')
            else:
                self.agent_state = "safe_to_proceed"

        except Exception as e:
            self.get_logger().error(f'Error processing scan data: {e}')

    def behavior_timer_callback(self):
        """
        Timer callback that executes the agent's behavior logic periodically.
        """
        if self.emergency_stop_active:
            # Only send stop command if emergency stop is active
            self.send_stop_command()
            return

        if self.last_scan_msg is None:
            # No sensor data yet, send stop command
            self.send_stop_command()
            return

        # Execute behavior based on current state
        if self.agent_state == "obstacle_detected":
            self.execute_avoidance_behavior()
        elif self.agent_state == "safe_to_proceed":
            self.execute_navigation_behavior()
        else:
            self.send_stop_command()

    def execute_navigation_behavior(self):
        """
        Execute normal navigation behavior when environment is clear.
        """
        cmd = Twist()

        # Move forward at a safe speed
        cmd.linear.x = self.forward_speed
        cmd.angular.z = 0.0  # No rotation

        self.cmd_pub.publish(cmd)
        self.get_logger().info(f'Navigating forward: linear={cmd.linear.x}, angular={cmd.angular.z}')

        # Publish status
        status_msg = String()
        status_msg.data = f"State: {self.agent_state}, Navigating forward"
        self.status_pub.publish(status_msg)

    def execute_avoidance_behavior(self):
        """
        Execute obstacle avoidance behavior when obstacle is detected.
        """
        cmd = Twist()

        # Stop linear motion and rotate to avoid obstacle
        cmd.linear.x = 0.0
        cmd.angular.z = self.rotation_speed  # Rotate to avoid obstacle

        self.cmd_pub.publish(cmd)
        self.get_logger().info(f'Executing avoidance: linear={cmd.linear.x}, angular={cmd.angular.z}')

        # Publish status
        status_msg = String()
        status_msg.data = f"State: {self.agent_state}, Avoiding obstacle"
        self.status_pub.publish(status_msg)

    def send_stop_command(self):
        """
        Send a stop command to halt the robot.
        """
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

        # Publish status
        status_msg = String()
        status_msg.data = f"State: stopped, Emergency: {self.emergency_stop_active}"
        self.status_pub.publish(status_msg)

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
        self.get_logger().info('Emergency stop reset, resuming operation')


def main(args=None):
    """
    Main function to run the Python agent.
    Initializes ROS 2, creates the agent node, and spins to process callbacks.
    """
    rclpy.init(args=args)

    agent = PythonAgent()

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
#!/usr/bin/env python3
# Control Publisher for Python Agent
# This module handles publishing control commands from the Python agent

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
import math


class ControlPublisher(Node):
    """
    A specialized node that publishes control commands from the Python agent
    to various actuators and robot interfaces.
    """

    def __init__(self):
        super().__init__('control_publisher')

        # Create publishers for different control interfaces
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        self.control_status_pub = self.create_publisher(
            String,
            '/control_status',
            10
        )

        # Timer for periodic control updates
        self.control_timer = self.create_timer(0.05, self.control_timer_callback)  # 20Hz

        # Agent control parameters
        self.linear_velocity = 0.0  # m/s
        self.angular_velocity = 0.0  # rad/s
        self.joint_positions = {}  # Joint name to position mapping
        self.last_command_time = self.get_clock().now()
        self.emergency_stop = False

        # Robot-specific parameters
        self.max_linear_speed = 1.0  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.max_joint_speed = 2.0  # rad/s

        # Initialize joint states
        self.initialize_joint_states()

        self.get_logger().info('Control Publisher initialized')

    def initialize_joint_states(self):
        """
        Initialize joint positions for the robot.
        """
        # Example joint names for a simple manipulator or humanoid
        joint_names = ['joint1', 'joint2', 'joint3', 'left_wheel', 'right_wheel']
        for joint_name in joint_names:
            self.joint_positions[joint_name] = 0.0

    def send_velocity_command(self, linear_x, angular_z):
        """
        Send velocity command to the robot.
        """
        if self.emergency_stop:
            # Override with stop command if emergency stop is active
            linear_x = 0.0
            angular_z = 0.0

        # Clamp velocities to safe limits
        linear_x = max(min(linear_x, self.max_linear_speed), -self.max_linear_speed)
        angular_z = max(min(angular_z, self.max_angular_speed), -self.max_angular_speed)

        # Create and publish Twist message
        cmd = Twist()
        cmd.linear = Vector3(x=linear_x, y=0.0, z=0.0)
        cmd.angular = Vector3(x=0.0, y=0.0, z=angular_z)

        self.cmd_vel_pub.publish(cmd)
        self.linear_velocity = linear_x
        self.angular_velocity = angular_z
        self.last_command_time = self.get_clock().now()

        self.get_logger().debug(f'Published velocity command: linear={linear_x}, angular={angular_z}')

        # Publish status
        status_msg = String()
        status_msg.data = f"Velocity cmd: {linear_x:.2f}m/s, {angular_z:.2f}rad/s"
        self.control_status_pub.publish(status_msg)

    def send_joint_commands(self, joint_commands):
        """
        Send joint position commands to the robot.
        joint_commands: dictionary mapping joint names to target positions
        """
        # Update internal joint positions
        for joint_name, position in joint_commands.items():
            if joint_name in self.joint_positions:
                self.joint_positions[joint_name] = position

        # Create and publish JointState message
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_link'

        for joint_name, position in self.joint_positions.items():
            joint_state.name.append(joint_name)
            joint_state.position.append(position)
            # Add zero velocity and effort for simplicity
            joint_state.velocity.append(0.0)
            joint_state.effort.append(0.0)

        self.joint_state_pub.publish(joint_state)

        self.get_logger().debug(f'Published joint states for {len(joint_state.name)} joints')

    def send_stop_command(self):
        """
        Send immediate stop command to halt all robot motion.
        """
        self.send_velocity_command(0.0, 0.0)

    def execute_navigation_command(self, target_x, target_y, current_x, current_y, current_theta):
        """
        Execute navigation command to move towards a target position.
        This is a simple proportional controller example.
        """
        # Calculate error to target
        dx = target_x - current_x
        dy = target_y - current_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)

        # Calculate angle difference
        angle_diff = target_angle - current_theta
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Simple proportional control
        linear_vel = min(0.5 * distance_to_target, 0.3)  # Max 0.3 m/s
        angular_vel = 2.0 * angle_diff  # Gain of 2.0

        # If very close to target, slow down
        if distance_to_target < 0.2:
            linear_vel *= 0.5

        self.send_velocity_command(linear_vel, angular_vel)

        return distance_to_target < 0.1  # Return True if reached target

    def execute_trajectory_command(self, waypoints):
        """
        Execute a trajectory following command.
        waypoints: list of (x, y) tuples representing the path
        """
        if not waypoints:
            self.send_stop_command()
            return

        # For simplicity, just go to the first waypoint
        # In a real implementation, you would follow the entire path
        target_x, target_y = waypoints[0]
        # This would require current position feedback to work properly
        # For this example, we'll just move forward
        self.send_velocity_command(0.2, 0.0)

    def control_timer_callback(self):
        """
        Timer callback for periodic control updates.
        """
        # Check if we need to send a stop command due to timeout
        time_since_last_cmd = self.get_clock().now() - self.last_command_time
        if time_since_last_cmd.nanoseconds / 1e9 > 1.0:  # 1 second timeout
            self.get_logger().warn('No command received recently, sending stop command')
            self.send_stop_command()

    def enable_emergency_stop(self):
        """
        Enable emergency stop to halt all robot motion immediately.
        """
        self.emergency_stop = True
        self.send_stop_command()
        self.get_logger().warn('Emergency stop enabled')

    def disable_emergency_stop(self):
        """
        Disable emergency stop to allow normal operation.
        """
        self.emergency_stop = False
        self.get_logger().info('Emergency stop disabled')

    def is_emergency_stop_active(self):
        """
        Check if emergency stop is currently active.
        """
        return self.emergency_stop


class AgentController:
    """
    High-level controller that combines sensor processing and control publishing.
    This is the main interface for the Python agent to control the robot.
    """

    def __init__(self, control_publisher_node):
        self.ctrl_pub = control_publisher_node
        self.navigation_active = False
        self.current_target = None

    def move_forward(self, speed=0.3):
        """Move the robot forward at the specified speed."""
        self.ctrl_pub.send_velocity_command(speed, 0.0)

    def turn(self, angular_speed=0.5):
        """Turn the robot at the specified angular speed."""
        self.ctrl_pub.send_velocity_command(0.0, angular_speed)

    def stop(self):
        """Stop all robot motion."""
        self.ctrl_pub.send_stop_command()

    def navigate_to(self, x, y):
        """Navigate to a specific position (this would require localization)."""
        # In a real implementation, this would use current position
        # For this example, we'll just move forward
        self.ctrl_pub.send_velocity_command(0.2, 0.0)
        self.navigation_active = True
        self.current_target = (x, y)

    def set_joint_positions(self, joint_commands):
        """Set specific joint positions."""
        self.ctrl_pub.send_joint_commands(joint_commands)

    def execute_避障_behavior(self, sensor_data):
        """
        Execute obstacle avoidance behavior based on sensor data.
        This is a simplified example - in reality, you'd use the actual sensor data.
        """
        # This would analyze sensor_data to detect obstacles and avoid them
        # For this example, we'll just send a stop command if something is close
        self.ctrl_pub.send_velocity_command(0.0, 0.3)  # Turn to avoid


def main(args=None):
    """
    Main function to run the control publisher.
    This is typically used in conjunction with the sensor subscriber and agent logic.
    """
    rclpy.init(args=args)

    control_publisher = ControlPublisher()

    try:
        rclpy.spin(control_publisher)
    except KeyboardInterrupt:
        control_publisher.get_logger().info('Interrupted, sending stop command and shutting down')
        control_publisher.send_stop_command()
    except Exception as e:
        control_publisher.get_logger().error(f'Unexpected error: {e}')
    finally:
        control_publisher.send_stop_command()  # Ensure robot stops before shutting down
        control_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
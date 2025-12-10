#!/usr/bin/env python3
# Complete ROS 2 System Integration Example
# This example combines all concepts learned in the three chapters:
# 1. ROS 2 fundamentals (Nodes, Topics, Services)
# 2. Python Agent Integration with rclpy
# 3. URDF modeling concepts

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
import math
import time
from typing import List, Tuple, Optional


class IntegratedRobotSystem(Node):
    """
    A complete ROS 2 system that integrates:
    1. Sensor processing (LIDAR, Odometry)
    2. Python-based AI agent decision making
    3. Robot control based on URDF model
    4. Safety and error handling
    """

    def __init__(self):
        super().__init__('integrated_robot_system')

        # QoS profiles for different reliability requirements
        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        best_effort_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers for sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            reliable_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            reliable_qos
        )

        # Publishers for control and status
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            reliable_qos
        )

        self.status_pub = self.create_publisher(
            String,
            '/system_status',
            best_effort_qos
        )

        self.safety_pub = self.create_publisher(
            Bool,
            '/safety_status',
            reliable_qos
        )

        # System state variables
        self.current_scan: Optional[LaserScan] = None
        self.current_pose: Optional[Pose] = None
        self.current_twist: Optional[Twist] = None
        self.safety_distance = 0.5  # meters
        self.operational_state = "IDLE"  # IDLE, NAVIGATING, AVOIDING, EMERGENCY
        self.emergency_stop_active = False
        self.last_command_time = self.get_clock().now()
        self.error_count = 0
        self.max_errors_before_stop = 10

        # Robot physical properties (based on URDF model)
        self.robot_properties = {
            'max_linear_speed': 0.5,      # m/s
            'max_angular_speed': 1.0,     # rad/s
            'length': 0.5,                # meters
            'width': 0.4,                 # meters
            'wheel_radius': 0.1,          # meters
            'wheel_separation': 0.3       # meters
        }

        # Timer for main behavior loop
        self.behavior_timer = self.create_timer(0.1, self.behavior_callback)  # 10Hz

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.05, self.safety_check_callback)  # 20Hz

        self.get_logger().info('Integrated Robot System initialized and ready')

    def scan_callback(self, msg: LaserScan):
        """Handle laser scan data from sensors."""
        try:
            self.current_scan = msg
            self.process_environment_data()
        except Exception as e:
            self.get_logger().error(f'Error in scan callback: {e}')
            self.error_count += 1

    def odom_callback(self, msg: Odometry):
        """Handle odometry data for localization."""
        try:
            self.current_pose = msg.pose.pose
            self.current_twist = msg.twist.twist
        except Exception as e:
            self.get_logger().error(f'Error in odom callback: {e}')
            self.error_count += 1

    def process_environment_data(self):
        """Process sensor data to understand the environment."""
        if not self.current_scan:
            return

        # Process laser scan to detect obstacles
        ranges = self.current_scan.ranges
        valid_ranges = [r for r in ranges if r != float('inf') and not math.isnan(r) and r > 0]

        if not valid_ranges:
            return

        # Find minimum distance in key directions (front, left, right)
        n = len(ranges)
        front_idx = n // 2
        left_idx = int(n * 0.75)
        right_idx = int(n * 0.25)

        # Ensure indices are within bounds
        front_idx = max(0, min(front_idx, n-1))
        left_idx = max(0, min(left_idx, n-1))
        right_idx = max(0, min(right_idx, n-1))

        front_dist = ranges[front_idx] if ranges[front_idx] != float('inf') else float('inf')
        left_dist = ranges[left_idx] if ranges[left_idx] != float('inf') else float('inf')
        right_dist = ranges[right_idx] if ranges[right_idx] != float('inf') else float('inf')

        # Update operational state based on environment
        if front_dist < self.safety_distance:
            self.operational_state = "AVOIDING"
        elif min(left_dist, right_dist) < self.safety_distance * 1.5:
            self.operational_state = "CAUTIOUS"
        else:
            self.operational_state = "NAVIGATING"

    def behavior_callback(self):
        """Main behavior execution loop."""
        try:
            # Check error count and activate emergency stop if too many errors
            if self.error_count >= self.max_errors_before_stop:
                self.activate_emergency_stop()
                return

            # Execute behavior based on current state
            if self.emergency_stop_active:
                self.execute_emergency_stop()
            elif self.operational_state == "AVOIDING":
                self.execute_obstacle_avoidance()
            elif self.operational_state == "CAUTIOUS":
                self.execute_cautious_navigation()
            elif self.operational_state == "NAVIGATING":
                self.execute_normal_navigation()
            else:
                self.execute_idle_behavior()

            # Publish system status
            self.publish_status()

        except Exception as e:
            self.get_logger().error(f'Error in behavior callback: {e}')
            self.error_count += 1

    def safety_check_callback(self):
        """Regular safety checks."""
        try:
            # Check if we've gone too long without sensor data
            if self.current_scan:
                time_since_scan = self.get_clock().now() - Time(sec=self.current_scan.header.stamp.sec,
                                                               nanosec=self.current_scan.header.stamp.nanosec)
                if time_since_scan.nanoseconds / 1e9 > 1.0:  # 1 second timeout
                    self.get_logger().warn('No scan data received recently')
                    self.operational_state = "IDLE"

            # Check if we've gone too long without sending commands
            time_since_command = self.get_clock().now() - self.last_command_time
            if time_since_command.nanoseconds / 1e9 > 2.0:  # 2 seconds timeout
                self.get_logger().warn('No commands sent recently, stopping robot')
                self.send_stop_command()

        except Exception as e:
            self.get_logger().error(f'Error in safety check: {e}')
            self.error_count += 1

    def execute_emergency_stop(self):
        """Execute emergency stop procedure."""
        self.send_stop_command()
        self.get_logger().warn('EMERGENCY STOP - Robot halted')

    def execute_obstacle_avoidance(self):
        """Execute obstacle avoidance behavior."""
        cmd = Twist()

        # Stop forward motion and turn away from obstacle
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # Turn right to avoid obstacle

        self.send_command(cmd)
        self.get_logger().info('Executing obstacle avoidance')

    def execute_cautious_navigation(self):
        """Execute cautious navigation when obstacles are nearby."""
        cmd = Twist()

        # Move slowly forward with ability to turn
        cmd.linear.x = self.robot_properties['max_linear_speed'] * 0.3  # 30% speed
        cmd.angular.z = 0.0

        self.send_command(cmd)
        self.get_logger().info('Executing cautious navigation')

    def execute_normal_navigation(self):
        """Execute normal navigation behavior."""
        cmd = Twist()

        # Move forward at normal speed
        cmd.linear.x = self.robot_properties['max_linear_speed']
        cmd.angular.z = 0.0

        self.send_command(cmd)
        self.get_logger().info('Executing normal navigation')

    def execute_idle_behavior(self):
        """Execute idle behavior when no specific task is active."""
        self.send_stop_command()
        self.get_logger().info('Robot in idle state')

    def send_command(self, cmd: Twist):
        """Send a command to the robot with safety checks."""
        # Clamp command values to robot limits
        cmd.linear.x = max(-self.robot_properties['max_linear_speed'],
                          min(self.robot_properties['max_linear_speed'], cmd.linear.x))
        cmd.angular.z = max(-self.robot_properties['max_angular_speed'],
                           min(self.robot_properties['max_angular_speed'], cmd.angular.z))

        # Send command
        self.cmd_pub.publish(cmd)
        self.last_command_time = self.get_clock().now()

    def send_stop_command(self):
        """Send immediate stop command."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.send_command(cmd)

    def publish_status(self):
        """Publish system status information."""
        status_msg = String()
        status_msg.data = f"State: {self.operational_state}, Errors: {self.error_count}, " \
                         f"Emergency: {self.emergency_stop_active}"
        self.status_pub.publish(status_msg)

        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = not self.emergency_stop_active
        self.safety_pub.publish(safety_msg)

    def activate_emergency_stop(self):
        """Activate emergency stop for safety."""
        self.emergency_stop_active = True
        self.send_stop_command()
        self.get_logger().error(f'EMERGENCY STOP ACTIVATED due to {self.error_count} errors')

    def reset_system(self):
        """Reset the system to normal operation."""
        self.emergency_stop_active = False
        self.error_count = 0
        self.operational_state = "IDLE"
        self.get_logger().info('System reset to normal operation')


class URDFBasedController:
    """
    A controller that uses URDF model information for more intelligent control.
    This demonstrates how URDF concepts integrate with ROS 2 control systems.
    """

    def __init__(self, robot_description: dict):
        """
        Initialize controller with robot physical properties from URDF.

        Args:
            robot_description: Dictionary containing robot properties parsed from URDF
        """
        self.robot_desc = robot_description
        self.dimensions = robot_description.get('dimensions', {})
        self.joint_limits = robot_description.get('joint_limits', {})
        self.mass_properties = robot_description.get('mass_properties', {})

    def calculate_dynamic_limits(self, current_speed: float) -> Tuple[float, float]:
        """
        Calculate speed limits based on robot dynamics and safety considerations.

        Args:
            current_speed: Current linear speed of the robot

        Returns:
            Tuple of (max_linear_speed, max_angular_speed) based on dynamics
        """
        # Reduce speed as current speed increases for stability
        base_linear_limit = self.robot_desc.get('max_linear_speed', 0.5)
        base_angular_limit = self.robot_desc.get('max_angular_speed', 1.0)

        # Apply dynamic scaling based on current speed
        speed_factor = 1.0 - (abs(current_speed) / base_linear_limit) * 0.3
        speed_factor = max(0.1, speed_factor)  # Don't go below 10% of max speed

        max_linear = base_linear_limit * speed_factor
        max_angular = base_angular_limit * speed_factor

        return max_linear, max_angular

    def validate_command(self, cmd: Twist) -> bool:
        """
        Validate a command against robot physical limits from URDF.

        Args:
            cmd: Twist command to validate

        Returns:
            True if command is valid, False otherwise
        """
        # Check linear speed limits
        if abs(cmd.linear.x) > self.robot_desc.get('max_linear_speed', 0.5):
            return False

        # Check angular speed limits
        if abs(cmd.angular.z) > self.robot_desc.get('max_angular_speed', 1.0):
            return False

        # Additional checks could include:
        # - Acceleration limits based on mass
        # - Torque limits based on joint specifications
        # - Collision avoidance based on robot dimensions

        return True

    def plan_safe_path(self, start_pose: Pose, goal_pose: Pose, obstacles: List[Tuple[float, float]]) -> List[Pose]:
        """
        Plan a safe path considering robot dimensions from URDF.

        Args:
            start_pose: Starting pose of the robot
            goal_pose: Goal pose to reach
            obstacles: List of obstacle positions [(x1, y1), (x2, y2), ...]

        Returns:
            List of poses forming a safe path
        """
        # This is a simplified path planning example
        # In a real system, this would use the robot's dimensions from URDF
        # to ensure the path is collision-free for the actual robot geometry

        robot_width = self.dimensions.get('width', 0.4)
        robot_length = self.dimensions.get('length', 0.5)
        safety_margin = max(robot_width, robot_length) * 0.5  # Add margin around robot

        # Simplified path planning - in reality this would use A*, RRT, or other algorithms
        path = [start_pose]  # In a real implementation, this would be calculated

        # Check if path is collision-free considering robot dimensions
        for obstacle in obstacles:
            # Check if obstacle is too close to any point in path
            # (This is a simplified check)
            pass

        return path


def main(args=None):
    """
    Main function to run the integrated robot system.
    This demonstrates how all the concepts from the three chapters work together.
    """
    rclpy.init(args=args)

    # Robot description based on URDF model
    robot_description = {
        'name': 'simple_humanoid',
        'max_linear_speed': 0.5,
        'max_angular_speed': 1.0,
        'dimensions': {
            'width': 0.4,
            'length': 0.5,
            'height': 0.6
        },
        'joint_limits': {
            'left_shoulder': {'min': -1.57, 'max': 1.57},
            'right_shoulder': {'min': -1.57, 'max': 1.57},
            'left_hip': {'min': -0.79, 'max': 0.79},
            'right_hip': {'min': -0.79, 'max': 0.79}
        },
        'mass_properties': {
            'total_mass': 15.0,
            'center_of_mass': [0.0, 0.0, 0.3]
        }
    }

    # Create the integrated system
    system = IntegratedRobotSystem()

    # Create the URDF-based controller
    controller = URDFBasedController(robot_description)

    system.get_logger().info('Integrated Robot System running...')

    try:
        # Run the system
        rclpy.spin(system)
    except KeyboardInterrupt:
        system.get_logger().info('Interrupted by user')
    except Exception as e:
        system.get_logger().error(f'Unexpected error: {e}')
    finally:
        # Ensure robot stops before shutting down
        system.send_stop_command()
        system.get_logger().info('System stopped safely')
        system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
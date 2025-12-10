---
sidebar_position: 6
title: "Exercise 2: Isaac ROS Implementation"
---

# Exercise 2: Isaac ROS Implementation

## Overview
This hands-on exercise will guide you through implementing a complete Isaac ROS navigation system for a humanoid robot. You'll configure Isaac ROS components, integrate them with Navigation2, and validate the system's performance in both simulation and (optionally) real-world scenarios.

## Learning Objectives
After completing this exercise, you will be able to:
- Install and configure Isaac ROS components for navigation
- Integrate Isaac ROS with Navigation2 stack
- Implement GPU-accelerated VSLAM for humanoid robot localization
- Configure perception-enhanced navigation
- Validate system performance and troubleshoot common issues

## Prerequisites
- Completed Chapter 1 (Isaac Sim) and Chapter 2 theory sections
- NVIDIA GPU with CUDA support (RTX series recommended)
- Isaac Sim environment with humanoid robot model
- Basic ROS 2 and Navigation2 knowledge
- Estimated time: 90-120 minutes

## Exercise Setup

### 1. Environment Preparation
First, ensure your system meets the requirements:

```bash
# Verify Isaac ROS installation
ros2 pkg list | grep isaac_ros

# Verify GPU availability
nvidia-smi

# Verify Isaac Sim availability (if using simulation)
# Launch Isaac Sim and verify it runs properly
```

### 2. Create Exercise Workspace
```bash
# Create a dedicated workspace for this exercise
mkdir -p ~/isaac_ros_exercise_ws/src
cd ~/isaac_ros_exercise_ws

# Source ROS 2
source /opt/ros/humble/setup.bash

# Build the workspace (even though it's empty, this sets up the environment)
colcon build
source install/setup.bash
```

## Exercise Tasks

### Task 1: Isaac ROS Visual SLAM Setup (20 minutes)

#### 1.1 Configure Visual SLAM Parameters
Create a configuration file for Isaac ROS Visual SLAM:

```bash
# Create config directory
mkdir -p ~/isaac_ros_exercise_ws/src/isaac_ros_exercise/config
```

Create `~/isaac_ros_exercise_ws/src/isaac_ros_exercise/config/vslam_config.yaml`:

```yaml
# vslam_config.yaml
isaac_ros_visual_slam:
  ros__parameters:
    # Feature extraction settings
    max_num_features: 1000
    min_num_features: 100
    feature_detector_type: "ORB"
    matcher_type: "BF"

    # Tracking and mapping settings
    enable_mapping: true
    enable_localization_n_mapping: true
    enable_rectification: true
    enable_debug_mode: false

    # Loop closure settings
    enable_loop_closure: true
    loop_closure_threshold: 0.3
    min_loop_closure_translation: 0.5
    min_loop_closure_rotation: 0.1

    # Optical flow settings
    optical_flow_error: 10.0
    optical_flow_window_size: 21
    max_level_pyramid: 3

    # Camera parameters (adjust based on your camera calibration)
    camera_matrix: [320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]
    distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]

    # GPU acceleration settings
    use_gpu: true
    cuda_stream_count: 2
    enable_cuda_graph: true

    # Performance optimization
    processing_frequency: 10.0  # Hz
    max_processing_latency: 0.1  # seconds
```

#### 1.2 Create Launch File
Create `~/isaac_ros_exercise_ws/src/isaac_ros_exercise/launch/vslam_launch.py`:

```python
# vslam_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get config directory
    config_dir = os.path.join(
        get_package_share_directory('isaac_ros_exercise'),
        'config'
    )

    # Isaac ROS Visual SLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='isaac_ros_visual_slam_node',
        parameters=[
            os.path.join(config_dir, 'vslam_config.yaml'),
            {
                # Additional parameters can be added here
                'enable_rectification': True,
                'use_gpu': True,
            }
        ],
        remappings=[
            ('/stereo_camera/left/image_rect_color', '/camera/left/image_rect_color'),
            ('/stereo_camera/right/image_rect_color', '/camera/right/image_rect_color'),
            ('/stereo_camera/left/camera_info', '/camera/left/camera_info'),
            ('/stereo_camera/right/camera_info', '/camera/right/camera_info'),
            ('/visual_slam/tracking/pose_graph/poses', '/vslam/poses'),
            ('/visual_slam/map/landmarks', '/vslam/landmarks'),
        ],
        output='screen'
    )

    return LaunchDescription([
        visual_slam_node,
    ])
```

#### 1.3 Launch and Verify
```bash
# Launch Isaac ROS Visual SLAM
# Note: This assumes you have camera topics available
# You can simulate this with Isaac Sim or use real camera data

# Check if Isaac ROS Visual SLAM node is available
ros2 component types | grep visual_slam

# If using Isaac Sim, launch your simulation environment first
```

### Task 2: Isaac ROS Navigation Integration (30 minutes)

#### 2.1 Create Navigation Configuration
Create `~/isaac_ros_exercise_ws/src/isaac_ros_exercise/config/nav2_isaac_config.yaml`:

```yaml
# nav2_isaac_config.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.5
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::IsaacROSOdometryModel"  # Custom Isaac ROS model
    save_pose_rate: 0.5
    set_initial_pose: false
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "navigate_to_pose_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_to_pose_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_updated_goal_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_assisted_teleop_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Isaac ROS Enhanced Controller
    FollowPath:
      plugin: "nav2_mppi::IsaacROSPathFollowingController"
      max_linear_speed: 0.5
      min_linear_speed: 0.1
      max_angular_speed: 1.0
      min_angular_speed: 0.1
      speed_scaling_mechanism: "aggressive"
      max_allowed_time_to_collision: 1.0
      lookahead_ratio: 1.5
      lookahead_min_dist: 0.3
      lookahead_max_dist: 0.6
      use_interpolation: true
      use_velocity_scaled_lookahead_dist: true
      enable_bounded_velocities: false
      use_rotate_to_heading: false
      rotate_to_heading_angular_vel: 1.8
      max_angular_accel: 3.2
      max_linear_accel: 2.5

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::IsaacROSInflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: pointcloud
        pointcloud:
          topic: "/isaac_ros/pointcloud"  # Isaac ROS point cloud topic
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: True
      robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: pointcloud
        pointcloud:
          topic: "/isaac_ros/pointcloud"  # Isaac ROS point cloud topic
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::IsaacROSInflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 1.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/IsaacROSNavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

#### 2.2 Create Isaac ROS Navigation Launch File
Create `~/isaac_ros_exercise_ws/src/isaac_ros_exercise/launch/isaac_nav2_launch.py`:

```python
# isaac_nav2_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    # Isaac ROS Visual SLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='isaac_ros_visual_slam_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('isaac_ros_exercise'),
                'config',
                'vslam_config.yaml'
            ])
        ],
        remappings=[
            ('/stereo_camera/left/image_rect_color', '/camera/left/image_rect_color'),
            ('/stereo_camera/right/image_rect_color', '/camera/right/image_rect_color'),
            ('/stereo_camera/left/camera_info', '/camera/left/camera_info'),
            ('/stereo_camera/right/camera_info', '/camera/right/camera_info'),
        ],
        output='screen'
    )

    # Navigation2 nodes
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[params_file],
        remappings=[('cmd_vel', 'cmd_vel')]
    )

    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file],
        remappings=[]
    )

    recoveries_server_node = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        parameters=[params_file]
    )

    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file],
        remappings=[
            ('cmd_vel', 'cmd_vel'),
            ('global_costmap', 'global_costmap/costmap_raw'),
            ('local_costmap', 'local_costmap/costmap_raw'),
            ('global_costmap/GlobalCostmap', 'global_costmap/costmap'),
            ('local_costmap/LocalCostmap', 'local_costmap/costmap')
        ]
    )

    lifecycle_nodes = ['controller_server',
                       'planner_server',
                       'recoveries_server',
                       'bt_navigator']

    # Actually do the loading of the lifecycle nodes
    load_composable_nodes = LoadComposableNodes(
        target_container='//{}_container'.format('nav2'),
        composable_node_descriptions=[
            ComposableNode(
                package='nav2_controller',
                plugin='nav2_controller::ControllerServer',
                name='controller_server',
                parameters=[params_file]),
            ComposableNode(
                package='nav2_planner',
                plugin='nav2_planner::PlannerServer',
                name='planner_server',
                parameters=[params_file]),
            ComposableNode(
                package='nav2_recoveries',
                plugin='nav2_recoveries::RecoveryServer',
                name='recoveries_server',
                parameters=[params_file]),
            ComposableNode(
                package='nav2_bt_navigator',
                plugin='nav2_bt_navigator::BtNavigator',
                name='bt_navigator',
                parameters=[params_file]),
        ],
    )

    return LaunchDescription([
        # Isaac ROS Components
        visual_slam_node,

        # Navigation2 Components
        controller_server_node,
        planner_server_node,
        recoveries_server_node,
        bt_navigator_node,
    ])
```

### Task 3: Implementation and Testing (30 minutes)

#### 3.1 Create a Simple Test Node
Create `~/isaac_ros_exercise_ws/src/isaac_ros_exercise/isaac_ros_exercise/test_vslam_node.py`:

```python
#!/usr/bin/env python3
# test_vslam_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose

class IsaacROSVSLAMTester(Node):
    def __init__(self):
        super().__init__('isaac_ros_vslam_tester')

        # Create TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to Isaac ROS VSLAM output
        self.vslam_pose_sub = self.create_subscription(
            Odometry,  # Isaac ROS VSLAM typically outputs Odometry
            '/visual_slam/tracking/pose_graph/poses',  # Actual topic may vary
            self.vslam_pose_callback,
            10
        )

        # Publisher for navigation goals
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        # Publisher for status updates
        self.status_pub = self.create_publisher(
            String,
            '/isaac_ros_tester/status',
            10
        )

        # Timer for periodic checks
        self.check_timer = self.create_timer(5.0, self.periodic_check)

        # Initialize state variables
        self.current_pose = None
        self.pose_count = 0
        self.last_pose_time = self.get_clock().now()

        self.get_logger().info('Isaac ROS VSLAM Tester initialized')

    def vslam_pose_callback(self, msg):
        """Handle Isaac ROS VSLAM pose updates"""
        self.current_pose = msg.pose.pose
        self.pose_count += 1

        # Calculate time since last pose
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_pose_time).nanoseconds / 1e9
        self.last_pose_time = current_time

        # Publish status update
        status_msg = String()
        status_msg.data = f'VSLAM Tracking: Pose #{self.pose_count}, Last update: {time_diff:.2f}s ago'
        self.status_pub.publish(status_msg)

        self.get_logger().info(f'VSLAM Pose updated - Position: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})')

    def periodic_check(self):
        """Periodic system check"""
        if self.pose_count == 0:
            self.get_logger().warn('No VSLAM poses received yet')
            return

        # Calculate pose update frequency
        current_time = self.get_clock().now()
        time_since_first = (current_time - self.start_time).nanoseconds / 1e9 if hasattr(self, 'start_time') else 0

        if not hasattr(self, 'start_time'):
            self.start_time = current_time

        if time_since_first > 0:
            avg_frequency = self.pose_count / time_since_first
            self.get_logger().info(f'Average VSLAM update frequency: {avg_frequency:.2f} Hz')

def main(args=None):
    rclpy.init(args=args)

    node = IsaacROSVSLAMTester()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('VSLAM tester stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 3.2 Create Isaac ROS Package Structure
```bash
# Create the package
cd ~/isaac_ros_exercise_ws/src
ros2 pkg create --build-type ament_python isaac_ros_exercise --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs tf2_ros tf2_geometry_msgs

# Create directories
mkdir -p isaac_ros_exercise/isaac_ros_exercise
mkdir -p isaac_ros_exercise/launch
mkdir -p isaac_ros_exercise/config
```

### Task 4: System Integration Test (20 minutes)

#### 4.1 Create Integration Test Script
Create `~/isaac_ros_exercise_ws/src/isaac_ros_exercise/test/integration_test.py`:

```python
#!/usr/bin/env python3
# integration_test.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import time
import numpy as np

class IsaacROSIntegrationTester(Node):
    def __init__(self):
        super().__init__('isaac_ros_integration_tester')

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.vslam_pose_sub = self.create_subscription(
            Odometry,
            '/visual_slam/tracking/pose_graph/poses',
            self.vslam_pose_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Initialize test state
        self.current_vslam_pose = None
        self.current_odom_pose = None
        self.test_start_time = None
        self.test_completed = False

        # Test parameters
        self.test_duration = 60  # seconds
        self.target_positions = [
            (1.0, 0.0),   # Move 1m forward
            (1.0, 1.0),   # Move 1m right
            (0.0, 1.0),   # Move 1m back
            (0.0, 0.0),   # Return to start
        ]
        self.current_target_idx = 0

        # Timer for test execution
        self.test_timer = self.create_timer(1.0, self.execute_test_step)

        self.get_logger().info('Isaac ROS Integration Tester initialized')

    def vslam_pose_callback(self, msg):
        """Handle VSLAM pose updates"""
        self.current_vslam_pose = msg.pose.pose

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_odom_pose = msg.pose.pose

    def execute_test_step(self):
        """Execute one step of the integration test"""
        if self.test_completed:
            return

        if self.test_start_time is None:
            self.test_start_time = time.time()

        # Check if test duration has elapsed
        elapsed_time = time.time() - self.test_start_time
        if elapsed_time > self.test_duration:
            self.complete_test()
            return

        # Execute navigation to current target
        if self.current_target_idx < len(self.target_positions):
            target = self.target_positions[self.current_target_idx]
            self.navigate_to_target(target)

    def navigate_to_target(self, target_pos):
        """Navigate to target position using available pose information"""
        if self.current_vslam_pose is None:
            self.get_logger().warn('No VSLAM pose available for navigation')
            return

        # Calculate required movement
        current_x = self.current_vslam_pose.position.x
        current_y = self.current_vslam_pose.position.y

        dx = target_pos[0] - current_x
        dy = target_pos[1] - current_y
        distance = np.sqrt(dx*dx + dy*dy)

        # Check if we've reached the target
        if distance < 0.3:  # 30cm tolerance
            self.get_logger().info(f'Reached target {self.current_target_idx}: {target_pos}')
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.target_positions):
                self.get_logger().info('All targets reached successfully!')
                self.complete_test()
            return

        # Generate velocity command to move toward target
        cmd_vel = Twist()

        # Proportional controller for movement
        cmd_vel.linear.x = min(0.3, max(-0.3, 0.5 * dx))  # Forward/backward
        cmd_vel.angular.z = min(0.5, max(-0.5, 1.0 * np.arctan2(dy, dx)))  # Rotation

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        self.get_logger().info(f'Moving to target {self.current_target_idx}: ({target_pos[0]:.2f}, {target_pos[1]:.2f}), Distance: {distance:.2f}m')

    def complete_test(self):
        """Complete the integration test"""
        self.test_completed = True
        self.get_logger().info('Integration test completed')

        # Stop the robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Calculate and report results
        if self.current_vslam_pose:
            final_pos = (self.current_vslam_pose.position.x, self.current_vslam_pose.position.y)
            start_pos = self.target_positions[0] if self.target_positions else (0, 0)
            return_distance = np.sqrt((final_pos[0] - start_pos[0])**2 + (final_pos[1] - start_pos[1])**2)

            self.get_logger().info(f'Final position: {final_pos}')
            self.get_logger().info(f'Distance from start: {return_distance:.2f}m')
            self.get_logger().info('Integration test completed successfully!' if return_distance < 0.5 else 'Integration test completed with significant drift')

def main(args=None):
    rclpy.init(args=args)

    node = IsaacROSIntegrationTester()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Integration tester stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 5: Performance Validation (10 minutes)

#### 5.1 Create Performance Monitoring Script
Create `~/isaac_ros_exercise_ws/src/isaac_ros_exercise/test/performance_monitor.py`:

```python
#!/usr/bin/env python3
# performance_monitor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64, String
import time
from collections import deque
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class IsaacROSPerformanceMonitor(Node):
    def __init__(self):
        super().__init__('isaac_ros_performance_monitor')

        # Parameters
        self.declare_parameter('monitoring_interval', 1.0)
        self.monitoring_interval = self.get_parameter('monitoring_interval').value

        # Publishers for performance metrics
        self.fps_pub = self.create_publisher(Float64, '/isaac_ros/fps', 10)
        self.cpu_pub = self.create_publisher(Float64, '/isaac_ros/cpu_usage', 10)
        self.gpu_pub = self.create_publisher(Float64, '/isaac_ros/gpu_usage', 10)
        self.status_pub = self.create_publisher(String, '/isaac_ros/performance_status', 10)

        # Subscribers to monitor
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )

        # Initialize monitoring variables
        self.frame_count = 0
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = None

        # Timer for performance reporting
        self.perf_timer = self.create_timer(self.monitoring_interval, self.report_performance)

        self.get_logger().info('Isaac ROS Performance Monitor initialized')

    def image_callback(self, msg):
        """Monitor image processing rate"""
        current_time = time.time()

        if self.last_frame_time:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)

        self.last_frame_time = current_time
        self.frame_count += 1

    def report_performance(self):
        """Report performance metrics"""
        # Calculate FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

            # Publish FPS
            fps_msg = Float64()
            fps_msg.data = fps
            self.fps_pub.publish(fps_msg)
        else:
            fps = 0.0

        # Get CPU usage
        cpu_percent = psutil.cpu_percent()
        cpu_msg = Float64()
        cpu_msg.data = cpu_percent
        self.cpu_pub.publish(cpu_msg)

        # Get GPU usage if available
        gpu_percent = 0.0
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100

        gpu_msg = Float64()
        gpu_msg.data = gpu_percent
        self.gpu_pub.publish(gpu_msg)

        # Create status message
        status_msg = String()
        status_msg.data = f'FPS: {fps:.1f}, CPU: {cpu_percent:.1f}%, GPU: {gpu_percent:.1f}%'
        self.status_pub.publish(status_msg)

        # Log performance (only if performance is concerning)
        if fps < 10.0:  # Below real-time requirements
            self.get_logger().warn(f'Low FPS detected: {fps:.1f} (threshold: 10.0)')
        elif fps > 30.0:
            self.get_logger().info(f'Good FPS performance: {fps:.1f}')

        if cpu_percent > 80.0:
            self.get_logger().warn(f'High CPU usage: {cpu_percent:.1f}%')

        if gpu_percent > 85.0:
            self.get_logger().warn(f'High GPU usage: {gpu_percent:.1f}%')

def main(args=None):
    rclpy.init(args=args)

    node = IsaacROSPerformanceMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Performance monitor stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Validation Steps

### 1. System Verification
Run the following commands to verify your implementation:

```bash
# 1. Check that Isaac ROS packages are available
ros2 pkg list | grep isaac_ros

# 2. Check that your exercise package was created
ls -la ~/isaac_ros_exercise_ws/src/isaac_ros_exercise/

# 3. Build your workspace
cd ~/isaac_ros_exercise_ws
colcon build --packages-select isaac_ros_exercise
source install/setup.bash

# 4. Check for compilation errors
# Address any errors before proceeding
```

### 2. Component Testing
Test each component individually:

```bash
# 1. Test the VSLAM configuration
# Check if config file is properly formatted
python3 -c "import yaml; print(yaml.safe_load(open('~/isaac_ros_exercise_ws/src/isaac_ros_exercise/config/vslam_config.yaml')))"

# 2. Test the performance monitor node
ros2 run isaac_ros_exercise performance_monitor

# 3. Test the VSLAM tester node
ros2 run isaac_ros_exercise test_vslam_node
```

### 3. Integration Testing
If using Isaac Sim, you can run a complete integration test:

```bash
# 1. Launch Isaac Sim with a humanoid robot
# 2. Launch your Isaac ROS nodes
# 3. Run the integration test
ros2 run isaac_ros_exercise integration_test
```

## Expected Outcomes

Upon successful completion of this exercise, you should have:

1. **Isaac ROS VSLAM configured** and publishing pose estimates
2. **Navigation2 integrated** with Isaac ROS perception data
3. **Performance monitoring** in place to track system metrics
4. **Integration test** demonstrating complete navigation functionality
5. **Configuration files** properly set up for your specific robot platform

## Troubleshooting

### Common Issues and Solutions:

1. **"Isaac ROS packages not found"**:
   - Verify Isaac ROS is installed: `sudo apt list --installed | grep isaac-ros`
   - Check ROS 2 sourcing: `source /opt/ros/humble/setup.bash`

2. **"GPU not detected"**:
   - Verify NVIDIA drivers: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`

3. **"Low FPS performance"**:
   - Reduce camera resolution
   - Lower VSLAM feature count
   - Check GPU memory usage

4. **"Tracking fails in VSLAM"**:
   - Ensure adequate lighting in environment
   - Verify camera calibration
   - Check for sufficient visual features in scene

## Next Steps

After completing this exercise, you should:
1. Experiment with different Isaac ROS parameters to optimize performance
2. Integrate additional perception components (segmentation, detection)
3. Test the system in more complex environments
4. Explore Isaac Sim for comprehensive testing and validation
5. Prepare for Chapter 3 which will focus on Nav2 integration for humanoid-specific navigation

## Assessment Questions

1. How does Isaac ROS Visual SLAM differ from traditional CPU-based SLAM approaches?
2. What are the key advantages of GPU-accelerated navigation for humanoid robots?
3. How do you validate that Isaac ROS components are properly integrated with Navigation2?
4. What performance metrics are most important for Isaac ROS-based navigation systems?
5. How would you adapt this system for different humanoid robot platforms?

## Challenge Extension

For advanced learners, try extending the system to include:
- Isaac ROS segmentation for semantic navigation
- Dynamic obstacle detection and avoidance
- Multi-session mapping with map persistence
- Integration with Isaac Sim for comprehensive testing
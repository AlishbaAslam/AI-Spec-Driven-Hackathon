---
sidebar_position: 5
title: "Navigation Examples with Isaac ROS"
---

# Navigation Examples with Isaac ROS

## Overview
This section provides practical examples of navigation implementations using Isaac ROS components. Isaac ROS provides GPU-accelerated navigation capabilities that complement the Navigation2 stack in ROS 2, offering enhanced performance for perception-based navigation tasks.

## Learning Objectives
After completing this section, you will be able to:
- Implement GPU-accelerated navigation using Isaac ROS components
- Integrate Isaac ROS navigation with Navigation2 stack
- Configure navigation parameters for optimal performance
- Implement perception-based navigation
- Validate navigation performance and safety

## Prerequisites
- Isaac ROS installed and configured
- Navigation2 stack installed
- VSLAM implementation (from previous section)
- Basic understanding of ROS 2 navigation concepts
- Sensor setup (camera, IMU, etc.)

## Isaac ROS Navigation Architecture

### Navigation Stack Integration
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Isaac ROS      │    │  Navigation2    │    │  Robot          │
│  Perception     │───▶│  Stack         │───▶│  Controllers    │
│  Components     │    │  (Nav2)        │    │                 │
│                 │    │                 │    │                 │
│  • VSLAM        │    │  • Global       │    │  • Motion       │
│  • Segmentation │    │    Planner      │    │    Controller   │
│  • Object       │    │  • Local        │    │  • Trajectory   │
│    Detection    │    │    Planner      │    │    Generator    │
│  • Semantic     │    │  • Behavior     │    │  • Safety       │
│    Mapping      │    │    Tree         │    │    Manager      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### GPU-Accelerated Navigation Components
Isaac ROS provides several navigation-enhancing components:
- **GPU-Accelerated Path Planning**: Faster path computation
- **Perception-Enhanced Mapping**: Better map generation from sensor data
- **Dynamic Obstacle Detection**: Real-time obstacle detection and avoidance
- **Semantic Navigation**: Navigation based on semantic understanding

## Basic Navigation Setup

### 1. Navigation Configuration
```yaml
# nav2_params_isaac.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
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
    default_nav_through_poses_bt_xml: "nav2_bt_xml_v04/navigate_through_poses_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "nav2_bt_xml_v04/navigate_to_pose_w_replanning_and_recovery.xml"
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
    - nav2_is_battery_charging_condition_bt_node
    - nav2_is_toggle_enabled_condition_bt_node

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
      plugin: "nav2_mppi::IsaacROSPathFollowingController"  # Custom Isaac ROS controller
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
        plugin: "nav2_costmap_2d::IsaacROSInflationLayer"  # Isaac ROS enhanced inflation
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: pointcloud
        pointcloud:
          topic: "/intel_realsense_r200_depth/points"  # Isaac ROS point cloud topic
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True

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
          topic: "/intel_realsense_r200_depth/points"  # Isaac ROS point cloud topic
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::IsaacROSInflationLayer"  # Isaac ROS enhanced inflation
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/IsaacROSNavfnPlanner"  # Isaac ROS enhanced planner
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

### 2. Isaac ROS Navigation Launch File
```xml
<!-- isaac_ros_navigation_launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    # Launch arguments
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    # Isaac ROS specific launch arguments
    isaac_ros_vslam_enabled = LaunchConfiguration('isaac_ros_vslam_enabled', default='true')
    isaac_ros_perception_enabled = LaunchConfiguration('isaac_ros_perception_enabled', default='true')

    # Create parameter substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'default_bt_xml_filename': default_bt_xml_filename,
        'map_subscribe_transient_local': map_subscribe_transient_local
    }

    # Create configuration file substitutions
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True
    )

    # Isaac ROS VSLAM Node (if enabled)
    isaac_ros_vslam_node = Node(
        condition=IfCondition(isaac_ros_vslam_enabled),
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='isaac_ros_visual_slam_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('your_navigation_package'),
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

    # Isaac ROS Perception Node (if enabled)
    isaac_ros_perception_node = Node(
        condition=IfCondition(isaac_ros_perception_enabled),
        package='isaac_ros_detectnet',
        executable='isaac_ros_detectnet',
        name='isaac_ros_detectnet',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('your_navigation_package'),
                'config',
                'detectnet_config.yaml'
            ])
        ],
        remappings=[
            ('image_input', '/camera/image_rect_color'),
            ('detections_output', '/isaac_ros/detections'),
        ],
        output='screen'
    )

    # Navigation2 Stack Nodes
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[configured_params],
        remappings=[('cmd_vel', 'cmd_vel')]
    )

    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[configured_params],
        remappings=[]
    )

    recoveries_server_node = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        parameters=[configured_params]
    )

    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[configured_params],
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

    # Launch the actions
    return LaunchDescription([
        # Isaac ROS Nodes
        isaac_ros_vslam_node,
        isaac_ros_perception_node,

        # Navigation2 Stack
        controller_server_node,
        planner_server_node,
        recoveries_server_node,
        bt_navigator_node,
    ])
```

## Perception-Enhanced Navigation

### 1. Semantic Navigation
```python
# semantic_navigation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import numpy as np
from typing import List, Tuple, Dict

class SemanticNavigation(Node):
    def __init__(self):
        super().__init__('semantic_navigation')

        # Subscribe to Isaac ROS segmentation output
        self.segmentation_sub = self.create_subscription(
            Image,  # This would be the actual Isaac ROS segmentation message type
            '/isaac_ros/segmentation/result',
            self.segmentation_callback,
            10
        )

        # Subscribe to 3D point cloud
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # Publisher for semantic costmap
        self.semantic_costmap_pub = self.create_publisher(
            MarkerArray,
            '/semantic_navigation/semantic_markers',
            10
        )

        # Publisher for navigation goals based on semantics
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/semantic_navigation/goal',
            10
        )

        # Initialize semantic navigation
        self.semantic_objects = {}
        self.object_poses = {}
        self.class_colors = {
            1: (1.0, 0.0, 0.0),  # Red for obstacles
            2: (0.0, 1.0, 0.0),  # Green for drivable areas
            3: (0.0, 0.0, 1.0),  # Blue for landmarks
            # Add more semantic classes as needed
        }

        # Timer for semantic map updates
        self.semantic_update_timer = self.create_timer(1.0, self.update_semantic_map)

    def segmentation_callback(self, msg):
        """Process segmentation data from Isaac ROS"""
        # Convert segmentation image to numpy array
        seg_array = self.image_to_numpy(msg)  # Implementation depends on message type

        # Identify semantic objects
        self.identify_semantic_objects(seg_array)

        # Update navigation based on semantics
        self.update_navigation_goals()

    def pointcloud_callback(self, msg):
        """Process 3D point cloud for semantic object positioning"""
        # Process point cloud data to get 3D positions of semantic objects
        # This would typically use PCL or similar libraries
        pass

    def identify_semantic_objects(self, seg_array: np.ndarray):
        """Identify and track semantic objects in the scene"""
        # Get unique semantic classes in the image
        unique_classes = np.unique(seg_array)

        for class_id in unique_classes:
            if class_id in self.class_colors:
                # Find pixels belonging to this class
                mask = (seg_array == class_id)

                # Calculate centroid of the object
                y_coords, x_coords = np.where(mask)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    centroid_x = np.mean(x_coords)
                    centroid_y = np.mean(y_coords)

                    # Store object information
                    self.semantic_objects[class_id] = {
                        'pixel_coords': (centroid_x, centroid_y),
                        'pixel_mask': mask,
                        'area': np.sum(mask)
                    }

    def update_navigation_goals(self):
        """Update navigation goals based on semantic understanding"""
        # Example: Navigate toward blue landmarks (class_id = 3)
        if 3 in self.semantic_objects and self.semantic_objects[3]['area'] > 100:
            # Calculate world position of landmark (simplified)
            landmark_2d = self.semantic_objects[3]['pixel_coords']

            # Convert pixel coordinates to world coordinates
            # This requires camera calibration and depth information
            world_pos = self.pixel_to_world(landmark_2d)

            if world_pos is not None:
                # Create navigation goal
                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.pose.position.x = world_pos[0]
                goal.pose.position.y = world_pos[1]
                goal.pose.position.z = 0.0

                # Publish goal
                self.navigation_goal_pub.publish(goal)

    def pixel_to_world(self, pixel_coords: Tuple[float, float]) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates"""
        # This is a simplified example - actual implementation requires:
        # - Camera intrinsic parameters
        # - Depth information
        # - Robot pose estimation
        px, py = pixel_coords

        # Convert to normalized coordinates
        # Assumes camera parameters are known
        fx, fy = 320.0, 320.0  # Focal lengths
        cx, cy = 320.0, 240.0  # Principal points

        # This is a simplified calculation
        # Real implementation needs depth and proper projection
        world_x = (px - cx) / fx  # Simplified - needs depth for Z
        world_y = (py - cy) / fy

        return (world_x, world_y, 0.0)

    def update_semantic_map(self):
        """Update semantic map visualization"""
        marker_array = MarkerArray()

        for i, (class_id, obj_info) in enumerate(self.semantic_objects.items()):
            if obj_info['area'] > 50:  # Only visualize significant objects
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = 'semantic_objects'
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD

                # Set position (simplified - needs proper 3D positioning)
                marker.pose.position.x = obj_info['pixel_coords'][0] / 100.0  # Scale appropriately
                marker.pose.position.y = obj_info['pixel_coords'][1] / 100.0
                marker.pose.position.z = 0.5
                marker.pose.orientation.w = 1.0

                # Set size
                marker.scale.x = 0.3  # Diameter
                marker.scale.y = 0.3
                marker.scale.z = 1.0  # Height

                # Set color based on semantic class
                color = self.class_colors.get(class_id, (1.0, 1.0, 1.0))
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.7

                marker_array.markers.append(marker)

        # Publish semantic map markers
        self.semantic_costmap_pub.publish(marker_array)
```

### 2. Dynamic Obstacle Avoidance
```python
# dynamic_obstacle_avoidance.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import MarkerArray
import numpy as np
from typing import List, Tuple

class DynamicObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_avoidance')

        # Subscribe to Isaac ROS object detection
        self.detection_sub = self.create_subscription(
            # Replace with actual Isaac ROS detection message type
            # This would typically be a custom message or Detection2DArray
            'isaac_ros/detections',
            self.detection_callback,
            10
        )

        # Subscribe to laser scan for traditional obstacle detection
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Subscribe to robot odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher for visualization
        self.obstacle_viz_pub = self.create_publisher(MarkerArray, '/dynamic_obstacles', 10)

        # Initialize state
        self.robot_pose = None
        self.detected_objects = []
        self.dynamic_obstacles = []
        self.safe_zone_radius = 1.0  # Safe zone around robot

        # Timer for obstacle processing
        self.obstacle_timer = self.create_timer(0.1, self.process_dynamic_obstacles)

    def detection_callback(self, msg):
        """Process object detections from Isaac ROS"""
        # Process detections and update dynamic obstacle list
        new_objects = []

        # Assuming detection message has bounding boxes and velocities
        # This is a conceptual example - actual message type varies
        for detection in msg.detections:  # Replace with actual field name
            obj = {
                'id': detection.id,  # Replace with actual field
                'position': (detection.position.x, detection.position.y),  # Replace with actual fields
                'velocity': (detection.velocity.x, detection.velocity.y),  # Replace with actual fields
                'bbox': detection.bbox,  # Replace with actual field
                'timestamp': self.get_clock().now().seconds_nanoseconds()
            }
            new_objects.append(obj)

        self.detected_objects = new_objects

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Process traditional laser scan for backup obstacle detection
        # This provides redundancy in case visual detection fails
        pass

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.robot_pose = {
            'position': (msg.pose.pose.position.x, msg.pose.pose.position.y),
            'orientation': msg.pose.pose.orientation,
            'velocity': (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        }

    def process_dynamic_obstacles(self):
        """Process detected objects and plan evasive maneuvers"""
        if not self.robot_pose or not self.detected_objects:
            return

        robot_x, robot_y = self.robot_pose['position']

        # Update dynamic obstacle predictions
        for obj in self.detected_objects:
            # Predict future position based on velocity
            dt = 0.5  # Predict 0.5 seconds ahead
            future_x = obj['position'][0] + obj['velocity'][0] * dt
            future_y = obj['position'][1] + obj['velocity'][1] * dt

            # Calculate distance to future position
            dist = np.sqrt((future_x - robot_x)**2 + (future_y - robot_y)**2)

            # Check if obstacle is in collision path
            if dist < self.safe_zone_radius:
                # Plan evasive maneuver
                self.execute_evasive_maneuver(obj)

    def execute_evasive_maneuver(self, obstacle):
        """Execute evasive maneuver based on obstacle position and velocity"""
        if not self.robot_pose:
            return

        robot_x, robot_y = self.robot_pose['position']
        obs_x, obs_y = obstacle['position']
        obs_vx, obs_vy = obstacle['velocity']

        # Calculate relative position and velocity
        rel_x = obs_x - robot_x
        rel_y = obs_y - robot_y
        rel_dist = np.sqrt(rel_x**2 + rel_y**2)

        # Calculate avoidance direction (perpendicular to relative vector)
        # Rotate 90 degrees counter-clockwise
        avoid_x = -rel_y / rel_dist
        avoid_y = rel_x / rel_dist

        # Calculate avoidance magnitude based on proximity
        avoidance_strength = max(0.0, 1.0 - rel_dist / self.safe_zone_radius)

        # Create velocity command
        cmd_vel = Twist()

        # Apply lateral avoidance
        cmd_vel.linear.x = min(0.5, max(-0.5, self.robot_pose['velocity'][0]))  # Maintain forward speed
        cmd_vel.angular.z = avoidance_strength * 0.5 * (avoid_x * obs_vy - avoid_y * obs_vx)  # Proportional to perpendicular component

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        self.get_logger().info(f'Evasive maneuver: angular velocity = {cmd_vel.angular.z:.3f}')
```

## Performance Optimization

### 1. GPU-Accelerated Path Planning
```yaml
# gpu_path_planning_config.yaml
gpu_path_planning:
  ros__parameters:
    # GPU acceleration settings
    use_gpu: true
    cuda_device_id: 0
    enable_tensor_cores: true

    # Path planning optimization
    max_iterations: 1000
    max_planning_time: 2.0  # seconds
    min_distance_between_waypoints: 0.1  # meters

    # Multi-threading
    num_threads: 8
    enable_async_computation: true

    # Memory management
    enable_memory_pool: true
    memory_pool_size: 1073741824  # 1GB

    # Algorithm settings
    algorithm_type: "RRT_STAR_GPU"  # or "DYNASTRUCTURE_GPU"
    optimization_metric: "distance_and_smoothness"

    # Robot constraints (for GPU computation)
    robot_radius: 0.3  # meters
    max_linear_velocity: 0.5  # m/s
    max_angular_velocity: 1.0  # rad/s
```

### 2. Perception-Enhanced Costmap
```yaml
# perception_enhanced_costmap.yaml
perception_enhanced_costmap:
  ros__parameters:
    # Semantic costmap layers
    enable_semantic_layer: true
    semantic_layer:
      plugin: "nav2_costmap_2d::IsaacROSSemanticLayer"
      enabled: true
      observation_sources: semantic_segmentation
      semantic_segmentation:
        topic: "/isaac_ros/segmentation/result"
        max_obstacle_height: 2.0
        clearing: true
        marking: true
        data_type: "Image"
        obstacle_threshold: 100  # Semantic class ID for obstacles
        valid_classes: [1, 2, 3, 4]  # Valid semantic classes

    # Dynamic obstacle layer
    enable_dynamic_layer: true
    dynamic_layer:
      plugin: "nav2_costmap_2d::IsaacROSDynamicLayer"
      enabled: true
      observation_sources: object_detection
      object_detection:
        topic: "/isaac_ros/detections"
        max_obstacle_height: 2.0
        clearing: true
        marking: true
        data_type: "Detection2DArray"
        velocity_threshold: 0.1  # m/s for considering as dynamic

    # Multi-resolution layer
    enable_multiresolution_layer: true
    multiresolution_layer:
      plugin: "nav2_costmap_2d::IsaacROSResolutionLayer"
      enabled: true
      resolution_factors: [1.0, 2.0, 4.0]  # Multiple resolution levels
      weights: [1.0, 0.7, 0.3]  # Weighting for each resolution
```

## Navigation Validation and Testing

### 1. Navigation Performance Validator
```python
# navigation_validator.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import time
from typing import List, Tuple

class NavigationValidator(Node):
    def __init__(self):
        super().__init__('navigation_validator')

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscribers for navigation validation
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Initialize validation metrics
        self.trajectory = []
        self.navigation_start_time = None
        self.navigation_end_time = None
        self.navigation_success = False
        self.goal_reached = False

        # Timer for validation reporting
        self.validation_timer = self.create_timer(1.0, self.validate_navigation)

        # Initialize state
        self.current_pose = None
        self.current_velocity = None
        self.planned_path = None

    def odom_callback(self, msg):
        """Record robot pose for trajectory validation"""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

        # Record trajectory point
        trajectory_point = {
            'timestamp': self.get_clock().now().seconds_nanoseconds(),
            'position': (msg.pose.pose.position.x, msg.pose.pose.position.y),
            'orientation': msg.pose.pose.orientation,
            'linear_velocity': msg.twist.twist.linear.x,
            'angular_velocity': msg.twist.twist.angular.z
        }
        self.trajectory.append(trajectory_point)

    def path_callback(self, msg):
        """Record planned path"""
        self.planned_path = msg.poses

    def cmd_vel_callback(self, msg):
        """Monitor velocity commands"""
        self.current_velocity_cmd = msg

    def validate_navigation(self):
        """Validate navigation performance metrics"""
        if not self.trajectory or not self.current_pose:
            return

        # Calculate current metrics
        current_pos = self.trajectory[-1]['position']
        path_length = self.calculate_path_length(self.trajectory)
        avg_velocity = self.calculate_average_velocity()
        max_curvature = self.calculate_max_curvature()

        # Log metrics
        self.get_logger().info(f"""
Navigation Validation:
- Path Length: {path_length:.2f}m
- Average Velocity: {avg_velocity:.2f} m/s
- Max Curvature: {max_curvature:.3f} rad/m
- Trajectory Points: {len(self.trajectory)}
        """.strip())

    def calculate_path_length(self, trajectory: List) -> float:
        """Calculate total path length from trajectory"""
        if len(trajectory) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1]['position']
            curr_pos = trajectory[i]['position']
            segment_length = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_length += segment_length

        return total_length

    def calculate_average_velocity(self) -> float:
        """Calculate average linear velocity"""
        if not self.trajectory:
            return 0.0

        velocities = [point['linear_velocity'] for point in self.trajectory if 'linear_velocity' in point]
        if velocities:
            return np.mean(np.abs(velocities))
        return 0.0

    def calculate_max_curvature(self) -> float:
        """Calculate maximum curvature in the trajectory"""
        if len(self.trajectory) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(self.trajectory) - 1):
            p1 = np.array(self.trajectory[i-1]['position'])
            p2 = np.array(self.trajectory[i]['position'])
            p3 = np.array(self.trajectory[i+1]['position'])

            # Calculate curvature using three consecutive points
            # Curvature = 2 * |p1 - 2*p2 + p3| / |p1 - p3|^1.5
            numerator = 2 * np.linalg.norm(p1 - 2*p2 + p3)
            denominator = np.power(np.linalg.norm(p1 - p3), 1.5)
            if denominator > 0:
                curvature = numerator / denominator
                curvatures.append(curvature)

        return max(curvatures) if curvatures else 0.0

    def calculate_navigation_efficiency(self) -> dict:
        """Calculate navigation efficiency metrics"""
        if not self.planned_path or not self.trajectory:
            return {}

        # Calculate planned path length
        planned_length = 0.0
        for i in range(1, len(self.planned_path)):
            p1 = self.planned_path[i-1].pose.position
            p2 = self.planned_path[i].pose.position
            segment_length = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            planned_length += segment_length

        # Calculate actual path length
        actual_length = self.calculate_path_length(self.trajectory)

        # Calculate efficiency metrics
        metrics = {
            'planned_path_length': planned_length,
            'actual_path_length': actual_length,
            'path_efficiency_ratio': planned_length / actual_length if actual_length > 0 else 0,
            'excess_distance': actual_length - planned_length if actual_length > planned_length else 0
        }

        return metrics

    def execute_navigation_test(self, goal_x: float, goal_y: float) -> dict:
        """Execute navigation test and return performance metrics"""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation server not available')
            return {}

        # Create goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_x
        goal_msg.pose.pose.position.y = goal_y
        goal_msg.pose.pose.orientation.w = 1.0

        # Record start time
        self.navigation_start_time = time.time()
        self.trajectory = []  # Reset trajectory

        # Send goal
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

        # Wait for completion and return metrics
        # In practice, you'd want to wait asynchronously
        time.sleep(10)  # Wait for navigation to complete (in real implementation, use callbacks)

        # Calculate and return metrics
        efficiency_metrics = self.calculate_navigation_efficiency()
        return efficiency_metrics
```

## Safety and Recovery Behaviors

### 1. Safety Monitor
```python
# safety_monitor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration
import numpy as np

class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Subscribe to sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # Subscribe to velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publisher for safety status
        self.safety_status_pub = self.create_publisher(Bool, '/safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize safety parameters
        self.safety_radius = 0.5  # meters
        self.emergency_braking_distance = 0.3  # meters
        self.current_velocity = Twist()
        self.safety_engaged = False

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.1, self.check_safety)

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Find minimum distance in forward direction
        forward_angles = slice(int(len(msg.ranges)/2 - 30), int(len(msg.ranges)/2 + 30))
        forward_ranges = [r for r in msg.ranges[forward_angles] if not np.isnan(r) and r > 0]

        if forward_ranges:
            min_distance = min(forward_ranges)
            self.check_proximity(min_distance)

    def pointcloud_callback(self, msg):
        """Process point cloud for obstacle detection"""
        # This would typically use PCL or similar for point cloud processing
        # For simplicity, we'll use a conceptual approach
        pass

    def cmd_vel_callback(self, msg):
        """Monitor velocity commands"""
        self.current_velocity = msg

    def check_proximity(self, distance):
        """Check if obstacle is within safety distance"""
        if distance < self.emergency_braking_distance:
            self.trigger_emergency_stop()
        elif distance < self.safety_radius:
            self.reduce_speed_safely()

    def check_safety(self):
        """Main safety check function"""
        # This would integrate multiple sensor inputs
        # For now, we'll use a simplified approach
        pass

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        if not self.safety_engaged:
            self.get_logger().warn('EMERGENCY STOP TRIGGERED!')
            self.safety_engaged = True

            # Publish zero velocity command
            stop_cmd = Twist()
            self.emergency_stop_pub.publish(stop_cmd)

            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = False  # Unsafe
            self.safety_status_pub.publish(safety_msg)

    def reduce_speed_safely(self):
        """Reduce speed when approaching obstacles"""
        if not self.safety_engaged:
            # Reduce linear velocity proportionally to distance
            current_speed = self.current_velocity.linear.x
            reduction_factor = min(1.0, self.emergency_braking_distance / self.safety_radius)
            new_speed = current_speed * reduction_factor

            # Create reduced velocity command
            reduced_cmd = Twist()
            reduced_cmd.linear.x = max(0.0, new_speed)  # Only reduce forward speed
            reduced_cmd.angular.z = self.current_velocity.angular.z * 0.8  # Slightly reduce rotation

            # Publish reduced command
            self.emergency_stop_pub.publish(reduced_cmd)

            self.get_logger().info(f'Speed reduced to {new_speed:.2f} m/s for safety')

    def reset_safety(self):
        """Reset safety system"""
        self.safety_engaged = False
        safety_msg = Bool()
        safety_msg.data = True  # Safe
        self.safety_status_pub.publish(safety_msg)
        self.get_logger().info('Safety system reset')
```

## Integration Examples

### 1. Complete Navigation System Integration
```python
# complete_navigation_system.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import time

class CompleteNavigationSystem(Node):
    def __init__(self):
        super().__init__('complete_navigation_system')

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize all navigation components
        self.initialize_components()

    def initialize_components(self):
        """Initialize all navigation system components"""
        self.get_logger().info('Initializing Isaac ROS Navigation System...')

        # Initialize VSLAM
        # Initialize Perception
        # Initialize Path Planning
        # Initialize Safety Monitor
        # Initialize Semantic Navigation

        self.get_logger().info('Navigation system initialized successfully')

    def navigate_to_waypoints(self, waypoints: list):
        """Navigate through a series of waypoints"""
        for i, (x, y, theta) in enumerate(waypoints):
            self.get_logger().info(f'Navigating to waypoint {i+1}: ({x}, {y})')

            # Check if navigation server is available
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f'Navigation server not available for waypoint {i+1}')
                continue

            # Create navigation goal
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = x
            goal_msg.pose.pose.position.y = y
            goal_msg.pose.pose.position.z = 0.0

            # Convert theta to quaternion
            import math
            siny_cosp = math.sin(theta / 2)
            cosy_cosp = math.cos(theta / 2)
            goal_msg.pose.pose.orientation.z = siny_cosp
            goal_msg.pose.pose.orientation.w = cosy_cosp

            # Send goal
            goal_handle_future = self.nav_client.send_goal_async(goal_msg)

            # Wait for result (in practice, you'd use callbacks)
            rclpy.spin_until_future_complete(self, goal_handle_future)
            goal_handle = goal_handle_future.result()

            if not goal_handle.accepted:
                self.get_logger().error(f'Goal {i+1} was rejected')
                continue

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result

            self.get_logger().info(f'Completed waypoint {i+1}')

    def demo_navigation(self):
        """Demonstrate navigation capabilities"""
        # Define a simple square path
        waypoints = [
            (1.0, 0.0, 0.0),    # Move 1m forward
            (1.0, 1.0, 1.57),   # Move 1m right (with 90 degree turn)
            (0.0, 1.0, 3.14),   # Move 1m back (with 90 degree turn)
            (0.0, 0.0, -1.57),  # Move 1m left (with 90 degree turn) - back to start
        ]

        self.get_logger().info('Starting navigation demonstration...')
        self.navigate_to_waypoints(waypoints)
        self.get_logger().info('Navigation demonstration completed!')

def main(args=None):
    rclpy.init(args=args)

    # Create and run navigation system
    nav_system = CompleteNavigationSystem()

    # Run demonstration
    nav_system.demo_navigation()

    # Spin to keep node alive for any remaining callbacks
    rclpy.spin(nav_system)

    # Cleanup
    nav_system.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Common Navigation Issues

### 1. Common Issues and Solutions
```bash
# Issue: Navigation fails to find path
# Solution: Check costmap inflation and obstacle layers
ros2 param set /global_costmap.global_costmap.inflation_layer.inflation_radius 0.55
ros2 param set /local_costmap.local_costmap.inflation_layer.inflation_radius 0.55

# Issue: Robot oscillates near goal
# Solution: Adjust controller parameters
ros2 param set /controller_server.FollowPath.rotate_to_heading_angular_vel 0.5

# Issue: Navigation too slow
# Solution: Increase controller frequency
ros2 param set /controller_server.controller_frequency 30.0

# Issue: Local planner fails frequently
# Solution: Increase local costmap size
ros2 param set /local_costmap.local_costmap.width 5.0
ros2 param set /local_costmap.local_costmap.height 5.0
```

## Next Steps
After implementing navigation with Isaac ROS:
- Test in various environments
- Tune parameters for your specific robot
- Integrate with perception components for semantic navigation
- Validate safety systems
- Optimize for performance and reliability

## Additional Resources
- [Navigation2 Documentation](https://navigation.ros.org/)
- [Isaac ROS Navigation Components](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_navigation/index.html)
- [ROS 2 Navigation Tutorials](https://navigation.ros.org/tutorials/)

## Exercise
Implement a complete navigation system that:
- Uses Isaac ROS VSLAM for localization
- Incorporates semantic navigation based on Isaac ROS segmentation
- Implements dynamic obstacle avoidance
- Validates navigation performance metrics
- Demonstrates safe navigation in cluttered environments
---
sidebar_position: 2
title: "Nav2 Installation and Setup Guide for Humanoid Applications"
---

# Nav2 Installation and Setup Guide for Humanoid Applications

## Overview
This guide provides step-by-step instructions for installing and setting up Navigation2 (Nav2) specifically for humanoid robot applications. Nav2 is the ROS 2 navigation stack that provides path planning, obstacle avoidance, and navigation capabilities. For humanoid robots, special configuration is needed to account for bipedal locomotion patterns and stability requirements.

## Learning Objectives
After completing this section, you will be able to:
- Install Navigation2 on your preferred platform
- Configure Nav2 for humanoid-specific navigation requirements
- Verify the installation with sample humanoid navigation scenarios

## Prerequisites
- ROS 2 installation (Humble Hawksbill recommended)
- Basic understanding of ROS 2 concepts
- Sufficient disk space (2+ GB recommended)
- For simulation: Isaac Sim or Gazebo installed

## Platform-Specific Installation

### Ubuntu Installation (Recommended)

#### System Requirements
- Ubuntu 20.04 LTS or 22.04 LTS
- 8GB+ RAM recommended
- 2GB+ free disk space

#### Installation Steps
1. **Install ROS 2 Humble Hawksbill (if not already installed)**
   ```bash
   # Set locale
   locale  # check for UTF-8
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8

   # Add ROS 2 apt repository
   sudo apt update && sudo apt install curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

   # Install ROS 2
   sudo apt update
   sudo apt install ros-humble-desktop
   sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
   ```

2. **Install Navigation2 via apt**
   ```bash
   # Source ROS 2
   source /opt/ros/humble/setup.bash

   # Install Navigation2 packages
   sudo apt update
   sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-gazebo-spawner
   sudo apt install ros-humble-dwb-core ros-humble-dwb-plugins ros-humble-dwb-msgs
   sudo apt install ros-humble-nav-2d-msgs ros-humble-navfn ros-humble-global-planner
   sudo apt install ros-humble-robot-localization ros-humble-slam-toolbox
   ```

3. **Install Humanoid-Specific Packages (Optional)**
   ```bash
   # Install packages for humanoid-specific navigation
   sudo apt install ros-humble-humanoid-nav-msgs ros-humble-humanoid-path-planner
   ```

4. **Verify Installation**
   ```bash
   # Source ROS 2
   source /opt/ros/humble/setup.bash

   # Check available Nav2 packages
   ros2 pkg list | grep nav2
   ```

### From Source Installation (Advanced Users)

For the latest features or development work:

1. **Create Workspace**
   ```bash
   mkdir -p ~/nav2_ws/src
   cd ~/nav2_ws
   ```

2. **Clone Navigation2 Repository**
   ```bash
   cd ~/nav2_ws/src
   git clone -b humble https://github.com/ros-planning/navigation2.git
   ```

3. **Install Dependencies and Build**
   ```bash
   cd ~/nav2_ws
   rosdep install -y --from-paths src --ignore-src --rosdistro humble
   colcon build --symlink-install
   source install/setup.bash
   ```

## Humanoid-Specific Configuration

### Bipedal Navigation Considerations
Unlike wheeled robots, humanoid robots have unique navigation requirements:
- Stability during turning and movement
- Footstep planning for safe locomotion
- Center of mass considerations
- Dynamic balance maintenance

### Key Configuration Parameters

#### Footstep Planner Integration
```yaml
# In your Nav2 configuration file
planner_server:
  ros__parameters:
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      # For humanoid applications, adjust tolerance for smoother paths
      tolerance: 0.5  # Increased for humanoid stability
      use_astar: false
      allow_unknown: true

footstep_planner:
  ros__parameters:
    use_sim_time: True
    # Humanoid-specific parameters
    step_width: 0.20  # Typical humanoid foot width
    step_length: 0.30 # Typical humanoid step length
    max_step_height: 0.10 # Maximum step-up height
    min_step_size: 0.10  # Minimum step size for stability
```

#### Controller Configuration for Bipedal Motion
```yaml
# Controller configuration for humanoid navigation
controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 10.0  # Lower frequency for stability
    min_x_velocity_threshold: 0.05  # Minimum velocity for bipedal stability
    min_y_velocity_threshold: 0.05
    min_theta_velocity_threshold: 0.05
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: True
      min_vel_x: 0.05  # Minimum forward velocity for stability
      min_vel_y: 0.0
      max_vel_x: 0.3   # Reduced for humanoid stability
      max_vel_y: 0.0
      max_vel_theta: 0.5
      min_speed_xy: 0.0
      max_speed_xy: 0.3
      min_speed_theta: 0.0
      acc_lim_x: 0.5   # Acceleration limits for bipedal stability
      acc_lim_y: 0.0
      acc_lim_theta: 0.8
      decel_lim_x: -0.5
      decel_lim_y: 0.0
      decel_lim_theta: -0.8
```

## Verification Steps

### Basic Functionality Test
1. **Launch Nav2 in Simulation**
   ```bash
   # Source ROS 2 and Nav2
   source /opt/ros/humble/setup.bash
   source /usr/share/gazebo/setup.sh  # If using Gazebo

   # Launch Nav2 with a simple world
   ros2 launch nav2_bringup navigation_launch.py \
     use_sim_time:=True \
     params_file:=/path/to/humanoid-nav2-params.yaml
   ```

2. **Test Navigation Commands**
   ```bash
   # In another terminal, send a navigation goal
   ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {z: 0.0, w: 1.0}}, header: {frame_id: map}}}"
   ```

3. **Monitor Navigation Performance**
   ```bash
   # Check navigation status
   ros2 topic echo /navigation_status

   # Check robot pose
   ros2 topic echo /amcl_pose
   ```

### Humanoid-Specific Tests
1. **Stability Verification**: Ensure the robot maintains balance during navigation
2. **Path Smoothing**: Verify that paths are smooth enough for bipedal locomotion
3. **Obstacle Avoidance**: Test that the robot can navigate around obstacles while maintaining stability

## Troubleshooting Common Issues

### Navigation Instability
- **Problem**: Robot tips over during navigation
- **Solution**: Reduce maximum velocities and accelerations in controller configuration
- **Check**: Verify center of mass and stability margins

### Path Planning Issues
- **Problem**: Robot takes unstable or inefficient paths
- **Solution**: Adjust inflation radius and costmap parameters for humanoid safety
- **Alternative**: Use footstep planner for complex terrain

### Simulation Integration
- **Problem**: Nav2 doesn't work properly with simulation
- **Solution**: Ensure use_sim_time is set to true in all configurations
- **Check**: Verify TF frames are properly published

## Hardware Recommendations

### Simulation Environment
- RAM: 8GB+ for basic simulation
- CPU: Multi-core processor for real-time simulation
- GPU: For visual rendering (if using Gazebo or Isaac Sim)

### Real Robot Integration
- Real-time capable computer (RT or Xenomai kernel)
- Reliable IMU for balance feedback
- Proper joint controllers for stable locomotion

## Integration with Isaac Tools

### Isaac Sim Integration
- Use Isaac Sim for Nav2 testing and validation
- Leverage Isaac Sim's accurate physics for humanoid simulation
- Integrate with Isaac ROS for perception-based navigation

### Isaac ROS Integration
- Combine Isaac ROS perception with Nav2 for autonomous navigation
- Use Isaac ROS VSLAM for localization in Nav2
- Integrate with Isaac Sim for end-to-end testing

## Next Steps
After successfully installing and verifying Nav2 for humanoid applications, proceed to:
- Configuring specific humanoid robot parameters
- Setting up costmaps for bipedal navigation
- Integrating with perception systems for autonomous navigation

## Additional Resources
- [Navigation2 Documentation](https://navigation.ros.org/)
- [ROS 2 Navigation Tutorials](https://navigation.ros.org/tutorials/)
- [Humanoid Robot Navigation Best Practices](https://humanoid-navigation.org/)
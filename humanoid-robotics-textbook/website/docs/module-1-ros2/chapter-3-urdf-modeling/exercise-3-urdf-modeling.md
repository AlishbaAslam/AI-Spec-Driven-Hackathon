---
title: Exercise 3 - Humanoid Robot Modeling with URDF
sidebar_position: 2
---

# Exercise 3 - Humanoid Robot Modeling with URDF

In this exercise, you'll create your own humanoid robot model using URDF (Unified Robot Description Format). You'll learn how to define links, joints, and materials to create a complete robot model that can be visualized and simulated.

## Prerequisites

Before starting this exercise, make sure you have:

1. Completed Chapter 1 and 2 exercises
2. A working ROS 2 environment with Gazebo and RViz
3. Understanding of basic XML syntax

## Exercise Objectives

By the end of this exercise, you will:

1. Create a complete humanoid robot URDF model
2. Understand the structure of URDF files
3. Learn how to visualize URDF models in RViz
4. Test your URDF model in Gazebo simulation

## Part 1: Understanding URDF Structure

Before creating your own model, let's review the structure of a URDF file:

```xml
<?xml version="1.0"?>
<robot name="robot_name">
  <!-- Links define the rigid parts of the robot -->
  <link name="link_name">
    <!-- Inertial properties for physics simulation -->
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>

    <!-- Visual properties for visualization -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
      <material name="color_name">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>

    <!-- Collision properties for physics simulation -->
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints connect links -->
  <joint name="joint_name" type="revolute">
    <parent link="parent_link_name"/>
    <child link="child_link_name"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Part 2: Creating Your Own Humanoid Robot

Create a simple humanoid robot with a body, head, and limbs.

### Step 1: Create the Basic Robot File

Create a file named `my_humanoid.urdf`:

```xml
<?xml version="1.0"?>
<robot name="my_humanoid">
  <!-- Base link (torso) -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
      <material name="body_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="head_material">
        <color rgba="1.0 0.8 0.6 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_arm">
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="arm_material">
        <color rgba="0.6 0.6 0.6 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Shoulder Joint -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_arm"/>
    <origin xyz="0.2 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_arm">
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="arm_material">
        <color rgba="0.6 0.6 0.6 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Shoulder Joint -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_arm"/>
    <origin xyz="-0.2 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>
</robot>
```

### Step 2: Validate Your URDF

Before visualizing, validate your URDF file:

```bash
# Install the check_urdf tool if not already installed
sudo apt-get install ros-humble-urdf-tutorial

# Check your URDF file
check_urdf my_humanoid.urdf
```

You should see output showing the robot structure without errors.

## Part 3: Visualizing Your Robot in RViz

### Step 1: Launch RViz with Robot Model

Create a launch file to visualize your robot:

Create `display_robot.launch.py`:

```python
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('your_robot_description_package')
    urdf_file = os.path.join(pkg_share, 'urdf', 'my_humanoid.urdf')

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_file]).perform({}),
            'use_sim_time': False
        }]
    )

    # RViz2 node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        rviz
    ])
```

### Step 2: Run the Visualization

For this exercise, you can use the default robot state publisher to visualize:

```bash
# Terminal 1: Publish the robot state
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat my_humanoid.urdf)

# Terminal 2: Launch RViz
rviz2
```

In RViz:
1. Add a RobotModel display
2. Set the Robot Description to "robot_description"
3. You should see your robot model displayed

## Part 4: Creating a More Complex Model

Now let's extend the model with legs to make it a complete humanoid:

### Step 1: Update Your URDF

Add legs to your `my_humanoid.urdf` file by appending these links and joints:

```xml
  <!-- Left Leg (Thigh) -->
  <link name="left_thigh">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.002"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
      <material name="leg_material">
        <color rgba="0.4 0.4 0.4 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Hip Joint -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.79" upper="0.79" effort="20.0" velocity="1.0"/>
  </joint>

  <!-- Left Lower Leg -->
  <link name="left_lower_leg">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.0015"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="leg_material">
        <color rgba="0.4 0.4 0.4 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Knee Joint -->
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.36" effort="20.0" velocity="1.0"/>
  </joint>

  <!-- Right Leg (Thigh) -->
  <link name="right_thigh">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.002"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
      <material name="leg_material">
        <color rgba="0.4 0.4 0.4 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Hip Joint -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="-0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.79" upper="0.79" effort="20.0" velocity="1.0"/>
  </joint>

  <!-- Right Lower Leg -->
  <link name="right_lower_leg">
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.0015"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="leg_material">
        <color rgba="0.4 0.4 0.4 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Knee Joint -->
  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.36" effort="20.0" velocity="1.0"/>
  </joint>
</robot>
```

### Step 2: Validate the Extended Model

```bash
check_urdf my_humanoid.urdf
```

## Part 5: Testing in Gazebo (Optional)

If you have Gazebo installed, you can test your model in simulation:

### Step 1: Create a Gazebo World File

Create `simple_world.world`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Your robot will be spawned here -->
  </world>
</sdf>
```

### Step 2: Launch Gazebo with Your Robot

```bash
# Terminal 1: Start Gazebo
gz sim -r simple_world.world

# Terminal 2: Spawn your robot
ros2 run gazebo_ros spawn_entity.py -entity my_humanoid -file $(pwd)/my_humanoid.urdf -x 0 -y 0 -z 1
```

## Part 6: Common URDF Issues and Solutions

### Issue 1: Missing Inertial Properties
- **Problem**: Robot falls through the ground in simulation
- **Solution**: Add proper inertial elements with mass and inertia values

### Issue 2: Invalid Joint Limits
- **Problem**: Robot moves in unexpected ways
- **Solution**: Set appropriate joint limits based on real robot capabilities

### Issue 3: Non-physical Inertia Values
- **Problem**: Robot behaves erratically in simulation
- **Solution**: Use realistic inertia values (diagonal elements positive, satisfy triangle inequality)

## Verification Steps

1. **Validate your URDF**:
   ```bash
   check_urdf my_humanoid.urdf
   ```

2. **View robot information**:
   ```bash
   # This will show robot tree structure
   check_urdf my_humanoid.urdf | head -20
   ```

3. **Visualize in RViz** (if available):
   - Launch RViz
   - Add RobotModel display
   - Set Robot Description to your URDF content

## Troubleshooting

### Common Issues

1. **XML Syntax Errors**: Make sure all tags are properly closed and attributes are quoted.

2. **Missing Dependencies**: Install required ROS packages:
   ```bash
   sudo apt install ros-humble-robot-state-publisher ros-humble-joint-state-publisher ros-humble-urdf-tutorial
   ```

3. **Visualization Problems**: Ensure all links have visual elements with proper geometry.

4. **Simulation Issues**: Make sure all links have both visual and collision elements.

## Summary

In this exercise, you've:

1. Created a complete humanoid robot model in URDF format
2. Learned about the essential elements of URDF: links, joints, inertial, visual, and collision properties
3. Validated your URDF model using ROS tools
4. Prepared your model for visualization in RViz and simulation in Gazebo

This foundational knowledge is essential for creating robot models that can be used in ROS 2 applications, simulations, and real robot implementations.
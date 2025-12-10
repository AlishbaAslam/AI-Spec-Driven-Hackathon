---
title: Chapter 3 - Modeling Humanoid Robots with URDF
sidebar_position: 1
---

# Chapter 3 - Modeling Humanoid Robots with URDF

This chapter covers the Unified Robot Description Format (URDF), which is an XML format for representing robot models including kinematics and dynamics. You'll learn how to model humanoid robots for simulation and control.

## Learning Objectives

By the end of this chapter, you will be able to:
- Create URDF files that describe simple humanoid robots
- Understand the structure of URDF including links, joints, and materials
- Visualize URDF models in ROS 2 environments
- Integrate URDF models with ROS 2 simulation systems

## Table of Contents
1. [Introduction to URDF](#introduction-to-urdf)
2. [URDF Structure and Components](#urdf-structure-and-components)
3. [Links and Joints](#links-and-joints)
4. [Visual and Collision Properties](#visual-and-collision-properties)
5. [Materials and Gazebo Plugins](#materials-and-gazebo-plugins)
6. [Hands-on Exercises](#hands-on-exercises)

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robots. It captures the physical and visual properties of a robot, including:

- **Kinematic structure**: How links are connected by joints
- **Visual representation**: How the robot appears in visualization tools
- **Collision properties**: How the robot interacts in simulation
- **Inertial properties**: Mass, center of mass, and inertia for dynamics

### Why URDF is Important

- **Simulation**: URDF models are essential for robot simulation in Gazebo
- **Visualization**: URDF enables visualization of robots in RViz
- **Planning**: Motion planners use URDF for collision checking
- **Control**: Robot controllers use URDF for kinematic calculations

## URDF Structure and Components

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="robot_name">
  <!-- Links define the physical components -->
  <link name="link_name">
    <!-- Inertial properties -->
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>

    <!-- Visual properties -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>

    <!-- Collision properties -->
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints connect links -->
  <joint name="joint_name" type="revolute">
    <parent link="parent_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Links and Joints

### Links

Links represent the rigid parts of the robot. Each link has:

- **Inertial properties**: Mass, center of mass, and inertia matrix
- **Visual properties**: How the link appears in visualization
- **Collision properties**: How the link behaves in simulation

#### Link Example

```xml
<link name="base_link">
  <inertial>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <mass value="10.0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>

  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.2" length="1.0"/>
    </geometry>
    <material name="white">
      <color rgba="1 1 1 1"/>
    </material>
  </visual>

  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.2" length="1.0"/>
    </geometry>
  </collision>
</link>
```

### Joints

Joints connect links and define how they can move relative to each other. Joint types include:

- **Fixed**: No movement (0 DOF)
- **Revolute**: Rotational movement around an axis (1 DOF)
- **Continuous**: Revolute joint without limits (1 DOF)
- **Prismatic**: Linear movement along an axis (1 DOF)
- **Floating**: 6 DOF movement (6 DOF)
- **Planar**: Movement on a plane (3 DOF)

#### Joint Example

```xml
<joint name="hip_joint" type="revolute">
  <parent link="base_link"/>
  <child link="thigh_link"/>
  <origin xyz="0 0 -0.5" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Visual and Collision Properties

### Visual Properties

The visual element defines how the link appears in visualization tools like RViz:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://robot_description/meshes/link.stl"/>
  </geometry>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
</visual>
```

### Collision Properties

The collision element defines how the link interacts in physics simulation:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://robot_description/meshes/link_collision.stl"/>
  </geometry>
</collision>
```

### Geometry Types

URDF supports several geometry types:
- `<box size="x y z"/>`
- `<cylinder radius="r" length="l"/>`
- `<sphere radius="r"/>`
- `<mesh filename="path/to/mesh"/>`

## Materials and Gazebo Plugins

### Materials

Materials define the visual appearance of links:

```xml
<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<material name="red">
  <color rgba="1 0 0 1"/>
</material>

<material name="green">
  <color rgba="0 1 0 1"/>
</material>
```

### Gazebo Plugins

For simulation, you can include Gazebo-specific elements:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <joint_name>joint_name</joint_name>
  </plugin>
</gazebo>
```

## Example: Simple Humanoid Robot URDF

Here's a simple humanoid robot with a body, two legs, and two arms:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Body -->
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
      <material name="white">
        <color rgba="1 1 1 1"/>
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
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
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
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
```

## URDF Best Practices and Tips

### 1. Proper Inertial Calculations

For accurate simulation, inertial properties must be physically realistic:

- **Mass**: Should represent the actual weight of the link
- **Inertia**: Values must satisfy the triangle inequality (e.g., `ixx + iyy ≥ izz`)
- **Origin**: Should represent the center of mass of the link

### 2. Collision vs Visual Geometry

- Use simpler geometries for collision to improve simulation performance
- Use detailed meshes for visual representation
- Consider using bounding boxes instead of complex meshes for collision

### 3. Joint Limitations

Always set realistic joint limits based on the physical constraints of your robot:

```xml
<limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
```

### 4. Proper Naming Conventions

Use consistent and descriptive names for links and joints:

- Use underscores for multi-word names: `left_upper_arm`
- Use a consistent naming scheme throughout the robot
- Use descriptive names that indicate function: `wheel_joint`, `camera_link`

## Validating URDF Models

### Using check_urdf Tool

Before using your URDF model, validate it with the ROS tool:

```bash
# Install the tool if not already installed
sudo apt install ros-humble-urdf-tutorial

# Check your URDF file
check_urdf your_robot.urdf
```

This will output information about your robot's structure and any errors.

### Common Validation Issues

1. **Non-physical inertia values**: Ensure inertia values satisfy triangle inequality
2. **Missing required elements**: All links need inertial, visual, and collision elements
3. **Invalid joint connections**: All parent and child links must exist
4. **Zero mass values**: All links must have positive mass values

## Visualization and Simulation Setup

### Setting Up Robot State Publisher

To visualize your robot in RViz, you need to publish the robot state:

```xml
<node pkg="robot_state_publisher" type="robot_state_publisher" name="robot_state_publisher">
  <param name="robot_description" command="$(find xacro)/xacro.py $(arg model)"/>
</node>
```

### Loading URDF in RViz

1. Add a RobotModel display in RViz
2. Set the Robot Description parameter to "robot_description"
3. Ensure the TF frames are being published

## Advanced URDF Concepts

### Xacro for Complex Models

For complex robots, consider using Xacro (XML Macros) to simplify URDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_humanoid">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Macro for creating arms -->
  <xacro:macro name="arm" params="side">
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </visual>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:arm side="left"/>
  <xacro:arm side="right"/>
</robot>
```

### Transmission Elements

For hardware interface, define transmissions:

```xml
<transmission name="left_wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Hands-on Exercises

Complete these exercises to practice URDF modeling:

1. Create a simple humanoid robot URDF with body, head, and limbs
2. Visualize your URDF model in RViz
3. Load your URDF in Gazebo simulation and test joint movements

Continue to the next sections for detailed instructions on each exercise.
---
title: Exercise 1 - Basic Robot in Gazebo Environment
sidebar_position: 7
---

# Exercise 1 - Basic Robot in Gazebo Environment

## Learning Objective
Create a complete digital twin simulation by building a simple robot model, placing it in a custom environment, configuring physics properties, and controlling it through ROS.

## Estimated Time
60-90 minutes

## Prerequisites
- Gazebo installation completed
- Basic understanding of URDF
- ROS workspace setup
- Understanding of physics configuration from previous sections

## Materials Needed
- Text editor for creating URDF and world files
- Terminal access for running Gazebo and ROS
- Basic knowledge of Linux commands

---

### Part 1: Robot Model Creation
**Objective**: Create a simple differential drive robot model with proper physical properties.

**Steps**:
1. Create a new URDF file named `simple_robot.urdf` with the following structure:
   - A base link (cylindrical shape, 0.3m diameter, 0.1m height)
   - Two wheel links (cylindrical, 0.05m radius, 0.04m width each)
   - Proper joints connecting wheels to base
   - Physical properties (mass, inertia) for each link
   - Gazebo plugin for ROS control integration

2. Add the following content to define your robot:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
      <material name="light_blue">
        <color rgba="0.5 0.7 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.04"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0002"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0002"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.15 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Transmissions for ROS control -->
  <transmission name="left_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_wheel_joint">
      <hardwareInterface>velocity_controllers/JointVelocityInterface</hardwareInterface>
    </joint>
  </transmission>

  <transmission name="right_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wheel_joint">
      <hardwareInterface>velocity_controllers/JointVelocityInterface</hardwareInterface>
    </joint>
  </transmission>

  <!-- Gazebo plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/simple_robot</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

3. Save the file in your robot description package or in a convenient location.

**Verification**: Your URDF file should have proper syntax and define all necessary elements for a differential drive robot.

---

### Part 2: Environment Creation
**Objective**: Create a simple world file with obstacles for the robot to navigate.

**Steps**:
1. Create a world file named `simple_world.world` with the following content:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include default elements -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add some obstacles -->
    <model name="box1">
      <pose>2 1 0.2 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="box2">
      <pose>-1 -1 0.3 0 0 0.5</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.8 0.6</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.8 0.6</size>
            </box>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add a ramp -->
    <model name="ramp">
      <pose>0 2 0 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>file://meshes/ramp.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://meshes/ramp.dae</uri>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

2. If you don't have a ramp mesh, create a simple inclined plane using basic shapes instead:

```xml
    <!-- Simple ramp using box -->
    <model name="ramp">
      <pose>0 2 0.15 0 0.3 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.5 0.3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.5 0.3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.2 1</ambient>
            <diffuse>0.8 0.8 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
```

**Verification**: Your world file should load properly in Gazebo and contain at least 2-3 static obstacles.

---

### Part 3: ROS Integration and Control
**Objective**: Launch the robot in the environment and control it using ROS.

**Steps**:
1. Create a launch file named `simple_robot_world.launch`:

```xml
<launch>
  <!-- Load robot description -->
  <param name="robot_description" textfile="$(find your_robot_description_package)/urdf/simple_robot.urdf" />

  <!-- Start Gazebo with custom world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find your_robot_gazebo)/worlds/simple_world.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
  </include>

  <!-- Spawn robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-param robot_description -urdf -model simple_robot -x 0 -y 0 -z 0.2" />

  <!-- Robot state publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
</launch>
```

2. Create a simple controller configuration file `simple_robot_control.yaml`:

```yaml
# Joint state controller
joint_state_controller:
  type: joint_state_controller/JointStateController
  publish_rate: 50

# Differential drive controller
diff_drive_controller:
  type: diff_drive_controller/DiffDriveController
  left_wheel: 'left_wheel_joint'
  right_wheel: 'right_wheel_joint'
  publish_rate: 50
  cmd_vel_timeout: 0.25

  # Wheel parameters
  wheel_separation: 0.3
  wheel_radius: 0.05

  # Odometry
  enable_odom_tf: true
  odom_frame: odom
  base_frame: base_link
```

3. Load the controller configuration in your launch file by adding:

```xml
  <!-- Load controller configuration -->
  <rosparam file="$(find your_robot_control)/config/simple_robot_control.yaml" command="load"/>

  <!-- Controller spawner -->
  <node name="controller_spawner" pkg="controller_manager" type="spawner" respawn="false"
        output="screen" args="joint_state_controller diff_drive_controller"/>
```

4. Launch your simulation:
```bash
roslaunch your_robot_gazebo simple_robot_world.launch
```

5. Test robot control by publishing velocity commands:
```bash
rostopic pub /simple_robot/cmd_vel geometry_msgs/Twist "linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.2" -r 10
```

**Verification**: The robot should appear in Gazebo, respond to velocity commands, and interact properly with the environment obstacles.

---

### Part 4: Autonomous Navigation
**Objective**: Implement a simple autonomous navigation behavior.

**Steps**:
1. Create a Python script `robot_controller.py` that implements obstacle avoidance:

```python
#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class SimpleNavigation:
    def __init__(self):
        rospy.init_node('simple_navigation', anonymous=True)

        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/simple_robot/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/simple_robot/scan', LaserScan, self.scan_callback)

        # Robot parameters
        self.safe_distance = 0.5  # meters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.scan_data = None

        # Control rate
        self.rate = rospy.Rate(10)

    def scan_callback(self, msg):
        self.scan_data = msg

    def get_front_distance(self):
        if self.scan_data is None:
            return float('inf')

        # Get front-facing readings (center 30 degrees)
        n = len(self.scan_data.ranges)
        center_idx = n // 2
        front_range = self.scan_data.ranges[center_idx]

        # Validate range
        if front_range == float('inf') or math.isnan(front_range):
            return float('inf')

        return front_range

    def run(self):
        while not rospy.is_shutdown():
            twist = Twist()

            if self.scan_data is None:
                # No sensor data, stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                front_dist = self.get_front_distance()

                if front_dist < self.safe_distance:
                    # Turn right to avoid obstacle
                    twist.linear.x = 0.0
                    twist.angular.z = -self.angular_speed
                else:
                    # Move forward
                    twist.linear.x = self.linear_speed
                    twist.angular.z = 0.0

            self.cmd_pub.publish(twist)
            self.rate.sleep()

if __name__ == '__main__':
    nav = SimpleNavigation()
    try:
        nav.run()
    except rospy.ROSInterruptException:
        pass
```

2. Make the script executable and run it:
```bash
chmod +x robot_controller.py
rosrun your_robot_package robot_controller.py
```

**Verification**: The robot should navigate autonomously, avoiding obstacles in the environment.

---

### Troubleshooting
**Common Issues**:
- **Issue**: Robot falls through the ground
  - **Solution**: Check that collision geometries are properly defined and the robot is spawned above ground level

- **Issue**: Robot doesn't respond to commands
  - **Solution**: Verify controller is loaded: `rosservice call /controller_manager/list_controllers`

- **Issue**: Robot moves erratically
  - **Solution**: Check physics parameters (mass, inertia) and controller configuration

**Helpful Commands**:
- `rostopic list` - List all available topics
- `rostopic echo /simple_robot/odom` - Monitor robot position
- `rosservice call /gazebo/get_model_state "model_name: 'simple_robot'"` - Get model state

---

### Solution and Discussion
**Expected Outcome**: A functional robot that can be controlled via ROS topics, navigates autonomously around obstacles, and interacts properly with the simulated environment.

**Key Concepts Learned**:
- URDF model creation with proper physical properties
- World file creation with static objects
- ROS-Gazebo integration using ros_control
- Basic autonomous navigation algorithm

**Extensions**:
- Add sensors (LiDAR, camera) to the robot
- Implement more sophisticated navigation algorithms
- Create multiple environments to test different scenarios

---

### Assessment Questions
1. What happens if you change the robot's mass in the URDF? How does this affect simulation behavior?
2. Why is it important to have proper inertia values in robot simulation?
3. How would you modify the autonomous navigation to follow a specific path rather than just avoiding obstacles?
4. What are the advantages of using a physics-based simulation for robotics development?
---
sidebar_position: 5
title: "Complete Digital Twin Example"
---

# Complete Digital Twin Example: Mobile Robot with Sensors

## Overview

This chapter brings together all the concepts learned in the previous chapters to create a complete digital twin system for a mobile robot equipped with multiple sensors. We'll integrate physics simulation from Gazebo, visualization from Unity, and sensor simulation to create a comprehensive digital twin.

## Learning Objectives

After completing this chapter, you will be able to:
- Combine Gazebo physics simulation with Unity visualization
- Integrate multiple sensor simulations in a single system
- Create a complete digital twin workflow
- Validate the digital twin system against real-world expectations

## System Architecture

The complete digital twin system consists of:

1. **Gazebo Simulation Layer**: Physics engine, robot dynamics, sensor simulation
2. **ROS Communication Layer**: Message passing, topic management, services
3. **Unity Visualization Layer**: 3D rendering, user interface, interaction
4. **Synchronization Layer**: Real-time state synchronization between environments

## Robot Model Setup

### URDF Definition

First, let's define a simple mobile robot with multiple sensors in URDF format:

```xml
<?xml version="1.0"?>
<robot name="multi_sensor_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.25"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.42" ixy="0.0" ixz="0.0" iyy="0.42" iyz="0.0" izz="0.8"/>
    </inertial>
  </link>

  <!-- Lidar sensor -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>

  <!-- Camera sensor -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
  </link>

  <!-- IMU sensor -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="imu_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
  </link>

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="lidar_link">
    <sensor type="ray" name="lidar_sensor">
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-1.570796</min_angle>
            <max_angle>1.570796</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <topicName>/scan</topicName>
        <frameName>lidar_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>30.0</updateRate>
        <cameraName>camera</cameraName>
        <imageTopicName>image_raw</imageTopicName>
        <cameraInfoTopicName>camera_info</cameraInfoTopicName>
        <frameName>camera_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <alwaysOn>true</alwaysOn>
        <bodyName>imu_link</bodyName>
        <topicName>imu</topicName>
        <serviceName>imu_service</serviceName>
        <gaussianNoise>0.01</gaussianNoise>
        <updateRate>100.0</updateRate>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Gazebo World Setup

Create a world file for our simulation:

```xml
<?xml version="1.0" ?>
<sdf version="1.4">
  <world name="digital_twin_world">
    <!-- Include the sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Simple room with obstacles -->
    <model name="wall_1">
      <pose>0 5 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_1">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include our robot -->
    <include>
      <uri>model://multi_sensor_robot</uri>
      <pose>0 0 0.2 0 0 0</pose>
    </include>

    <!-- Lighting -->
    <light name="spotlight" type="spot">
      <pose>0 0 5 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>10</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
      <spot>
        <inner_angle>0.1</inner_angle>
        <outer_angle>0.3</outer_angle>
        <falloff>10</falloff>
      </spot>
    </light>
  </world>
</sdf>
```

## Launch File Configuration

Create a launch file to start the complete simulation:

```xml
<launch>
  <!-- Start Gazebo with our world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find multi_sensor_robot)/worlds/digital_twin_world.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Spawn the robot in Gazebo -->
  <param name="robot_description" command="$(find xacro)/xacro.py '$(find multi_sensor_robot)/urdf/multi_sensor_robot.urdf.xacro'" />
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-param robot_description -urdf -model multi_sensor_robot -x 0 -y 0 -z 0.2" />

  <!-- Robot state publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />

  <!-- Joint state publisher -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher">
    <param name="use_gui" value="false" />
  </node>

  <!-- TF broadcasters for sensors -->
  <node pkg="tf" type="static_transform_publisher" name="lidar_broadcaster"
        args="0.0 0.0 0.15 0 0 0 base_link lidar_link 100" />
  <node pkg="tf" type="static_transform_publisher" name="camera_broadcaster"
        args="0.2 0.0 0.1 0 0 0 base_link camera_link 100" />
  <node pkg="tf" type="static_transform_publisher" name="imu_broadcaster"
        args="0.0 0.0 0.05 0 0 0 base_link imu_link 100" />

  <!-- Start ROS bridge server -->
  <include file="$(find rosbridge_server)/launch/rosbridge_websocket.launch">
    <arg name="port" value="9090"/>
  </include>

  <!-- Start the integration synchronizer -->
  <node name="gazebo_unity_sync" pkg="multi_sensor_robot" type="gazebo_unity_sync_node" output="screen" />
</launch>
```

## Unity Robot Model Setup

In Unity, create the corresponding robot model with the same joint structure:

1. Create a base cylinder for the main robot body
2. Add a small cylinder for the lidar sensor at the top
3. Add a cube for the camera sensor at the front
4. Add a small cube for the IMU sensor

### Unity Robot Controller Script

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor_msgs;
using RosSharp.Messages.Geometry_msgs;
using RosSharp.Messages.Nav_msgs;

public class DigitalTwinRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "multi_sensor_robot";
    public string jointStateTopic = "/unity/joint_states";
    public string lidarTopic = "/scan";
    public string cameraTopic = "/camera/image_raw";
    public string imuTopic = "/imu";

    [Header("Joint Configuration")]
    public Transform lidarJoint;
    public Transform cameraJoint;
    public Transform imuJoint;

    [Header("Synchronization")]
    public float updateRate = 60f;
    public bool enableSynchronization = true;

    private RosSocket rosSocket;
    private JointStateSubscriber jointStateSubscriber;
    private LidarVisualizer lidarVisualizer;
    private CameraVisualizer cameraVisualizer;
    private ImuVisualizer imuVisualizer;

    void Start()
    {
        ConnectToRosBridge();
        InitializeRobotModel();
    }

    private void ConnectToRosBridge()
    {
        RosBridgeClient.Protocols.WebSocketNetProtocol protocol =
            new RosBridgeClient.Protocols.WebSocketNetProtocol("ws://127.0.0.1:9090");

        rosSocket = new RosSocket(protocol);

        // Subscribe to joint states from Gazebo
        jointStateSubscriber = new JointStateSubscriber(rosSocket, jointStateTopic, OnJointStateReceived);

        Debug.Log($"Connected to ROS Bridge for Digital Twin Robot: {robotName}");
    }

    private void InitializeRobotModel()
    {
        // Initialize joint positions to zero
        if (lidarJoint != null) lidarJoint.localRotation = Quaternion.identity;
        if (cameraJoint != null) cameraJoint.localRotation = Quaternion.identity;
        if (imuJoint != null) imuJoint.localRotation = Quaternion.identity;
    }

    private void OnJointStateReceived(JointState jointState)
    {
        if (!enableSynchronization) return;

        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];

            UpdateJointByName(jointName, jointPosition);
        }
    }

    private void UpdateJointByName(string name, float position)
    {
        switch (name)
        {
            case "lidar_joint":
                if (lidarJoint != null)
                    lidarJoint.localRotation = Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
                break;
            case "camera_joint":
                if (cameraJoint != null)
                    cameraJoint.localRotation = Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
                break;
            case "imu_joint":
                if (imuJoint != null)
                    imuJoint.localRotation = Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
                break;
        }
    }

    void OnDestroy()
    {
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}
```

## Sensor Visualization in Unity

### Lidar Visualization Script

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor_msgs;

public class LidarVisualizer : MonoBehaviour
{
    [Header("Lidar Configuration")]
    public int rayCount = 720;
    public float maxRange = 30.0f;
    public float minRange = 0.1f;
    public Material lidarMaterial;

    [Header("Visualization")]
    public GameObject lidarPointPrefab;
    public Transform pointContainer;

    private LineRenderer[] lineRenderers;
    private GameObject[] lidarPoints;
    private int currentPointIndex = 0;

    void Start()
    {
        InitializeLidarVisualization();
    }

    private void InitializeLidarVisualization()
    {
        // Create line renderers for each ray
        lineRenderers = new LineRenderer[rayCount];
        for (int i = 0; i < rayCount; i++)
        {
            GameObject lineObj = new GameObject($"LidarRay_{i}");
            lineObj.transform.SetParent(transform);
            LineRenderer lr = lineObj.AddComponent<LineRenderer>();
            lr.material = lidarMaterial;
            lr.startWidth = 0.01f;
            lr.endWidth = 0.01f;
            lr.positionCount = 2;
            lineRenderers[i] = lr;
        }

        // Create point container if not set
        if (pointContainer == null)
        {
            pointContainer = new GameObject("LidarPoints").transform;
            pointContainer.SetParent(transform);
        }

        // Initialize lidar points array
        lidarPoints = new GameObject[rayCount];
        for (int i = 0; i < rayCount; i++)
        {
            lidarPoints[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            lidarPoints[i].transform.SetParent(pointContainer);
            lidarPoints[i].GetComponent<Renderer>().material = lidarMaterial;
            lidarPoints[i].transform.localScale = Vector3.one * 0.02f;
            lidarPoints[i].SetActive(false);
        }
    }

    public void UpdateLidarData(LaserScan scan)
    {
        if (scan.ranges.Count != rayCount)
        {
            Debug.LogWarning($"Lidar data count mismatch: expected {rayCount}, got {scan.ranges.Count}");
            return;
        }

        for (int i = 0; i < rayCount; i++)
        {
            float range = (float)scan.ranges[i];
            if (range >= minRange && range <= maxRange)
            {
                // Calculate angle for this ray
                float angle = (float)scan.angle_min + i * (float)scan.angle_increment;

                // Calculate world position
                Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
                Vector3 worldPos = transform.position + direction * range;

                // Update line renderer
                lineRenderers[i].SetPosition(0, transform.position);
                lineRenderers[i].SetPosition(1, worldPos);

                // Update point visualization
                lidarPoints[i].transform.position = worldPos;
                lidarPoints[i].SetActive(true);
            }
            else
            {
                // Hide this ray
                lineRenderers[i].SetPosition(0, transform.position);
                lineRenderers[i].SetPosition(1, transform.position);
                lidarPoints[i].SetActive(false);
            }
        }
    }
}
```

### Camera Visualization Script

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor_msgs;

public class CameraVisualizer : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Material cameraMaterial;
    public int width = 640;
    public int height = 480;

    private Renderer cameraRenderer;
    private Texture2D cameraTexture;
    private Color32[] colorBuffer;

    void Start()
    {
        InitializeCameraVisualization();
    }

    private void InitializeCameraVisualization()
    {
        // Get the renderer component
        cameraRenderer = GetComponent<Renderer>();
        if (cameraRenderer == null)
        {
            Debug.LogError("CameraVisualizer requires a Renderer component");
            return;
        }

        // Create texture
        cameraTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        cameraRenderer.material.mainTexture = cameraTexture;

        // Initialize color buffer
        colorBuffer = new Color32[width * height];
    }

    public void UpdateCameraImage(Image image)
    {
        if (image.data.Count != width * height * 3)
        {
            Debug.LogWarning($"Camera data size mismatch: expected {width * height * 3}, got {image.data.Count}");
            return;
        }

        // Convert ROS image data to Color32 array
        for (int i = 0; i < width * height; i++)
        {
            Color32 color = new Color32(
                (byte)image.data[i * 3],     // R
                (byte)image.data[i * 3 + 1], // G
                (byte)image.data[i * 3 + 2], // B
                255                          // A
            );
            colorBuffer[i] = color;
        }

        // Update texture
        cameraTexture.SetPixels32(colorBuffer);
        cameraTexture.Apply();
    }
}
```

## Integration Test Script

Create a comprehensive test script to validate the complete digital twin system:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import time
import threading
from collections import deque

class DigitalTwinValidator:
    def __init__(self):
        rospy.init_node('digital_twin_validator')

        # Publishers for robot control
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscribers for sensor data
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.camera_sub = rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Data storage
        self.lidar_data = deque(maxlen=10)
        self.camera_data = deque(maxlen=10)
        self.imu_data = deque(maxlen=10)
        self.odom_data = deque(maxlen=10)

        # Validation metrics
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

        # Test timer
        self.test_timer = rospy.Timer(rospy.Duration(5.0), self.run_validation_tests)

        rospy.loginfo("Digital Twin Validator initialized")

    def lidar_callback(self, msg):
        self.lidar_data.append(msg)

    def camera_callback(self, msg):
        self.camera_data.append(msg)

    def imu_callback(self, msg):
        self.imu_data.append(msg)

    def odom_callback(self, msg):
        self.odom_data.append(msg)

    def run_validation_tests(self, event):
        """Run comprehensive validation tests on the digital twin system"""
        self.total_tests += 1

        rospy.loginfo(f"\n=== Running Digital Twin Validation Test #{self.total_tests} ===")

        # Test 1: Sensor data availability
        test1_passed = self.test_sensor_data_availability()
        if test1_passed:
            rospy.loginfo("✓ Test 1 PASSED: Sensor data is available")
        else:
            rospy.logerr("✗ Test 1 FAILED: Sensor data issues detected")

        # Test 2: Robot mobility
        test2_passed = self.test_robot_mobility()
        if test2_passed:
            rospy.loginfo("✓ Test 2 PASSED: Robot mobility verified")
        else:
            rospy.logerr("✗ Test 2 FAILED: Robot mobility issues detected")

        # Test 3: Sensor range validation
        test3_passed = self.test_sensor_ranges()
        if test3_passed:
            rospy.loginfo("✓ Test 3 PASSED: Sensor ranges are valid")
        else:
            rospy.logerr("✗ Test 3 FAILED: Sensor range issues detected")

        # Test 4: Data consistency
        test4_passed = self.test_data_consistency()
        if test4_passed:
            rospy.loginfo("✓ Test 4 PASSED: Data consistency verified")
        else:
            rospy.logerr("✗ Test 4 FAILED: Data consistency issues detected")

        # Overall result
        all_tests_passed = all([test1_passed, test2_passed, test3_passed, test4_passed])
        if all_tests_passed:
            self.passed_tests += 1
            rospy.loginfo(f"✓ OVERALL RESULT: Test #{self.total_tests} PASSED")
        else:
            self.failed_tests += 1
            rospy.logerr(f"✗ OVERALL RESULT: Test #{self.total_tests} FAILED")

        # Print summary
        self.print_test_summary()

    def test_sensor_data_availability(self):
        """Test that all sensor data is being published"""
        checks = [
            len(self.lidar_data) > 0,
            len(self.camera_data) > 0,
            len(self.imu_data) > 0,
            len(self.odom_data) > 0
        ]
        return all(checks)

    def test_robot_mobility(self):
        """Test that the robot can move when commanded"""
        # Store initial position
        if len(self.odom_data) < 1:
            return False

        initial_pose = self.odom_data[-1].pose.pose
        initial_x = initial_pose.position.x
        initial_y = initial_pose.position.y

        # Send movement command
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        # Wait for movement
        rospy.sleep(2.0)

        # Stop robot
        cmd.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)

        # Check if position changed significantly
        if len(self.odom_data) < 1:
            return False

        final_pose = self.odom_data[-1].pose.pose
        final_x = final_pose.position.x
        final_y = final_pose.position.y

        distance_moved = np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2)
        return distance_moved > 0.1  # Should have moved at least 10cm

    def test_sensor_ranges(self):
        """Test that sensor data is within expected ranges"""
        if len(self.lidar_data) < 1:
            return False

        lidar_msg = self.lidar_data[-1]

        # Check that lidar ranges are within expected bounds
        valid_ranges = [r for r in lidar_msg.ranges if lidar_msg.range_min <= r <= lidar_msg.range_max]
        range_validity = len(valid_ranges) == len(lidar_msg.ranges)

        # Check IMU data validity
        if len(self.imu_data) < 1:
            return False

        imu_msg = self.imu_data[-1]
        imu_validity = (
            abs(imu_msg.linear_acceleration.x) <= 20.0 and
            abs(imu_msg.linear_acceleration.y) <= 20.0 and
            abs(imu_msg.linear_acceleration.z) <= 20.0
        )

        return range_validity and imu_validity

    def test_data_consistency(self):
        """Test consistency of data streams"""
        # Check that data is being updated regularly
        if len(self.lidar_data) < 2 or len(self.odom_data) < 2:
            return False

        # Check timestamp consistency
        lidar_msg1 = self.lidar_data[-2]
        lidar_msg2 = self.lidar_data[-1]

        # Time between messages should be reasonable (less than 1 second for 10Hz+ updates)
        time_diff = abs((lidar_msg2.header.stamp - lidar_msg1.header.stamp).to_sec())
        reasonable_timing = 0.01 <= time_diff <= 1.0  # Between 10ms and 1s

        return reasonable_timing

    def print_test_summary(self):
        """Print overall test summary"""
        rospy.loginfo(f"\n=== Test Summary ===")
        rospy.loginfo(f"Total Tests Run: {self.total_tests}")
        rospy.loginfo(f"Passed: {self.passed_tests}")
        rospy.loginfo(f"Failed: {self.failed_tests}")
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            rospy.loginfo(f"Success Rate: {success_rate:.1f}%")
        rospy.loginfo("==================\n")

if __name__ == '__main__':
    try:
        validator = DigitalTwinValidator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Running the Complete Digital Twin

### 1. Start the Gazebo Simulation

```bash
roslaunch multi_sensor_robot digital_twin.launch
```

### 2. Start the Unity Visualization

1. Open the Unity project
2. Load the digital twin scene
3. Run the Unity application

### 3. Monitor the Integration

```bash
# Monitor joint states
rostopic echo /joint_states

# Monitor sensor data
rostopic echo /scan | head -n 20
rostopic echo /imu | head -n 10

# Monitor the synchronization
rostopic echo /integration/sync_quality
```

### 4. Control the Robot

```bash
# Send velocity commands to move the robot
rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.2" -r 10
```

## Performance Optimization

### 1. Gazebo Optimization

- Reduce physics update rate if high fidelity isn't required
- Use simpler collision meshes for better performance
- Limit sensor update rates to necessary frequencies

### 2. Unity Optimization

- Use Level of Detail (LOD) for complex models
- Implement occlusion culling for large scenes
- Optimize shader complexity for real-time rendering

### 3. Network Optimization

- Compress sensor data for transmission
- Use appropriate message throttling
- Implement quality-of-service settings

## Troubleshooting

### Common Issues and Solutions

1. **Synchronization Delay**
   - Check network latency between Gazebo and Unity
   - Increase update rates if necessary
   - Verify ROS bridge connection stability

2. **Sensor Data Not Updating**
   - Verify Gazebo plugins are properly configured
   - Check ROS topic connections
   - Validate sensor frame transformations

3. **Performance Issues**
   - Monitor CPU and GPU usage in both environments
   - Reduce simulation complexity if needed
   - Optimize Unity rendering settings

## Summary

This complete digital twin example demonstrates the integration of:

- **Physics Simulation**: Accurate robot dynamics and sensor modeling in Gazebo
- **Visualization**: Real-time 3D rendering in Unity
- **Synchronization**: Real-time state coordination between environments
- **Sensor Integration**: Multiple sensor types working together
- **Validation**: Comprehensive testing of the integrated system

The system provides a foundation for developing more complex digital twin applications that can be extended with additional sensors, more complex environments, and real-world robot integration.

## Next Steps

1. **Real Robot Integration**: Connect the digital twin to a physical robot
2. **Cloud Deployment**: Deploy the system for remote access
3. **Advanced Analytics**: Add data analysis and machine learning capabilities
4. **User Interface**: Create a more sophisticated user interface for interaction
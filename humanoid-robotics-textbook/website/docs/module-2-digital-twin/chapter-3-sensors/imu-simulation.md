---
title: IMU Simulation
sidebar_position: 4
---

# IMU Simulation

## Overview

Inertial Measurement Units (IMUs) are crucial sensors that provide measurements of linear acceleration, angular velocity, and sometimes magnetic field orientation. In digital twin applications, realistic IMU simulation is essential for accurate state estimation, navigation, and control. This tutorial will guide you through setting up realistic IMU simulation in Gazebo with proper noise models and drift characteristics.

## Learning Objectives

After completing this tutorial, you will be able to:
- Configure realistic IMU sensors in Gazebo with proper noise and drift characteristics
- Implement realistic sensor models that match real-world IMU behavior
- Generate IMU data that includes appropriate noise, bias, and drift
- Process IMU data in ROS for digital twin applications
- Validate simulated IMU data against real sensor specifications

## Prerequisites

- Completed Chapter 1 (Gazebo Physics Simulation)
- Basic understanding of ROS sensor_msgs and IMU message types
- Knowledge of 3D rotations and coordinate systems (quaternions, Euler angles)
- Understanding of sensor fusion concepts (optional but helpful)

## Understanding IMUs in Digital Twins

IMUs measure three-dimensional linear acceleration and angular velocity, providing crucial information about a robot's motion and orientation. In digital twin applications, IMU simulation must:

- Provide accurate acceleration and angular velocity measurements
- Include realistic noise, bias, and drift characteristics
- Operate at appropriate update rates (typically 100Hz+)
- Generate data in standard ROS formats
- Integrate with existing state estimation and control systems

## Setting Up a Basic IMU in Gazebo

### Creating an IMU Sensor Model

Let's create a basic IMU configuration in a Gazebo model file. Create a new file called `imu_sensor.model`:

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="imu_sensor">
    <link name="imu_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.01</mass>
        <inertia>
          <ixx>1e-6</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1e-6</iyy>
          <iyz>0</iyz>
          <izz>1e-6</izz>
        </inertia>
      </inertial>

      <visual name="imu_visual">
        <geometry>
          <box>
            <size>0.01 0.01 0.01</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>

      <collision name="imu_collision">
        <geometry>
          <box>
            <size>0.01 0.01 0.01</size>
          </box>
        </geometry>
      </collision>

      <sensor name="imu_sensor" type="imu">
        <pose>0 0 0 0 0 0</pose>
        <always_on>1</always_on>
        <update_rate>100</update_rate>
        <imu>
          <angular_velocity>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
              </noise>
            </z>
          </angular_velocity>
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
              </noise>
            </z>
          </linear_acceleration>
        </imu>

        <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
          <topicName>/imu/data</topicName>
          <bodyName>imu_link</bodyName>
          <serviceName>/imu/service</serviceName>
          <gaussianNoise>0.01</gaussianNoise>
          <updateRate>100.0</updateRate>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

### Understanding the Configuration

- `update_rate`: IMU update rate (100 Hz, typical for modern IMUs)
- `angular_velocity`: Noise characteristics for gyroscope measurements
- `linear_acceleration`: Noise characteristics for accelerometer measurements
- `gaussianNoise`: Overall noise parameter for the plugin

## Advanced IMU Configuration with Realistic Characteristics

Real IMUs have complex behaviors including bias, drift, and temperature effects. Let's create a more advanced configuration:

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="advanced_imu_sensor">
    <link name="imu_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.02</mass>
        <inertia>
          <ixx>2e-6</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>2e-6</iyy>
          <iyz>0</iyz>
          <izz>2e-6</izz>
        </inertia>
      </inertial>

      <visual name="imu_visual">
        <geometry>
          <box>
            <size>0.02 0.02 0.005</size>
          </box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.3 0.3 0.3 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>

      <collision name="imu_collision">
        <geometry>
          <box>
            <size>0.02 0.02 0.005</size>
          </box>
        </geometry>
      </collision>

      <sensor name="realistic_imu" type="imu">
        <pose>0 0 0 0 0 0</pose>
        <always_on>1</always_on>
        <update_rate>200</update_rate>
        <imu>
          <angular_velocity>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>1.745e-4</stddev> <!-- 0.01 deg/s -->
                <bias_mean>0.0</bias_mean>
                <bias_stddev>8.727e-7</bias_stddev> <!-- 0.05 deg/s over 1000s -->
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>1.745e-4</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>8.727e-7</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>1.745e-4</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>8.727e-7</bias_stddev>
              </noise>
            </z>
          </angular_velocity>
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0085</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0085</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0085</bias_stddev>
              </noise>
            </z>
          </linear_acceleration>
        </imu>

        <plugin name="advanced_imu_controller" filename="libgazebo_ros_imu.so">
          <topicName>/imu/data_raw</topicName>
          <bodyName>imu_link</bodyName>
          <serviceName>/imu/service</serviceName>
          <gaussianNoise>0.01</gaussianNoise>
          <updateRate>200.0</updateRate>
          <xyzOffset>0 0 0</xyzOffset>
          <rpyOffset>0 0 0</rpyOffset>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

### Key Features of Advanced Configuration

- **Higher update rate**: 200 Hz for more responsive measurements
- **Bias modeling**: Long-term bias drift characteristics
- **Realistic noise levels**: Based on typical commercial IMU specifications
- **Separate noise for each axis**: Individual noise parameters for X, Y, Z axes

## Creating a ROS Node for IMU Data Processing

Let's create a ROS node that processes the simulated IMU data:

```cpp
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <iostream>
#include <vector>
#include <numeric>

class IMUProcessor {
public:
    IMUProcessor(ros::NodeHandle& nh) : nh_(nh) {
        // Subscribe to IMU topic
        imu_sub_ = nh_.subscribe("/imu/data_raw", 100, &IMUProcessor::imuCallback, this);

        // Publisher for processed IMU data
        processed_imu_pub_ = nh_.advertise<sensor_msgs::Imu>("/imu/data_processed", 100);
        orientation_pub_ = nh_.advertise<geometry_msgs::Quaternion>("/imu/orientation", 100);

        // Initialize statistics
        measurement_count_ = 0;
        angular_velocity_bias_.x = 0.0;
        angular_velocity_bias_.y = 0.0;
        angular_velocity_bias_.z = 0.0;

        ROS_INFO("IMU Processor initialized");
    }

private:
    ros::NodeHandle& nh_;
    ros::Subscriber imu_sub_;
    ros::Publisher processed_imu_pub_;
    ros::Publisher orientation_pub_;

    int measurement_count_;
    geometry_msgs::Vector3 angular_velocity_bias_;
    std::vector<double> angular_velocity_x_history_;
    std::vector<double> angular_velocity_y_history_;
    std::vector<double> angular_velocity_z_history_;

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        measurement_count_++;

        // Process the raw IMU data
        sensor_msgs::Imu processed_msg = *msg;

        // Calculate running statistics for bias estimation
        angular_velocity_x_history_.push_back(msg->angular_velocity.x);
        angular_velocity_y_history_.push_back(msg->angular_velocity.y);
        angular_velocity_z_history_.push_back(msg->angular_velocity.z);

        // Keep only the last 1000 measurements for bias calculation
        if (angular_velocity_x_history_.size() > 1000) {
            angular_velocity_x_history_.erase(angular_velocity_x_history_.begin());
            angular_velocity_y_history_.erase(angular_velocity_y_history_.begin());
            angular_velocity_z_history_.erase(angular_velocity_z_history_.begin());
        }

        // Calculate bias if we have enough measurements
        if (angular_velocity_x_history_.size() >= 100) {
            angular_velocity_bias_.x = std::accumulate(angular_velocity_x_history_.begin(),
                                                      angular_velocity_x_history_.end(), 0.0) /
                                                      angular_velocity_x_history_.size();
            angular_velocity_bias_.y = std::accumulate(angular_velocity_y_history_.begin(),
                                                      angular_velocity_y_history_.end(), 0.0) /
                                                      angular_velocity_y_history_.size();
            angular_velocity_bias_.z = std::accumulate(angular_velocity_z_history_.begin(),
                                                      angular_velocity_z_history_.end(), 0.0) /
                                                      angular_velocity_z_history_.size();

            // Remove bias from the current measurements
            processed_msg.angular_velocity.x -= angular_velocity_bias_.x;
            processed_msg.angular_velocity.y -= angular_velocity_bias_.y;
            processed_msg.angular_velocity.z -= angular_velocity_bias_.z;
        }

        // Calculate orientation from angular velocity (simplified integration)
        static ros::Time last_time = msg->header.stamp;
        ros::Duration dt = msg->header.stamp - last_time;
        last_time = msg->header.stamp;

        if (dt.toSec() > 0) {
            // Integrate angular velocity to get orientation (simplified)
            // In practice, you'd use more sophisticated integration methods
            static double roll = 0.0, pitch = 0.0, yaw = 0.0;

            roll += msg->angular_velocity.x * dt.toSec();
            pitch += msg->angular_velocity.y * dt.toSec();
            yaw += msg->angular_velocity.z * dt.toSec();

            // Convert to quaternion
            tf2::Quaternion q;
            q.setRPY(roll, pitch, yaw);

            processed_msg.orientation.x = q.x();
            processed_msg.orientation.y = q.y();
            processed_msg.orientation.z = q.z();
            processed_msg.orientation.w = q.w();
        }

        // Publish processed data
        processed_imu_pub_.publish(processed_msg);

        // Publish orientation separately
        geometry_msgs::Quaternion orientation_msg;
        orientation_msg = processed_msg.orientation;
        orientation_pub_.publish(orientation_msg);

        // Log statistics periodically
        if (measurement_count_ % 1000 == 0) {
            ROS_INFO("IMU Stats - Bias: [%.6f, %.6f, %.6f] rad/s",
                     angular_velocity_bias_.x, angular_velocity_bias_.y, angular_velocity_bias_.z);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "imu_processor");
    ros::NodeHandle nh;

    IMUProcessor processor(nh);

    ros::spin();

    return 0;
}
```

## Python Alternative for IMU Processing

Here's a Python version of the IMU processor:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import deque
import math

class IMUProcessor:
    def __init__(self):
        rospy.init_node('imu_processor')

        # Subscribe to IMU topic
        self.imu_sub = rospy.Subscriber('/imu/data_raw', Imu, self.imu_callback)

        # Publisher for processed IMU data
        self.processed_imu_pub = rospy.Publisher('/imu/data_processed', Imu, queue_size=100)
        self.orientation_pub = rospy.Publisher('/imu/orientation', Quaternion, queue_size=10)

        # Initialize statistics
        self.measurement_count = 0
        self.angular_velocity_bias = np.array([0.0, 0.0, 0.0])

        # History buffers for bias calculation
        self.ang_vel_history = {
            'x': deque(maxlen=1000),
            'y': deque(maxlen=1000),
            'z': deque(maxlen=1000)
        }

        rospy.loginfo("IMU Processor initialized")

    def imu_callback(self, msg):
        self.measurement_count += 1

        # Process the raw IMU data
        processed_msg = Imu()
        processed_msg.header = msg.header

        # Calculate running statistics for bias estimation
        self.ang_vel_history['x'].append(msg.angular_velocity.x)
        self.ang_vel_history['y'].append(msg.angular_velocity.y)
        self.ang_vel_history['z'].append(msg.angular_velocity.z)

        # Calculate bias if we have enough measurements
        if len(self.ang_vel_history['x']) >= 100:
            bias_x = sum(self.ang_vel_history['x']) / len(self.ang_vel_history['x'])
            bias_y = sum(self.ang_vel_history['y']) / len(self.ang_vel_history['y'])
            bias_z = sum(self.ang_vel_history['z']) / len(self.ang_vel_history['z'])

            self.angular_velocity_bias = np.array([bias_x, bias_y, bias_z])

            # Remove bias from the current measurements
            processed_msg.angular_velocity.x = msg.angular_velocity.x - bias_x
            processed_msg.angular_velocity.y = msg.angular_velocity.y - bias_y
            processed_msg.angular_velocity.z = msg.angular_velocity.z - bias_z
        else:
            # Not enough data for bias calculation
            processed_msg.angular_velocity = msg.angular_velocity

        # Copy linear acceleration (with bias removal if needed)
        processed_msg.linear_acceleration = msg.linear_acceleration

        # Calculate orientation from angular velocity (simplified integration)
        current_time = rospy.Time.now()
        dt = 0.01  # Assume 100Hz rate, or calculate from timestamps

        # Integrate angular velocity to get orientation (simplified)
        # In practice, you'd use more sophisticated integration methods
        if not hasattr(self, 'last_time'):
            self.last_time = current_time
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
        else:
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time

            self.roll += processed_msg.angular_velocity.x * dt
            self.pitch += processed_msg.angular_velocity.y * dt
            self.yaw += processed_msg.angular_velocity.z * dt

        # Convert to quaternion
        q = quaternion_from_euler(self.roll, self.pitch, self.yaw)
        processed_msg.orientation.x = q[0]
        processed_msg.orientation.y = q[1]
        processed_msg.orientation.z = q[2]
        processed_msg.orientation.w = q[3]

        # Copy covariance matrices
        processed_msg.angular_velocity_covariance = msg.angular_velocity_covariance
        processed_msg.linear_acceleration_covariance = msg.linear_acceleration_covariance
        processed_msg.orientation_covariance = msg.orientation_covariance

        # Publish processed data
        self.processed_imu_pub.publish(processed_msg)

        # Publish orientation separately
        orientation_msg = Quaternion()
        orientation_msg.x = q[0]
        orientation_msg.y = q[1]
        orientation_msg.z = q[2]
        orientation_msg.w = q[3]
        self.orientation_pub.publish(orientation_msg)

        # Log statistics periodically
        if self.measurement_count % 1000 == 0:
            rospy.loginfo(f"IMU Stats - Bias: [{self.angular_velocity_bias[0]:.6f}, "
                         f"{self.angular_velocity_bias[1]:.6f}, {self.angular_velocity_bias[2]:.6f}] rad/s")

if __name__ == '__main__':
    try:
        processor = IMUProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Creating a Launch File for IMU Simulation

Create a launch file `imu_simulation.launch` to run the complete simulation:

```xml
<launch>
  <!-- Start Gazebo with a world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find your_robot_description)/worlds/simple.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Spawn the robot with IMU -->
  <param name="robot_description" command="$(find xacro)/xacro --inorder '$(find your_robot_description)/urdf/robot_with_imu.xacro'" />

  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-param robot_description -urdf -model robot -x 0 -y 0 -z 0.1"
        respawn="false" output="screen"/>

  <!-- Start the IMU processor node -->
  <node name="imu_processor" pkg="your_robot_perception" type="imu_processor" output="screen">
    <param name="frame_id" value="imu_link"/>
  </node>

  <!-- Robot State Publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />

  <!-- RViz for visualization -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find your_robot_perception)/rviz/imu_config.rviz" required="true"/>

</launch>
```

## IMU Calibration and Validation

To ensure your IMU simulation matches real-world characteristics, you should calibrate and validate:

1. **Bias estimation**: Verify initial bias values and drift characteristics
2. **Noise characteristics**: Validate noise parameters against sensor specifications
3. **Scale factor accuracy**: Check for proper scaling of measurements
4. **Cross-axis sensitivity**: Verify minimal coupling between axes

Here's a validation script:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from collections import deque
import statistics

class IMUValidator:
    def __init__(self):
        rospy.init_node('imu_validator')

        # Subscribe to IMU topic
        self.imu_sub = rospy.Subscriber('/imu/data_raw', Imu, self.imu_callback)

        # Validation parameters
        self.stationary_threshold = 0.1  # Threshold for "stationary" state (m/s²)
        self.stationary_duration = 5.0   # Duration to consider as stationary (seconds)

        # Statistics collection
        self.measurement_count = 0
        self.acceleration_history = deque(maxlen=1000)
        self.angular_velocity_history = deque(maxlen=1000)
        self.stationary_start_time = None
        self.bias_estimates = {'x': [], 'y': [], 'z': []}

        rospy.loginfo("IMU Validator initialized")

    def imu_callback(self, msg):
        self.measurement_count += 1

        # Extract measurements
        acc = np.array([msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z])

        # Store measurements
        self.acceleration_history.append(acc)
        self.angular_velocity_history.append(gyro)

        # Check if the robot is stationary (based on low acceleration)
        acc_magnitude = np.linalg.norm(acc - np.array([0, 0, 9.81]))  # Account for gravity
        if acc_magnitude < self.stationary_threshold:
            if self.stationary_start_time is None:
                self.stationary_start_time = rospy.Time.now()
            elif (rospy.Time.now() - self.stationary_start_time).to_sec() > self.stationary_duration:
                # Calculate bias during stationary period
                if len(self.angular_velocity_history) > 100:
                    recent_gyro = np.array(list(self.angular_velocity_history)[-100:])
                    bias_x = np.mean(recent_gyro[:, 0])
                    bias_y = np.mean(recent_gyro[:, 1])
                    bias_z = np.mean(recent_gyro[:, 2])

                    self.bias_estimates['x'].append(bias_x)
                    self.bias_estimates['y'].append(bias_y)
                    self.bias_estimates['z'].append(bias_z)

                    rospy.loginfo(f"Estimated bias during stationary period: "
                                 f"[{bias_x:.6f}, {bias_y:.6f}, {bias_z:.6f}] rad/s")
        else:
            self.stationary_start_time = None

        # Calculate and report statistics periodically
        if self.measurement_count % 500 == 0:
            if len(self.acceleration_history) > 100:
                acc_array = np.array(list(self.acceleration_history))
                gyro_array = np.array(list(self.angular_velocity_history))

                # Calculate statistics
                acc_mean = np.mean(acc_array, axis=0)
                acc_std = np.std(acc_array, axis=0)
                gyro_mean = np.mean(gyro_array, axis=0)
                gyro_std = np.std(gyro_array, axis=0)

                rospy.loginfo(f"IMU Stats (last 500 samples):")
                rospy.loginfo(f"  Acc - Mean: [{acc_mean[0]:.3f}, {acc_mean[1]:.3f}, {acc_mean[2]:.3f}], "
                             f"Std: [{acc_std[0]:.3f}, {acc_std[1]:.3f}, {acc_std[2]:.3f}]")
                rospy.loginfo(f"  Gyro - Mean: [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}], "
                             f"Std: [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, {gyro_std[2]:.6f}]")

                # Calculate bias estimates if available
                if self.bias_estimates['x']:
                    avg_bias_x = sum(self.bias_estimates['x'][-5:]) / len(self.bias_estimates['x'][-5:])
                    avg_bias_y = sum(self.bias_estimates['y'][-5:]) / len(self.bias_estimates['y'][-5:])
                    avg_bias_z = sum(self.bias_estimates['z'][-5:]) / len(self.bias_estimates['z'][-5:])

                    rospy.loginfo(f"  Avg Bias: [{avg_bias_x:.6f}, {avg_bias_y:.6f}, {avg_bias_z:.6f}] rad/s")

if __name__ == '__main__':
    try:
        validator = IMUValidator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Performance Optimization for IMU Simulation

Simulating realistic IMUs requires careful consideration of computational efficiency:

1. **Update rate**: Balance between realism and performance (100-200 Hz typical)
2. **Noise computation**: Use efficient algorithms for noise generation
3. **Bias modeling**: Implement computationally efficient bias drift models
4. **Filtering**: Apply appropriate filtering to reduce noise while maintaining accuracy

## Troubleshooting Common Issues

### Issue: IMU data shows constant bias
- **Solution**: Check that noise and bias parameters are properly configured
- **Check**: Verify that the IMU is attached to a moving body in the simulation

### Issue: High-frequency noise in measurements
- **Solution**: Adjust noise parameters to match real sensor characteristics
- **Check**: Consider applying low-pass filtering to the output

### Issue: Orientation integration drifts significantly
- **Solution**: Implement proper sensor fusion with other sensors (e.g., magnetometer)
- **Check**: Verify that angular velocity integration is properly implemented

### Issue: IMU topic not publishing
- **Solution**: Check that the Gazebo plugin is loaded correctly
- **Check**: Verify that the IMU link is properly attached to the robot model

## Next Steps

In the next section, we'll explore sensor data processing examples in ROS, including filtering, calibration, and fusion techniques that combine data from multiple sensors like the LiDAR, depth camera, and IMU we've covered.

## Exercise

Create a custom IMU configuration that simulates a specific real-world sensor (e.g., MPU-9250) with its exact specifications:
- Accelerometer range: ±2g to ±16g (configurable)
- Gyroscope range: ±250°/s to ±2000°/s (configurable)
- Magnetometer (if needed): ±4800 µT
- Noise density: Accelerometer 150 µg/√Hz, Gyro 13.8 mDPS/√Hz
- Bias instability: Accelerometer 30 mg, Gyro 5 dps over 50 seconds

Implement proper bias drift models and validate your configuration by comparing the simulated noise characteristics with the real sensor's specifications.
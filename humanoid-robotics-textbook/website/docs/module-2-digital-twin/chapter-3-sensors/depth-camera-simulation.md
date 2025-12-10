---
title: Depth Camera Simulation
sidebar_position: 3
---

# Depth Camera Simulation

## Overview

Depth cameras are essential sensors in robotics that provide both color (RGB) and depth information for each pixel. In digital twin applications, realistic depth camera simulation is crucial for tasks like 3D reconstruction, object recognition, and environment mapping. This tutorial will guide you through setting up realistic depth camera simulation in Gazebo with proper noise models and distortion characteristics.

## Learning Objectives

After completing this tutorial, you will be able to:
- Configure realistic depth cameras in Gazebo with RGB-D capabilities
- Implement proper noise and distortion models for depth cameras
- Generate realistic depth images that match real-world sensor characteristics
- Process depth camera data in ROS for digital twin applications
- Validate simulated depth camera data against real sensor specifications

## Prerequisites

- Completed Chapter 1 (Gazebo Physics Simulation)
- Basic understanding of ROS image processing with cv_bridge
- Knowledge of camera models and projection matrices
- Understanding of RGB-D sensor formats and data processing

## Understanding Depth Cameras in Digital Twins

Depth cameras, such as the Microsoft Kinect or Intel RealSense, capture both color and depth information simultaneously. The depth data provides distance measurements for each pixel, enabling 3D scene reconstruction and spatial understanding. In digital twin applications, depth camera simulation must:

- Provide accurate depth measurements
- Include realistic noise and distortion characteristics
- Operate at appropriate frame rates and resolutions
- Generate data in standard formats (RGB, depth, point clouds)
- Integrate with existing ROS image processing pipelines

## Setting Up a Basic Depth Camera in Gazebo

### Creating a Depth Camera Model

Let's create a basic depth camera configuration in a Gazebo model file. Create a new file called `depth_camera.model`:

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="depth_camera">
    <link name="camera_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.001</iyy>
          <iyz>0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>

      <visual name="camera_visual">
        <geometry>
          <box>
            <size>0.05 0.05 0.05</size>
          </box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>

      <collision name="camera_collision">
        <geometry>
          <box>
            <size>0.05 0.05 0.05</size>
          </box>
        </geometry>
      </collision>

      <sensor name="depth_camera" type="depth">
        <pose>0 0 0 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10.0</far>
          </clip>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.007</stddev>
          </noise>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>

        <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
          <baseline>0.2</baseline>
          <alwaysOn>true</alwaysOn>
          <updateRate>30.0</updateRate>
          <cameraName>depth_camera</cameraName>
          <imageTopicName>/rgb/image_raw</imageTopicName>
          <depthImageTopicName>/depth/image_raw</depthImageTopicName>
          <pointCloudTopicName>/depth/points</pointCloudTopicName>
          <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
          <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
          <frameName>camera_link</frameName>
          <pointCloudCutoff>0.1</pointCloudCutoff>
          <pointCloudCutoffMax>10.0</pointCloudCutoffMax>
          <distortion_k1>0.0</distortion_k1>
          <distortion_k2>0.0</distortion_k2>
          <distortion_k3>0.0</distortion_k3>
          <distortion_t1>0.0</distortion_t1>
          <distortion_t2>0.0</distortion_t2>
          <CxPrime>0</CxPrime>
          <Cx>320.5</Cx>
          <Cy>240.5</Cy>
          <focalLength>525.0</focalLength>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

### Understanding the Configuration

- `horizontal_fov`: Field of view of the camera (60 degrees in this case)
- `image`: Resolution and format of the captured images (640x480 RGB)
- `clip`: Near and far clipping distances for depth measurement
- `update_rate`: Frame rate of the camera (30 Hz)
- `noise`: Gaussian noise characteristics for realistic depth measurements

## Advanced Depth Camera Configuration with Distortion Models

Real depth cameras have lens distortion that must be simulated for realistic digital twins. Let's create a more advanced configuration:

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="advanced_depth_camera">
    <link name="camera_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.002</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.002</iyy>
          <iyz>0</iyz>
          <izz>0.002</izz>
        </inertia>
      </inertial>

      <visual name="camera_visual">
        <geometry>
          <box>
            <size>0.08 0.12 0.04</size>
          </box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.15 0.15 0.15 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>

      <collision name="camera_collision">
        <geometry>
          <box>
            <size>0.08 0.12 0.04</size>
          </box>
        </geometry>
      </collision>

      <sensor name="realistic_depth_camera" type="depth">
        <pose>0 0 0 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
          <image>
            <width>1280</width>
            <height>720</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.3</near>
            <far>8.0</far>
          </clip>
          <distortion>
            <k1>-0.125</k1>
            <k2>0.188</k2>
            <k3>-0.063</k3>
            <p1>0.002</p1>
            <p2>-0.001</p2>
            <center>0.5 0.5</center>
          </distortion>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.005</stddev>
          </noise>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>

        <plugin name="advanced_camera_controller" filename="libgazebo_ros_openni_kinect.so">
          <baseline>0.1</baseline>
          <alwaysOn>true</alwaysOn>
          <updateRate>30.0</updateRate>
          <cameraName>advanced_depth_camera</cameraName>
          <imageTopicName>/camera/rgb/image_raw</imageTopicName>
          <depthImageTopicName>/camera/depth/image_raw</depthImageTopicName>
          <pointCloudTopicName>/camera/depth/points</pointCloudTopicName>
          <cameraInfoTopicName>/camera/rgb/camera_info</cameraInfoTopicName>
          <depthImageCameraInfoTopicName>/camera/depth/camera_info</depthImageCameraInfoTopicName>
          <frameName>camera_link</frameName>
          <pointCloudCutoff>0.3</pointCloudCutoff>
          <pointCloudCutoffMax>8.0</pointCloudCutoffMax>
          <distortion_k1>-0.125</distortion_k1>
          <distortion_k2>0.188</distortion_k2>
          <distortion_k3>-0.063</distortion_k3>
          <distortion_t1>0.002</distortion_t1>
          <distortion_t2>-0.001</distortion_t2>
          <CxPrime>0</CxPrime>
          <Cx>640.0</Cx>
          <Cy>360.0</Cy>
          <focalLength>525.0</focalLength>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

### Key Features of Advanced Configuration

- **Higher resolution**: 1280x720 for better detail
- **Lens distortion**: Radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients
- **Extended range**: 0.3m to 8m for broader applications
- **Reduced noise**: Lower noise standard deviation for better quality

## Creating a ROS Node for Depth Camera Data Processing

Let's create a ROS node that processes the simulated depth camera data:

```cpp
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

class DepthCameraProcessor {
public:
    DepthCameraProcessor(ros::NodeHandle& nh) : nh_(nh), it_(nh) {
        // Subscribe to RGB and depth topics
        rgb_sub_ = it_.subscribe("/camera/rgb/image_raw", 1,
                                &DepthCameraProcessor::rgbCallback, this);
        depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1,
                                  &DepthCameraProcessor::depthCallback, this);
        camera_info_sub_ = nh_.subscribe("/camera/rgb/camera_info", 1,
                                        &DepthCameraProcessor::cameraInfoCallback, this);

        // Publisher for processed data
        processed_rgb_pub_ = it_.advertise("/camera/rgb/processed", 1);
        processed_depth_pub_ = it_.advertise("/camera/depth/processed", 1);
        pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/camera/depth/points_filtered", 10);

        ROS_INFO("Depth Camera Processor initialized");
    }

private:
    ros::NodeHandle& nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber rgb_sub_;
    image_transport::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    image_transport::Publisher processed_rgb_pub_;
    image_transport::Publisher processed_depth_pub_;
    ros::Publisher pointcloud_pub_;

    cv::Mat camera_matrix_;
    bool camera_matrix_received_ = false;

    void rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            // Process RGB image (example: apply edge detection)
            cv::Mat processed_img;
            cv::Canny(cv_ptr->image, processed_img, 50, 150);

            // Convert back to ROS image and publish
            cv_bridge::CvImage out_msg;
            out_msg.header = msg->header;
            out_msg.encoding = sensor_msgs::image_encodings::MONO8;
            out_msg.image = processed_img;

            processed_rgb_pub_.publish(out_msg.toImageMsg());
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);

            // Process depth image (example: filter out invalid ranges)
            cv::Mat filtered_depth = cv_ptr->image.clone();

            // Set invalid depth values to 0 (or NaN)
            cv::Mat mask = filtered_depth < 0.1;  // Too close
            filtered_depth.setTo(0, mask);

            mask = filtered_depth > 8.0;  // Too far
            filtered_depth.setTo(0, mask);

            // Convert back to ROS image and publish
            cv_bridge::CvImage out_msg;
            out_msg.header = msg->header;
            out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            out_msg.image = filtered_depth;

            processed_depth_pub_.publish(out_msg.toImageMsg());
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        if (!camera_matrix_received_) {
            // Extract camera matrix from CameraInfo message
            camera_matrix_ = cv::Mat(3, 3, CV_64F);
            for (int i = 0; i < 9; ++i) {
                camera_matrix_.at<double>(i / 3, i % 3) = msg->K[i];
            }
            camera_matrix_received_ = true;
            ROS_INFO("Camera matrix received and stored");
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_camera_processor");
    ros::NodeHandle nh;

    DepthCameraProcessor processor(nh);

    ros::spin();

    return 0;
}
```

## Python Alternative for Depth Camera Processing

Here's a Python version of the depth camera processor:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

class DepthCameraProcessor:
    def __init__(self):
        rospy.init_node('depth_camera_processor')

        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()

        # Subscribe to RGB and depth topics
        self.rgb_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)

        # Publisher for processed data
        self.processed_rgb_pub = rospy.Publisher('/camera/rgb/processed', Image, queue_size=10)
        self.processed_depth_pub = rospy.Publisher('/camera/depth/processed', Image, queue_size=10)
        self.pointcloud_pub = rospy.Publisher('/camera/depth/points_filtered', Image, queue_size=10)

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.camera_info_received = False

        rospy.loginfo("Depth Camera Processor initialized")

    def rgb_callback(self, rgb_msg):
        """Process incoming RGB image data"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")

            # Process RGB image (example: apply edge detection)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Convert back to ROS image and publish
            processed_msg = self.bridge.cv2_to_imgmsg(edges, "mono8")
            processed_msg.header = rgb_msg.header
            self.processed_rgb_pub.publish(processed_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def depth_callback(self, depth_msg):
        """Process incoming depth image data"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            # Process depth image (example: filter out invalid ranges)
            filtered_depth = np.copy(cv_depth)

            # Set invalid depth values to 0 (or NaN)
            filtered_depth[filtered_depth < 0.1] = 0  # Too close
            filtered_depth[filtered_depth > 8.0] = 0  # Too far

            # Convert back to ROS image and publish
            processed_msg = self.bridge.cv2_to_imgmsg(filtered_depth, "32FC1")
            processed_msg.header = depth_msg.header
            self.processed_depth_pub.publish(processed_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def camera_info_callback(self, camera_info_msg):
        """Store camera calibration parameters"""
        if not self.camera_info_received:
            # Extract camera matrix and distortion coefficients
            self.camera_matrix = np.array(camera_info_msg.K).reshape(3, 3)
            self.distortion_coeffs = np.array(camera_info_msg.D)
            self.camera_info_received = True
            rospy.loginfo("Camera calibration parameters received")

if __name__ == '__main__':
    try:
        processor = DepthCameraProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Creating a Launch File for Depth Camera Simulation

Create a launch file `depth_camera_simulation.launch` to run the complete simulation:

```xml
<launch>
  <!-- Start Gazebo with a world that includes textured objects -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find your_robot_description)/worlds/depth_test.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Spawn the robot with depth camera -->
  <param name="robot_description" command="$(find xacro)/xacro --inorder '$(find your_robot_description)/urdf/robot_with_depth_camera.xacro'" />

  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-param robot_description -urdf -model robot -x 0 -y 0 -z 0.1"
        respawn="false" output="screen"/>

  <!-- Start the depth camera processor node -->
  <node name="depth_camera_processor" pkg="your_robot_perception" type="depth_camera_processor" output="screen">
    <param name="frame_id" value="camera_link"/>
  </node>

  <!-- Image processing node -->
  <node name="image_proc" pkg="image_proc" type="image_proc" ns="camera" />

  <!-- RViz for visualization -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find your_robot_perception)/rviz/depth_camera_config.rviz" required="true"/>

</launch>
```

## Depth Camera Calibration and Validation

To ensure your depth camera simulation matches real-world characteristics, you should calibrate and validate:

1. **Intrinsic calibration**: Verify focal length, principal point, and distortion coefficients
2. **Depth accuracy**: Validate distance measurements against known objects
3. **Color fidelity**: Check RGB color reproduction accuracy
4. **Temporal consistency**: Ensure stable frame rates and timing

Here's a validation script:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

class DepthCameraValidator:
    def __init__(self):
        rospy.init_node('depth_camera_validator')

        self.bridge = CvBridge()

        # Subscribe to depth and camera info topics
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)

        # Validation parameters
        self.known_distance = 2.0  # Known distance to test object (meters)
        self.distance_tolerance = 0.05  # Acceptable tolerance (meters)

        # Statistics collection
        self.frame_count = 0
        self.distance_errors = []
        self.depth_variance = []

        # Camera parameters
        self.camera_info_received = False

        rospy.loginfo("Depth Camera Validator initialized")

    def depth_callback(self, depth_msg):
        """Validate the accuracy of depth camera measurements"""
        self.frame_count += 1

        try:
            # Convert ROS Image message to OpenCV image
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            # Get the center pixel value as a test
            height, width = cv_depth.shape
            center_x, center_y = width // 2, height // 2
            center_depth = cv_depth[center_y, center_x]

            if not np.isnan(center_depth) and center_depth > 0:
                error = abs(center_depth - self.known_distance)
                self.distance_errors.append(error)

                # Calculate variance in a small region around the center
                region = cv_depth[center_y-10:center_y+10, center_x-10:center_x+10]
                valid_pixels = region[~np.isnan(region) & (region > 0)]
                if len(valid_pixels) > 0:
                    self.depth_variance.append(np.var(valid_pixels))

                # Check if the measurement is within tolerance
                if error > self.distance_tolerance:
                    rospy.logwarn(f"Depth camera error: {error:.3f}m (tolerance: {self.distance_tolerance}m)")
                else:
                    rospy.loginfo_throttle(5.0, f"Depth camera accurate: {center_depth:.3f}m")

        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

        # Print statistics periodically
        if self.frame_count % 50 == 0:
            if self.distance_errors:
                mean_error = np.mean(self.distance_errors[-50:])  # Last 50 frames
                std_error = np.std(self.distance_errors[-50:])
                mean_variance = np.mean(self.depth_variance[-50:]) if self.depth_variance else 0
                rospy.loginfo(f"Validation stats (last 50 frames): Mean error: {mean_error:.3f}m, "
                              f"Std: {std_error:.3f}m, Mean variance: {mean_variance:.6f}")

    def camera_info_callback(self, camera_info_msg):
        """Process camera calibration information"""
        if not self.camera_info_received:
            # Extract camera matrix
            camera_matrix = np.array(camera_info_msg.K).reshape(3, 3)
            distortion_coeffs = np.array(camera_info_msg.D)

            rospy.loginfo(f"Camera matrix: {camera_matrix}")
            rospy.loginfo(f"Distortion coefficients: {distortion_coeffs}")
            self.camera_info_received = True

if __name__ == '__main__':
    try:
        validator = DepthCameraValidator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Performance Optimization for Depth Camera Simulation

Simulating high-resolution depth cameras can be computationally expensive. Here are optimization strategies:

1. **Reduce resolution**: Use lower resolution when high detail isn't required
2. **Adjust update rate**: Lower frame rate for less demanding applications
3. **Optimize distortion**: Use simpler distortion models when possible
4. **Limit range**: Use minimum required range for your application
5. **Use efficient encoding**: Choose appropriate image formats for your processing pipeline

## Troubleshooting Common Issues

### Issue: Depth images show all zeros or invalid values
- **Solution**: Check near/far clipping distances in the configuration
- **Check**: Verify that objects are within the camera's range

### Issue: High CPU/GPU usage
- **Solution**: Reduce image resolution or frame rate
- **Check**: Monitor Gazebo performance metrics

### Issue: RGB and depth images not synchronized
- **Solution**: Check timing parameters and buffer sizes
- **Check**: Use message_filters to synchronize topics

### Issue: Distortion parameters not applied
- **Solution**: Verify that distortion parameters are correctly specified in both SDF and plugin
- **Check**: Ensure the camera_info topic is being published with correct calibration

## Next Steps

In the next section, we'll explore IMU simulation with realistic characteristics and sensor fusion techniques. The concepts learned here about sensor modeling, noise characteristics, and ROS integration will apply to other sensor types as well.

## Exercise

Create a custom depth camera configuration that simulates a specific real-world sensor (e.g., Intel RealSense D435) with its exact specifications:
- Resolution: 1280x720 RGB, 1280x720 depth
- Field of view: 69° horizontal, 42° vertical for RGB; 87° horizontal, 58° vertical for depth
- Operating range: 0.2m to 10m
- Noise characteristics matching the real sensor

Test your configuration by comparing the simulated depth accuracy and field of view with the real sensor's specifications.
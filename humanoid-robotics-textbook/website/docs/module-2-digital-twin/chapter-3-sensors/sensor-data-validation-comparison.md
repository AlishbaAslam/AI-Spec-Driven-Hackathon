---
title: Sensor Data Validation and Comparison
sidebar_position: 8
---

# Sensor Data Validation and Comparison

## Overview

Sensor data validation is critical for ensuring the reliability and accuracy of digital twin systems. This chapter provides comprehensive code examples for validating sensor data quality, comparing simulated vs. real sensor characteristics, and implementing validation techniques for LiDAR, depth camera, and IMU sensors.

## Learning Objectives

After completing this chapter, you will be able to:
- Implement validation techniques for different sensor types
- Compare simulated sensor data with real-world characteristics
- Detect and handle sensor failures or degraded performance
- Create automated validation pipelines for sensor systems
- Evaluate sensor fusion quality and consistency

## Prerequisites

- Understanding of sensor simulation concepts from previous chapters
- Basic knowledge of statistical analysis and probability
- Experience with ROS message types and processing
- Familiarity with sensor characteristics and specifications

## LiDAR Data Validation

### Range Validation and Outlier Detection

```cpp
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <vector>
#include <algorithm>
#include <cmath>

class LidarValidator {
public:
    LidarValidator(ros::NodeHandle& nh) : nh_(nh) {
        scan_sub_ = nh_.subscribe("/laser_scan", 10, &LidarValidator::scanCallback, this);
        validated_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/validated_scan", 10);

        // Parameters for validation
        nh_.param<double>("min_range", min_range_, 0.1);
        nh_.param<double>("max_range", max_range_, 30.0);
        nh_.param<double>("max_jump", max_jump_, 1.0);

        ROS_INFO("LiDAR Validator initialized");
    }

private:
    ros::NodeHandle& nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher validated_pub_;

    double min_range_, max_range_, max_jump_;
    std::vector<double> recent_ranges_;

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        sensor_msgs::LaserScan validated_msg = *msg;

        // Validate individual range measurements
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            double range = msg->ranges[i];

            // Check range bounds
            if (std::isnan(range) || std::isinf(range) ||
                range < msg->range_min || range > msg->range_max) {
                validated_msg.ranges[i] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            // Check for unrealistic jumps between adjacent measurements
            if (i > 0 && i < msg->ranges.size() - 1) {
                double prev_range = msg->ranges[i-1];
                double next_range = msg->ranges[i+1];

                if (!std::isnan(prev_range) && !std::isinf(prev_range) &&
                    !std::isnan(next_range) && !std::isinf(next_range)) {

                    double jump_to_prev = std::abs(range - prev_range);
                    double jump_to_next = std::abs(range - next_range);

                    // If this measurement is significantly different from both neighbors,
                    // it might be an outlier
                    if (jump_to_prev > max_jump_ && jump_to_next > max_jump_) {
                        validated_msg.ranges[i] = (prev_range + next_range) / 2.0;
                        ROS_WARN_THROTTLE(1.0, "LiDAR outlier detected and corrected at index %zu", i);
                    }
                }
            }
        }

        validated_pub_.publish(validated_msg);

        // Update statistics
        updateRangeStatistics(validated_msg);
    }

    void updateRangeStatistics(const sensor_msgs::LaserScan& msg) {
        // Collect valid ranges for statistical analysis
        std::vector<double> valid_ranges;
        for (double range : msg.ranges) {
            if (!std::isnan(range) && !std::isinf(range) &&
                range >= msg.range_min && range <= msg.range_max) {
                valid_ranges.push_back(range);
            }
        }

        if (valid_ranges.empty()) return;

        // Calculate statistics
        double mean = std::accumulate(valid_ranges.begin(), valid_ranges.end(), 0.0) / valid_ranges.size();
        double variance = 0.0;
        for (double val : valid_ranges) {
            variance += (val - mean) * (val - mean);
        }
        variance /= valid_ranges.size();
        double std_dev = std::sqrt(variance);

        ROS_INFO_THROTTLE(2.0, "LiDAR Stats - Mean: %.2f, Std Dev: %.2f, Valid Points: %zu",
                         mean, std_dev, valid_ranges.size());
    }
};
```

### Point Cloud Density Validation

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from collections import defaultdict

class PointCloudValidator:
    def __init__(self):
        rospy.init_node('pointcloud_validator')

        self.pc_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.pc_callback)

        # Validation parameters
        self.min_points = 100
        self.min_density = 0.01  # points per cubic meter
        self.grid_resolution = 1.0  # 1m grid cells for density check

        rospy.loginfo("Point Cloud Validator initialized")

    def pc_callback(self, msg):
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        if len(points) == 0:
            rospy.logwarn("No valid points in point cloud")
            return

        points = np.array(points)

        # Validate point count
        if len(points) < self.min_points:
            rospy.logwarn(f"Point cloud has only {len(points)} points, less than minimum {self.min_points}")

        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        volume = np.prod(max_coords - min_coords)

        if volume > 0:
            density = len(points) / volume
            if density < self.min_density:
                rospy.logwarn(f"Point cloud density too low: {density:.4f} points/m³")
            else:
                rospy.loginfo_throttle(5.0, f"Point cloud density: {density:.4f} points/m³")

        # Validate spatial distribution using grid-based approach
        self.validate_spatial_distribution(points, min_coords, max_coords)

        # Check for extreme outliers
        self.check_outliers(points)

    def validate_spatial_distribution(self, points, min_coords, max_coords):
        """Validate that points are reasonably distributed in space"""
        # Create 3D grid and count points per cell
        grid_size = np.ceil((max_coords - min_coords) / self.grid_resolution).astype(int)
        grid_size = np.maximum(grid_size, 1)  # Ensure at least 1 in each dimension

        if np.any(grid_size > 1000):  # Prevent excessive memory usage
            rospy.logwarn("Grid too large, skipping spatial distribution check")
            return

        grid_counts = np.zeros(grid_size)

        for point in points:
            grid_idx = ((point - min_coords) / self.grid_resolution).astype(int)
            # Clamp indices to grid bounds
            grid_idx = np.clip(grid_idx, 0, np.array(grid_size) - 1)
            grid_counts[tuple(grid_idx)] += 1

        # Calculate statistics on grid cell occupancy
        occupied_cells = np.count_nonzero(grid_counts)
        total_cells = np.prod(grid_size)
        occupancy_ratio = occupied_cells / total_cells if total_cells > 0 else 0

        if occupancy_ratio < 0.01:  # Less than 1% of cells occupied
            rospy.logwarn(f"Sparse point distribution: only {occupancy_ratio*100:.2f}% of space occupied")
        else:
            rospy.loginfo_throttle(5.0, f"Point distribution: {occupancy_ratio*100:.2f}% of space occupied")

    def check_outliers(self, points):
        """Check for extreme outliers using statistical methods"""
        # Use modified Z-score method (more robust to outliers)
        median = np.median(points, axis=0)
        mad = np.median(np.abs(points - median), axis=0)  # Median absolute deviation

        # Avoid division by zero
        mad = np.where(mad == 0, 1e-8, mad)

        modified_z_scores = 0.6745 * (points - median) / mad
        outlier_mask = np.any(np.abs(modified_z_scores) > 3.5, axis=1)

        outlier_count = np.sum(outlier_mask)
        if outlier_count > 0:
            outlier_ratio = outlier_count / len(points)
            if outlier_ratio > 0.1:  # More than 10% are outliers
                rospy.logwarn(f"High outlier ratio: {outlier_ratio*100:.2f}% of points are outliers")
            else:
                rospy.loginfo_throttle(5.0, f"Outlier ratio: {outlier_ratio*100:.2f}%")
```

## Depth Camera Validation

### Image Quality Assessment

```cpp
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

class DepthCameraValidator {
public:
    DepthCameraValidator(ros::NodeHandle& nh) : nh_(nh) {
        rgb_sub_ = nh_.subscribe("/camera/rgb/image_raw", 10,
                                &DepthCameraValidator::rgbCallback, this);
        depth_sub_ = nh_.subscribe("/camera/depth/image_raw", 10,
                                  &DepthCameraValidator::depthCallback, this);
        camera_info_sub_ = nh_.subscribe("/camera/rgb/camera_info", 10,
                                        &DepthCameraValidator::cameraInfoCallback, this);

        // Validation parameters
        nh_.param<double>("min_depth", min_depth_, 0.1);
        nh_.param<double>("max_depth", max_depth_, 10.0);
        nh_.param<double>("min_brightness", min_brightness_, 20.0);
        nh_.param<double>("max_brightness", max_brightness_, 220.0);

        ROS_INFO("Depth Camera Validator initialized");
    }

private:
    ros::NodeHandle& nh_;
    ros::Subscriber rgb_sub_, depth_sub_, camera_info_sub_;

    double min_depth_, max_depth_;
    double min_brightness_, max_brightness_;
    bool camera_info_received_ = false;
    cv::Mat camera_matrix_;

    void rgbCallback(const sensor_msgs::Image::ConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            // Convert to grayscale for brightness analysis
            cv::Mat gray;
            cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

            // Calculate image statistics
            cv::Scalar mean_brightness = cv::mean(gray);
            cv::Mat stddev;
            cv::meanStdDev(gray, cv::noArray(), stddev);

            double brightness = mean_brightness[0];
            double std_dev = stddev.at<double>(0, 0);

            // Validate brightness
            if (brightness < min_brightness_ || brightness > max_brightness_) {
                ROS_WARN_THROTTLE(2.0, "RGB image brightness out of range: %.2f (min: %.2f, max: %.2f)",
                                 brightness, min_brightness_, max_brightness_);
            }

            // Check for overexposure (too many pixels at max value)
            cv::Mat overexposed_mask;
            cv::inRange(gray, 250, 255, overexposed_mask);
            double overexposed_ratio = cv::countNonZero(overexposed_mask) / (double)(gray.rows * gray.cols);

            if (overexposed_ratio > 0.1) {  // More than 10% overexposed
                ROS_WARN_THROTTLE(2.0, "RGB image overexposed: %.2f%% pixels at max brightness",
                                 overexposed_ratio * 100);
            }

            // Check for underexposure (too many pixels at min value)
            cv::Mat underexposed_mask;
            cv::inRange(gray, 0, 5, underexposed_mask);
            double underexposed_ratio = cv::countNonZero(underexposed_mask) / (double)(gray.rows * gray.cols);

            if (underexposed_ratio > 0.3) {  // More than 30% underexposed
                ROS_WARN_THROTTLE(2.0, "RGB image underexposed: %.2f%% pixels at min brightness",
                                 underexposed_ratio * 100);
            }

            // Calculate image sharpness using Laplacian variance
            cv::Mat laplacian;
            cv::Laplacian(gray, laplacian, CV_64F);
            cv::Scalar mu, sigma;
            cv::meanStdDev(laplacian, mu, sigma);
            double sharpness = sigma[0] * sigma[0];

            if (sharpness < 100) {  // Threshold for blur detection
                ROS_WARN_THROTTLE(2.0, "RGB image appears blurry: sharpness score = %.2f", sharpness);
            } else {
                ROS_INFO_THROTTLE(5.0, "RGB image sharpness: %.2f", sharpness);
            }

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception in RGB callback: %s", e.what());
        }
    }

    void depthCallback(const sensor_msgs::Image::ConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);

            // Count valid depth values
            cv::Mat valid_mask = cv_ptr->image > 0;
            int valid_count = cv::countNonZero(valid_mask);
            int total_count = cv_ptr->image.rows * cv_ptr->image.cols;
            double valid_ratio = (double)valid_count / total_count;

            if (valid_ratio < 0.1) {  // Less than 10% valid depth values
                ROS_WARN_THROTTLE(2.0, "Depth image has low valid data: %.2f%%", valid_ratio * 100);
            }

            // Calculate depth statistics
            cv::Mat valid_depths;
            cv_ptr->image.copyTo(valid_depths, valid_mask);

            if (valid_count > 0) {
                cv::Scalar mean_depth = cv::mean(valid_depths);
                cv::Mat stddev;
                cv::meanStdDev(valid_depths, cv::noArray(), stddev);

                double avg_depth = mean_depth[0];
                double depth_std = stddev.at<double>(0, 0);

                // Validate depth range
                if (avg_depth < min_depth_ || avg_depth > max_depth_) {
                    ROS_WARN_THROTTLE(2.0, "Average depth out of expected range: %.2f (min: %.2f, max: %.2f)",
                                     avg_depth, min_depth_, max_depth_);
                }

                // Check for unrealistic depth variance
                if (depth_std > max_depth_ * 0.5) {  // Standard deviation too high
                    ROS_WARN_THROTTLE(2.0, "High depth variance: %.2f, may indicate sensor issues", depth_std);
                }

                ROS_INFO_THROTTLE(5.0, "Depth stats - Mean: %.2f, Std: %.2f, Valid: %.2f%%",
                                 avg_depth, depth_std, valid_ratio * 100);
            }

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception in depth callback: %s", e.what());
        }
    }

    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        if (!camera_info_received_) {
            // Extract camera matrix
            camera_matrix_ = cv::Mat(3, 3, CV_64F);
            for (int i = 0; i < 9; ++i) {
                camera_matrix_.at<double>(i / 3, i % 3) = msg->K[i];
            }
            camera_info_received_ = true;
            ROS_INFO("Camera info received - focal length: (%.2f, %.2f)",
                     camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1));
        }
    }
};
```

## IMU Validation

### Sensor Consistency Checks

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_multiply, quaternion_conjugate
from collections import deque

class IMUValidator:
    def __init__(self):
        rospy.init_node('imu_validator')

        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # Validation parameters
        self.angular_velocity_threshold = 10.0  # rad/s
        self.linear_acceleration_threshold = 50.0  # m/s²
        self.gravity_magnitude = 9.81  # m/s²
        self.gravity_tolerance = 1.0  # m/s²

        # Historical data for consistency checks
        self.angular_velocity_history = deque(maxlen=50)
        self.linear_acceleration_history = deque(maxlen=50)
        self.orientation_history = deque(maxlen=10)

        # Bias estimation
        self.angular_velocity_bias = np.array([0.0, 0.0, 0.0])
        self.bias_estimation_samples = 0
        self.max_bias_samples = 1000  # Maximum samples for bias calculation

        rospy.loginfo("IMU Validator initialized")

    def imu_callback(self, msg):
        # Extract measurements
        ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        lin_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

        # Store for history
        self.angular_velocity_history.append(ang_vel.copy())
        self.linear_acceleration_history.append(lin_acc.copy())
        self.orientation_history.append(orientation.copy())

        # Validate angular velocity magnitude
        ang_vel_mag = np.linalg.norm(ang_vel)
        if ang_vel_mag > self.angular_velocity_threshold:
            rospy.logwarn(f"High angular velocity: {ang_vel_mag:.2f} rad/s")

        # Validate linear acceleration magnitude (excluding gravity)
        # Assuming z-axis is up, subtract gravity from z component
        gravity_corrected = lin_acc.copy()
        gravity_corrected[2] -= self.gravity_magnitude

        lin_acc_mag = np.linalg.norm(gravity_corrected)
        if lin_acc_mag > self.linear_acceleration_threshold:
            rospy.logwarn(f"High linear acceleration: {lin_acc_mag:.2f} m/s²")

        # Check for stationary periods to estimate bias
        if ang_vel_mag < 0.1 and lin_acc_mag < 1.0:  # Likely stationary
            if self.bias_estimation_samples < self.max_bias_samples:
                self.angular_velocity_bias = (
                    (self.bias_estimation_samples * self.angular_velocity_bias + ang_vel) /
                    (self.bias_estimation_samples + 1)
                )
                self.bias_estimation_samples += 1

                if self.bias_estimation_samples % 100 == 0:
                    rospy.loginfo(f"Estimated IMU bias: {self.angular_velocity_bias}")

        # Validate orientation consistency
        self.validate_orientation(orientation)

        # Check consistency between measurements
        self.check_measurement_consistency(ang_vel, lin_acc, orientation)

        # Update statistics periodically
        if len(self.angular_velocity_history) == 50:
            self.calculate_statistics()

    def validate_orientation(self, orientation):
        """Validate quaternion normalization and consistency"""
        # Check quaternion normalization
        norm = np.linalg.norm(orientation)
        if abs(norm - 1.0) > 0.01:
            rospy.logwarn(f"Orientation quaternion not normalized: {norm:.4f}")

        # Check for orientation jumps (if we have previous orientation)
        if len(self.orientation_history) > 1:
            prev_orientation = self.orientation_history[-2]

            # Calculate the dot product to check for large orientation changes
            dot_product = abs(np.dot(orientation, prev_orientation))

            # If dot product is close to 1, orientations are similar
            # If close to 0, they are perpendicular
            # If close to -1, they are opposite (which shouldn't happen in continuous motion)
            if dot_product < 0.7:  # Large change (about 45 degrees)
                rospy.logwarn(f"Large orientation change detected: dot product = {dot_product:.3f}")

    def check_measurement_consistency(self, ang_vel, lin_acc, orientation):
        """Check consistency between different IMU measurements"""
        # Check if linear acceleration is consistent with gravity when stationary
        ang_vel_mag = np.linalg.norm(ang_vel)
        lin_acc_mag = np.linalg.norm(lin_acc)

        if ang_vel_mag < 0.1:  # Stationary or near-stationary
            gravity_diff = abs(lin_acc_mag - self.gravity_magnitude)
            if gravity_diff > self.gravity_tolerance:
                rospy.logwarn(f"Linear acceleration not consistent with gravity: {lin_acc_mag:.2f} m/s²")

    def calculate_statistics(self):
        """Calculate and report IMU statistics"""
        if len(self.angular_velocity_history) == 0 or len(self.linear_acceleration_history) == 0:
            return

        # Convert to numpy arrays for statistics
        ang_vel_array = np.array(self.angular_velocity_history)
        lin_acc_array = np.array(self.linear_acceleration_history)

        # Calculate statistics
        ang_vel_mean = np.mean(ang_vel_array, axis=0)
        ang_vel_std = np.std(ang_vel_array, axis=0)
        ang_vel_max = np.max(np.abs(ang_vel_array), axis=0)

        lin_acc_mean = np.mean(lin_acc_array, axis=0)
        lin_acc_std = np.std(lin_acc_array, axis=0)
        lin_acc_max = np.max(np.abs(lin_acc_array), axis=0)

        rospy.loginfo_throttle(
            5.0,  # Log every 5 seconds
            "IMU Stats - Ang.Vel: mean[%.3f,%.3f,%.3f], std[%.4f,%.4f,%.4f], max[%.2f,%.2f,%.2f]\n"
            "          Lin.Acc: mean[%.2f,%.2f,%.2f], std[%.2f,%.2f,%.2f], max[%.2f,%.2f,%.2f]",
            ang_vel_mean[0], ang_vel_mean[1], ang_vel_mean[2],
            ang_vel_std[0], ang_vel_std[1], ang_vel_std[2],
            ang_vel_max[0], ang_vel_max[1], ang_vel_max[2],
            lin_acc_mean[0], lin_acc_mean[1], lin_acc_mean[2],
            lin_acc_std[0], lin_acc_std[1], lin_acc_std[2],
            lin_acc_max[0], lin_acc_max[1], lin_acc_max[2]
        )
```

## Multi-Sensor Validation and Comparison

### Cross-Sensor Consistency Checking

```cpp
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class MultiSensorValidator {
public:
    MultiSensorValidator(ros::NodeHandle& nh) : nh_(nh), tf_listener_(ros::Duration(10.0)) {
        lidar_sub_ = nh_.subscribe("/laser_scan", 10, &MultiSensorValidator::lidarCallback, this);
        depth_sub_ = nh_.subscribe("/camera/depth/image_raw", 10, &MultiSensorValidator::depthCallback, this);
        imu_sub_ = nh_.subscribe("/imu/data", 100, &MultiSensorValidator::imuCallback, this);
        pose_sub_ = nh_.subscribe("/fused_pose", 10, &MultiSensorValidator::poseCallback, this);

        // Parameters for validation
        nh_.param<double>("max_pose_deviation", max_pose_deviation_, 1.0);
        nh_.param<double>("time_sync_tolerance", time_sync_tolerance_, 0.1);

        ROS_INFO("Multi-Sensor Validator initialized");
    }

private:
    ros::NodeHandle& nh_;
    ros::Subscriber lidar_sub_, depth_sub_, imu_sub_, pose_sub_;
    tf::TransformListener tf_listener_;

    double max_pose_deviation_, time_sync_tolerance_;

    // Storage for sensor data with timestamps
    sensor_msgs::LaserScan::ConstPtr last_lidar_;
    sensor_msgs::Image::ConstPtr last_depth_;
    sensor_msgs::Imu::ConstPtr last_imu_;
    geometry_msgs::PoseStamped::ConstPtr last_pose_;

    ros::Time last_validation_time_;

    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        last_lidar_ = msg;
        validateMultiSensorConsistency();
    }

    void depthCallback(const sensor_msgs::Image::ConstPtr& msg) {
        last_depth_ = msg;
        validateMultiSensorConsistency();
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        last_imu_ = msg;
        validateMultiSensorConsistency();
    }

    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        last_pose_ = msg;
        validateMultiSensorConsistency();
    }

    void validateMultiSensorConsistency() {
        // Only validate if we have all sensor types and sufficient time has passed
        ros::Time now = ros::Time::now();
        if ((now - last_validation_time_).toSec() < 1.0) return;  // Validate at 1Hz

        if (!last_lidar_ || !last_depth_ || !last_imu_ || !last_pose_) return;

        // Check temporal consistency (all data should be recent)
        ros::Time latest_time = std::max({last_lidar_->header.stamp,
                                         last_depth_->header.stamp,
                                         last_imu_->header.stamp,
                                         last_pose_->header.stamp});
        ros::Time earliest_time = std::min({last_lidar_->header.stamp,
                                          last_depth_->header.stamp,
                                          last_imu_->header.stamp,
                                          last_pose_->header.stamp});

        double time_diff = (latest_time - earliest_time).toSec();
        if (time_diff > time_sync_tolerance_) {
            ROS_WARN_THROTTLE(2.0, "Sensor data time sync issue: %.3fs between oldest and newest", time_diff);
        }

        // Validate pose consistency with IMU orientation
        validatePoseImuConsistency();

        // Validate depth consistency with LiDAR measurements
        validateDepthLidarConsistency();

        last_validation_time_ = now;
    }

    void validatePoseImuConsistency() {
        // Compare pose orientation with IMU orientation
        if (!last_pose_ || !last_imu_) return;

        // Extract quaternions
        tf::Quaternion pose_quat(last_pose_->pose.orientation.x,
                                last_pose_->pose.orientation.y,
                                last_pose_->pose.orientation.z,
                                last_pose_->pose.orientation.w);
        tf::Quaternion imu_quat(last_imu_->orientation.x,
                               last_imu_->orientation.y,
                               last_imu_->orientation.z,
                               last_imu_->orientation.w);

        // Calculate angle difference between orientations
        tf::Quaternion diff_quat = pose_quat * imu_quat.inverse();
        double angle_diff = diff_quat.getAngle();

        if (angle_diff > M_PI) angle_diff = 2 * M_PI - angle_diff;  // Handle wrap-around

        if (angle_diff > 0.5) {  // More than ~28 degrees difference
            ROS_WARN_THROTTLE(2.0, "Pose-IMU orientation inconsistency: %.2f degrees",
                             angle_diff * 180.0 / M_PI);
        } else {
            ROS_INFO_THROTTLE(10.0, "Pose-IMU consistency: %.2f degrees",
                             angle_diff * 180.0 / M_PI);
        }
    }

    void validateDepthLidarConsistency() {
        // Compare depth camera and LiDAR measurements for consistency
        if (!last_depth_ || !last_lidar_) return;

        try {
            cv_bridge::CvImagePtr cv_depth = cv_bridge::toCvCopy(last_depth_, sensor_msgs::image_encodings::TYPE_32FC1);

            // Sample depth values at locations corresponding to LiDAR beams
            // This is a simplified approach - in practice, you'd need to transform coordinates
            int center_x = cv_depth->image.cols / 2;
            int center_y = cv_depth->image.rows / 2;

            float center_depth = cv_depth->image.at<float>(center_y, center_x);

            // Compare with LiDAR center reading (assuming center beam)
            size_t center_idx = last_lidar_->ranges.size() / 2;
            float lidar_distance = last_lidar_->ranges[center_idx];

            if (!std::isnan(center_depth) && !std::isnan(lidar_distance)) {
                float diff = std::abs(center_depth - lidar_distance);

                // Consider them consistent if within 10% or 0.1m
                float threshold = std::max(0.1f, 0.1f * std::max(center_depth, lidar_distance));

                if (diff > threshold) {
                    ROS_WARN_THROTTLE(2.0, "Depth-LiDAR inconsistency: depth=%.2f, lidar=%.2f, diff=%.2f",
                                     center_depth, lidar_distance, diff);
                } else {
                    ROS_INFO_THROTTLE(10.0, "Depth-LiDAR consistency: diff=%.3f", diff);
                }
            }

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception in depth-LiDAR validation: %s", e.what());
        }
    }
};
```

## Statistical Validation and Performance Metrics

### Comprehensive Validation Node

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import sensor_msgs.point_cloud2 as pc2
from collections import defaultdict, deque
import time

class ComprehensiveValidator:
    def __init__(self):
        rospy.init_node('comprehensive_validator')

        # Subscribe to all sensor types
        self.lidar_sub = rospy.Subscriber('/laser_scan', LaserScan, self.lidar_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.pose_sub = rospy.Subscriber('/fused_pose', PoseStamped, self.pose_callback)

        # Publishers for validation metrics
        self.lidar_quality_pub = rospy.Publisher('/lidar/quality_score', Float32, queue_size=10)
        self.camera_quality_pub = rospy.Publisher('/camera/quality_score', Float32, queue_size=10)
        self.imu_quality_pub = rospy.Publisher('/imu/quality_score', Float32, queue_size=10)
        self.system_health_pub = rospy.Publisher('/system/health_score', Float32, queue_size=10)

        # Validation parameters
        self.validation_window = 100  # Number of samples for statistical validation

        # Data storage
        self.lidar_data = deque(maxlen=self.validation_window)
        self.depth_data = deque(maxlen=self.validation_window)
        self.imu_data = deque(maxlen=self.validation_window)
        self.pose_data = deque(maxlen=self.validation_window)

        # Validation thresholds
        self.lidar_thresholds = {
            'min_valid_ratio': 0.7,
            'max_noise_std': 0.1,
            'min_range_stability': 0.95
        }

        self.camera_thresholds = {
            'min_brightness': 30,
            'max_brightness': 220,
            'min_sharpness': 50,
            'max_blur_ratio': 0.1
        }

        self.imu_thresholds = {
            'max_bias_drift': 0.01,  # rad/s per minute
            'max_noise_std': 0.01,
            'min_stability_ratio': 0.9
        }

        # Timing for performance metrics
        self.start_time = rospy.Time.now()

        rospy.loginfo("Comprehensive Validator initialized")

    def lidar_callback(self, msg):
        # Calculate quality metrics for LiDAR data
        valid_ranges = [r for r in msg.ranges if not (np.isnan(r) or np.isinf(r))]
        valid_ratio = len(valid_ranges) / len(msg.ranges) if msg.ranges else 0

        # Calculate range stability (consistency across multiple scans)
        range_stability = self.calculate_range_stability(valid_ranges)

        # Calculate noise level
        noise_std = self.estimate_lidar_noise(valid_ranges)

        # Store data for statistical analysis
        self.lidar_data.append({
            'timestamp': msg.header.stamp,
            'valid_ratio': valid_ratio,
            'range_stability': range_stability,
            'noise_std': noise_std
        })

        # Calculate quality score
        quality_score = self.calculate_lidar_quality_score()
        self.lidar_quality_pub.publish(Float32(quality_score))

        # Log warnings if quality is poor
        if quality_score < 0.7:
            rospy.logwarn(f"LiDAR quality score is low: {quality_score:.2f}")

    def depth_callback(self, msg):
        try:
            # Convert to OpenCV image
            import cv2
            from cv_bridge import CvBridge
            bridge = CvBridge()

            cv_image = bridge.imgmsg_to_cv2(msg, "32FC1")

            # Calculate brightness statistics
            valid_pixels = cv_image[~np.isnan(cv_image) & (cv_image > 0)]
            if len(valid_pixels) > 0:
                mean_brightness = np.mean(valid_pixels)
                std_brightness = np.std(valid_pixels)
            else:
                mean_brightness = 0
                std_brightness = 0

            # Calculate sharpness using Laplacian
            laplacian = cv2.Laplacian(cv_image, cv2.CV_64F)
            sharpness = cv2.meanStdDev(laplacian)[1][0][0] ** 2

            # Calculate blur ratio (percentage of low-contrast pixels)
            grad_x = cv2.Sobel(cv_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(cv_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            blur_ratio = np.sum(gradient_magnitude < 5) / gradient_magnitude.size

            # Store data
            self.depth_data.append({
                'timestamp': msg.header.stamp,
                'mean_brightness': mean_brightness,
                'sharpness': sharpness,
                'blur_ratio': blur_ratio
            })

            # Calculate quality score
            quality_score = self.calculate_camera_quality_score()
            self.camera_quality_pub.publish(Float32(quality_score))

        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

    def imu_callback(self, msg):
        # Extract IMU data
        ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        lin_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        # Calculate statistics
        ang_vel_norm = np.linalg.norm(ang_vel)
        lin_acc_norm = np.linalg.norm(lin_acc)

        # Store data
        self.imu_data.append({
            'timestamp': msg.header.stamp,
            'ang_vel_norm': ang_vel_norm,
            'lin_acc_norm': lin_acc_norm,
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        })

        # Calculate quality score
        quality_score = self.calculate_imu_quality_score()
        self.imu_quality_pub.publish(Float32(quality_score))

    def pose_callback(self, msg):
        # Store pose data for consistency checks
        self.pose_data.append({
            'timestamp': msg.header.stamp,
            'position': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            'orientation': [msg.pose.orientation.x, msg.pose.orientation.y,
                           msg.pose.orientation.z, msg.pose.orientation.w]
        })

    def calculate_range_stability(self, valid_ranges):
        """Calculate stability of range measurements over time"""
        if len(self.lidar_data) < 10 or len(valid_ranges) == 0:
            return 1.0  # Assume stable if insufficient data

        # Compare with previous measurements at similar angles
        prev_ranges = [data['valid_ratio'] for data in list(self.lidar_data)[-10:]]
        if len(prev_ranges) > 0:
            stability = 1.0 - np.std(prev_ranges)  # Lower std = more stable
            return max(0.0, min(1.0, stability))
        return 1.0

    def estimate_lidar_noise(self, valid_ranges):
        """Estimate noise level in LiDAR measurements"""
        if len(valid_ranges) < 2:
            return 0.0

        # Calculate differences between adjacent measurements
        if len(valid_ranges) > 1:
            range_diffs = [abs(valid_ranges[i] - valid_ranges[i-1])
                          for i in range(1, len(valid_ranges))]
            return np.std(range_diffs) if range_diffs else 0.0
        return 0.0

    def calculate_lidar_quality_score(self):
        """Calculate overall quality score for LiDAR data"""
        if not self.lidar_data:
            return 0.0

        recent_data = list(self.lidar_data)[-10:]  # Look at recent data

        # Calculate component scores
        avg_valid_ratio = np.mean([d['valid_ratio'] for d in recent_data])
        avg_noise_std = np.mean([d['noise_std'] for d in recent_data])
        avg_stability = np.mean([d['range_stability'] for d in recent_data])

        # Normalize to 0-1 scale
        valid_score = min(1.0, avg_valid_ratio / self.lidar_thresholds['min_valid_ratio'])
        noise_score = max(0.0, 1.0 - (avg_noise_std / self.lidar_thresholds['max_noise_std']))
        stability_score = avg_stability

        # Weighted average
        quality_score = (0.4 * valid_score + 0.3 * noise_score + 0.3 * stability_score)
        return max(0.0, min(1.0, quality_score))

    def calculate_camera_quality_score(self):
        """Calculate overall quality score for camera data"""
        if not self.depth_data:
            return 0.0

        recent_data = list(self.depth_data)[-10:]

        # Calculate component scores
        avg_brightness = np.mean([d['mean_brightness'] for d in recent_data])
        avg_sharpness = np.mean([d['sharpness'] for d in recent_data])
        avg_blur_ratio = np.mean([d['blur_ratio'] for d in recent_data])

        # Brightness score (should be in middle range)
        brightness_score = 1.0 - min(1.0,
            max(abs(avg_brightness - 127) / 100, 0.0))  # Tolerate brightness in 27-227 range

        # Sharpness score
        sharpness_score = min(1.0, avg_sharpness / 100.0)  # Normalize assuming max sharpness of 100

        # Blur score (lower blur ratio is better)
        blur_score = max(0.0, 1.0 - avg_blur_ratio / self.camera_thresholds['max_blur_ratio'])

        # Weighted average
        quality_score = (0.3 * brightness_score + 0.4 * sharpness_score + 0.3 * blur_score)
        return max(0.0, min(1.0, quality_score))

    def calculate_imu_quality_score(self):
        """Calculate overall quality score for IMU data"""
        if not self.imu_data:
            return 0.0

        recent_data = list(self.imu_data)[-50:]  # Use more samples for IMU

        if len(recent_data) < 10:
            return 0.8  # Assume reasonable quality if insufficient data for detailed analysis

        # Calculate angular velocity statistics
        ang_vel_norms = [d['ang_vel_norm'] for d in recent_data]
        avg_ang_vel = np.mean(ang_vel_norms)
        std_ang_vel = np.std(ang_vel_norms)

        # Calculate linear acceleration statistics (should include gravity)
        lin_acc_norms = [d['lin_acc_norm'] for d in recent_data]
        avg_lin_acc = np.mean(lin_acc_norms)
        std_lin_acc = np.std(lin_acc_norms)

        # Angular velocity score (should be reasonable)
        ang_vel_score = max(0.0, min(1.0, 2.0 - avg_ang_vel / 1.0))  # Assume max reasonable velocity is 1.0 rad/s

        # Acceleration score (should be around gravity + motion)
        acc_score = max(0.0, min(1.0, 1.0 - abs(avg_lin_acc - 9.81) / 5.0))  # Allow deviation from 9.81 m/s²

        # Stability score (lower std = more stable readings)
        stability_score = max(0.0, min(1.0, 1.0 - std_ang_vel / 0.1))  # Assume max reasonable std is 0.1

        # Weighted average
        quality_score = (0.4 * ang_vel_score + 0.3 * acc_score + 0.3 * stability_score)
        return max(0.0, min(1.0, quality_score))

    def calculate_system_health(self):
        """Calculate overall system health based on all sensors"""
        lidar_score = self.calculate_lidar_quality_score() if self.lidar_data else 1.0
        camera_score = self.calculate_camera_quality_score() if self.depth_data else 1.0
        imu_score = self.calculate_imu_quality_score() if self.imu_data else 1.0

        # Calculate weighted average (IMU is critical for navigation)
        health_score = (0.3 * lidar_score + 0.3 * camera_score + 0.4 * imu_score)

        # Publish system health
        self.system_health_pub.publish(Float32(health_score))

        # Log system status
        if health_score > 0.8:
            status = "HEALTHY"
        elif health_score > 0.6:
            status = "CAUTION"
        else:
            status = "ISSUES DETECTED"

        rospy.loginfo_throttle(5.0, f"System Health: {status} (Score: {health_score:.2f}) - "
                         f"Lidar: {lidar_score:.2f}, Camera: {camera_score:.2f}, IMU: {imu_score:.2f}")

        return health_score

if __name__ == '__main__':
    try:
        validator = ComprehensiveValidator()

        # Run system health calculation at 0.2 Hz
        rate = rospy.Rate(0.2)
        while not rospy.is_shutdown():
            validator.calculate_system_health()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
```

## Validation Report Generation

### Automated Validation Report

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, Imu, Image
import json
from datetime import datetime
import os

class ValidationReportGenerator:
    def __init__(self):
        rospy.init_node('validation_report_generator')

        # Subscribe to sensors
        self.lidar_sub = rospy.Subscriber('/laser_scan', LaserScan, self.lidar_stats)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_stats)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_stats)

        # Statistics storage
        self.lidar_stats_data = {
            'count': 0,
            'range_stats': {'min': [], 'max': [], 'mean': [], 'std': []},
            'valid_ratio': [],
            'timestamps': []
        }

        self.imu_stats_data = {
            'count': 0,
            'ang_vel_stats': {'x': [], 'y': [], 'z': []},
            'lin_acc_stats': {'x': [], 'y': [], 'z': []},
            'timestamps': []
        }

        self.depth_stats_data = {
            'count': 0,
            'depth_values': [],
            'valid_ratio': [],
            'timestamps': []
        }

        # Report generation timer
        rospy.Timer(rospy.Duration(60), self.generate_report)  # Generate report every minute

        rospy.loginfo("Validation Report Generator initialized")

    def lidar_stats(self, msg):
        self.lidar_stats_data['count'] += 1
        self.lidar_stats_data['timestamps'].append(rospy.Time.now().to_sec())

        # Calculate range statistics
        valid_ranges = [r for r in msg.ranges if not (np.isnan(r) or np.isinf(r))]
        if valid_ranges:
            self.lidar_stats_data['range_stats']['min'].append(min(valid_ranges))
            self.lidar_stats_data['range_stats']['max'].append(max(valid_ranges))
            self.lidar_stats_data['range_stats']['mean'].append(np.mean(valid_ranges))
            self.lidar_stats_data['range_stats']['std'].append(np.std(valid_ranges))

        valid_ratio = len(valid_ranges) / len(msg.ranges) if msg.ranges else 0
        self.lidar_stats_data['valid_ratio'].append(valid_ratio)

    def imu_stats(self, msg):
        self.imu_stats_data['count'] += 1
        self.imu_stats_data['timestamps'].append(rospy.Time.now().to_sec())

        self.imu_stats_data['ang_vel_stats']['x'].append(msg.angular_velocity.x)
        self.imu_stats_data['ang_vel_stats']['y'].append(msg.angular_velocity.y)
        self.imu_stats_data['ang_vel_stats']['z'].append(msg.angular_velocity.z)

        self.imu_stats_data['lin_acc_stats']['x'].append(msg.linear_acceleration.x)
        self.imu_stats_data['lin_acc_stats']['y'].append(msg.linear_acceleration.y)
        self.imu_stats_data['lin_acc_stats']['z'].append(msg.linear_acceleration.z)

    def depth_stats(self, msg):
        try:
            import cv2
            from cv_bridge import CvBridge
            bridge = CvBridge()

            cv_image = bridge.imgmsg_to_cv2(msg, "32FC1")
            valid_pixels = cv_image[~np.isnan(cv_image) & (cv_image > 0)]

            self.depth_stats_data['count'] += 1
            self.depth_stats_data['timestamps'].append(rospy.Time.now().to_sec())

            if len(valid_pixels) > 0:
                self.depth_stats_data['depth_values'].extend(valid_pixels)
                valid_ratio = len(valid_pixels) / (cv_image.shape[0] * cv_image.shape[1])
                self.depth_stats_data['valid_ratio'].append(valid_ratio)

        except Exception as e:
            rospy.logerr(f"Error processing depth stats: {e}")

    def generate_report(self, event):
        """Generate and save validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            'timestamp': timestamp,
            'duration': 60,  # Report covers 60 seconds
            'lidar_validation': self.generate_lidar_report(),
            'imu_validation': self.generate_imu_report(),
            'depth_validation': self.generate_depth_report()
        }

        # Save report to file
        report_dir = rospy.get_param('~report_dir', '/tmp/sensor_validation_reports')
        os.makedirs(report_dir, exist_ok=True)

        report_filename = os.path.join(report_dir, f"validation_report_{timestamp}.json")
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)

        rospy.loginfo(f"Validation report saved to: {report_filename}")

    def generate_lidar_report(self):
        """Generate LiDAR-specific validation report"""
        if not self.lidar_stats_data['range_stats']['mean']:
            return {'status': 'NO_DATA', 'message': 'No LiDAR data received'}

        range_stats = self.lidar_stats_data['range_stats']
        valid_ratios = self.lidar_stats_data['valid_ratio']

        report = {
            'status': 'ANALYZED',
            'total_messages': self.lidar_stats_data['count'],
            'range_analysis': {
                'min_distance': {
                    'mean': float(np.mean(range_stats['min'])) if range_stats['min'] else 0,
                    'std': float(np.std(range_stats['min'])) if range_stats['min'] else 0,
                    'min': float(np.min(range_stats['min'])) if range_stats['min'] else 0,
                    'max': float(np.max(range_stats['min'])) if range_stats['min'] else 0
                },
                'max_distance': {
                    'mean': float(np.mean(range_stats['max'])) if range_stats['max'] else 0,
                    'std': float(np.std(range_stats['max'])) if range_stats['max'] else 0,
                    'min': float(np.min(range_stats['max'])) if range_stats['max'] else 0,
                    'max': float(np.max(range_stats['max'])) if range_stats['max'] else 0
                },
                'mean_distance': {
                    'mean': float(np.mean(range_stats['mean'])) if range_stats['mean'] else 0,
                    'std': float(np.std(range_stats['mean'])) if range_stats['mean'] else 0
                }
            },
            'data_quality': {
                'average_valid_ratio': float(np.mean(valid_ratios)) if valid_ratios else 0,
                'valid_ratio_std': float(np.std(valid_ratios)) if valid_ratios else 0
            }
        }

        # Add validation status
        avg_valid_ratio = report['data_quality']['average_valid_ratio']
        if avg_valid_ratio < 0.5:
            report['status'] = 'POOR'
            report['issues'] = ['Low data validity ratio']
        elif avg_valid_ratio < 0.8:
            report['status'] = 'FAIR'
        else:
            report['status'] = 'GOOD'

        return report

    def generate_imu_report(self):
        """Generate IMU-specific validation report"""
        if not self.imu_stats_data['ang_vel_stats']['x']:
            return {'status': 'NO_DATA', 'message': 'No IMU data received'}

        report = {
            'status': 'ANALYZED',
            'total_messages': self.imu_stats_data['count'],
            'angular_velocity_analysis': {
                'x': {
                    'mean': float(np.mean(self.imu_stats_data['ang_vel_stats']['x'])),
                    'std': float(np.std(self.imu_stats_data['ang_vel_stats']['x'])),
                    'min': float(np.min(self.imu_stats_data['ang_vel_stats']['x'])),
                    'max': float(np.max(self.imu_stats_data['ang_vel_stats']['x']))
                },
                'y': {
                    'mean': float(np.mean(self.imu_stats_data['ang_vel_stats']['y'])),
                    'std': float(np.std(self.imu_stats_data['ang_vel_stats']['y'])),
                    'min': float(np.min(self.imu_stats_data['ang_vel_stats']['y'])),
                    'max': float(np.max(self.imu_stats_data['ang_vel_stats']['y']))
                },
                'z': {
                    'mean': float(np.mean(self.imu_stats_data['ang_vel_stats']['z'])),
                    'std': float(np.std(self.imu_stats_data['ang_vel_stats']['z'])),
                    'min': float(np.min(self.imu_stats_data['ang_vel_stats']['z'])),
                    'max': float(np.max(self.imu_stats_data['ang_vel_stats']['z']))
                }
            },
            'linear_acceleration_analysis': {
                'x': {
                    'mean': float(np.mean(self.imu_stats_data['lin_acc_stats']['x'])),
                    'std': float(np.std(self.imu_stats_data['lin_acc_stats']['x'])),
                    'min': float(np.min(self.imu_stats_data['lin_acc_stats']['x'])),
                    'max': float(np.max(self.imu_stats_data['lin_acc_stats']['x']))
                },
                'y': {
                    'mean': float(np.mean(self.imu_stats_data['lin_acc_stats']['y'])),
                    'std': float(np.std(self.imu_stats_data['lin_acc_stats']['y'])),
                    'min': float(np.min(self.imu_stats_data['lin_acc_stats']['y'])),
                    'max': float(np.max(self.imu_stats_data['lin_acc_stats']['y']))
                },
                'z': {
                    'mean': float(np.mean(self.imu_stats_data['lin_acc_stats']['z'])),
                    'std': float(np.std(self.imu_stats_data['lin_acc_stats']['z'])),
                    'min': float(np.min(self.imu_stats_data['lin_acc_stats']['z'])),
                    'max': float(np.max(self.imu_stats_data['lin_acc_stats']['z']))
                }
            }
        }

        # Check for potential issues
        issues = []
        avg_ang_vel_mag = np.mean([
            np.sqrt(x**2 + y**2 + z**2)
            for x, y, z in zip(
                self.imu_stats_data['ang_vel_stats']['x'],
                self.imu_stats_data['ang_vel_stats']['y'],
                self.imu_stats_data['ang_vel_stats']['z']
            )
        ])

        if avg_ang_vel_mag > 5.0:  # Very high average angular velocity
            issues.append('High average angular velocity detected')

        avg_lin_acc_mag = np.mean([
            np.sqrt(x**2 + y**2 + z**2)
            for x, y, z in zip(
                self.imu_stats_data['lin_acc_stats']['x'],
                self.imu_stats_data['lin_acc_stats']['y'],
                self.imu_stats_data['lin_acc_stats']['z']
            )
        ])

        if abs(avg_lin_acc_mag - 9.81) > 2.0:  # Deviation from gravity more than 2 m/s²
            issues.append('Linear acceleration significantly different from expected gravity')

        if issues:
            report['status'] = 'ISSUES_DETECTED'
            report['issues'] = issues
        else:
            report['status'] = 'GOOD'

        return report

    def generate_depth_report(self):
        """Generate depth camera validation report"""
        if not self.depth_stats_data['depth_values']:
            return {'status': 'NO_DATA', 'message': 'No depth data received'}

        report = {
            'status': 'ANALYZED',
            'total_messages': self.depth_stats_data['count'],
            'depth_analysis': {
                'total_valid_points': len(self.depth_stats_data['depth_values']),
                'depth_range': {
                    'min': float(np.min(self.depth_stats_data['depth_values'])),
                    'max': float(np.max(self.depth_stats_data['depth_values'])),
                    'mean': float(np.mean(self.depth_stats_data['depth_values'])),
                    'std': float(np.std(self.depth_stats_data['depth_values']))
                }
            },
            'data_quality': {
                'average_valid_ratio': float(np.mean(self.depth_stats_data['valid_ratio'])) if self.depth_stats_data['valid_ratio'] else 0,
                'valid_ratio_std': float(np.std(self.depth_stats_data['valid_ratio'])) if self.depth_stats_data['valid_ratio'] else 0
            }
        }

        # Add validation status
        avg_valid_ratio = report['data_quality']['average_valid_ratio']
        if avg_valid_ratio < 0.3:
            report['status'] = 'POOR'
            report['issues'] = ['Very low depth data validity ratio']
        elif avg_valid_ratio < 0.6:
            report['status'] = 'FAIR'
        else:
            report['status'] = 'GOOD'

        return report

if __name__ == '__main__':
    try:
        generator = ValidationReportGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Next Steps

In the next section, we'll document sensor fusion techniques specifically in the context of digital twin applications, building on the validation and comparison techniques covered in this chapter.

## Exercises

1. **Implementation Exercise**: Create a validation node that monitors multiple sensors simultaneously and generates alerts when sensor quality drops below acceptable thresholds.

2. **Comparison Exercise**: Develop a system that compares simulated sensor data with real sensor specifications to validate the realism of the simulation.

3. **Performance Exercise**: Implement a validation pipeline that can operate in real-time without significantly impacting system performance.
---
title: Sensor Fusion in Digital Twins
sidebar_position: 9
---

# Sensor Fusion in Digital Twins

## Overview

Sensor fusion is the process of combining data from multiple sensors to achieve improved accuracy, reliability, and robustness compared to using individual sensors alone. In digital twin applications, sensor fusion is crucial for creating accurate virtual representations of physical systems that can be used for monitoring, analysis, and prediction. This chapter explores various sensor fusion techniques specifically tailored for digital twin applications with multiple sensor types.

## Learning Objectives

After completing this chapter, you will be able to:
- Understand the principles and benefits of sensor fusion in digital twin contexts
- Implement various sensor fusion algorithms (Kalman filters, particle filters, etc.)
- Design fusion architectures that integrate LiDAR, camera, and IMU data
- Evaluate the performance of sensor fusion systems in digital twin applications
- Handle sensor failures and degraded performance gracefully

## Prerequisites

- Understanding of individual sensor simulation from previous chapters
- Basic knowledge of probability theory and statistics
- Experience with ROS message types and processing
- Familiarity with coordinate frame transformations

## Introduction to Sensor Fusion in Digital Twins

Digital twins require accurate and reliable state estimation to maintain synchronization with their physical counterparts. Sensor fusion plays a critical role by:

1. **Improving accuracy**: Combining multiple sensors reduces overall measurement uncertainty
2. **Enhancing reliability**: Redundant sensors provide backup when individual sensors fail
3. **Increasing robustness**: Different sensors complement each other's strengths and weaknesses
4. **Enabling comprehensive perception**: Multiple modalities provide richer information about the environment

### Types of Sensor Fusion

**Low-level Fusion**: Combines raw sensor measurements before any processing
**Mid-level Fusion**: Combines processed sensor data (features, detections)
**High-level Fusion**: Combines decisions or interpretations from different sensors

## Kalman Filter-Based Fusion

### Extended Kalman Filter (EKF) for Multi-Sensor Fusion

The Extended Kalman Filter is suitable for non-linear systems and can fuse data from different sensor types:

```cpp
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Dense>

class EKFSensorFusion {
public:
    EKFSensorFusion(ros::NodeHandle& nh) : nh_(nh) {
        // Initialize subscribers
        imu_sub_ = nh_.subscribe("/imu/data", 100, &EKFSensorFusion::imuCallback, this);
        lidar_sub_ = nh_.subscribe("/laser_scan", 10, &EKFSensorFusion::lidarCallback, this);

        // Publisher for fused state
        pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/fused_pose", 10);
        tf_broadcaster_ = std::make_shared<tf::TransformBroadcaster>();

        // Initialize EKF
        initializeEKF();

        ROS_INFO("EKF Sensor Fusion initialized");
    }

private:
    ros::NodeHandle& nh_;
    ros::Subscriber imu_sub_, lidar_sub_;
    ros::Publisher pose_pub_;
    std::shared_ptr<tf::TransformBroadcaster> tf_broadcaster_;

    // EKF components
    Eigen::VectorXd state_;  // [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    Eigen::MatrixXd P_;      // Error covariance matrix
    Eigen::MatrixXd Q_;      // Process noise covariance
    Eigen::MatrixXd R_imu_;  // IMU measurement noise
    Eigen::MatrixXd R_lidar_;// LiDAR measurement noise
    Eigen::MatrixXd F_;      // State transition model
    Eigen::MatrixXd H_imu_;  // IMU observation model
    Eigen::MatrixXd H_lidar_;// LiDAR observation model

    bool initialized_ = false;
    ros::Time last_update_time_;

    void initializeEKF() {
        // State vector: position (3), velocity (3), orientation (4 as quaternion)
        state_ = Eigen::VectorXd::Zero(10);
        P_ = Eigen::MatrixXd::Identity(10, 10) * 1000;  // Large initial uncertainty

        // Process noise covariance
        Q_ = Eigen::MatrixXd::Identity(10, 10);
        Q_.block<3,3>(0,0) *= 0.1;   // Position process noise
        Q_.block<3,3>(3,3) *= 1.0;   // Velocity process noise
        Q_.block<4,4>(6,6) *= 0.01;  // Orientation process noise

        // IMU measurement noise (angular velocity and linear acceleration)
        R_imu_ = Eigen::MatrixXd::Identity(6, 6);
        R_imu_.block<3,3>(0,0) *= 0.01;  // Angular velocity noise
        R_imu_.block<3,3>(3,3) *= 0.1;   // Linear acceleration noise

        // LiDAR measurement noise (position)
        R_lidar_ = Eigen::MatrixXd::Identity(3, 3) * 0.1;  // Position noise

        // Initialize observation models
        H_imu_ = Eigen::MatrixXd::Zero(6, 10);
        H_imu_.block<3,3>(0, 0) = Eigen::Matrix3d::Identity();  // Angular velocity part
        H_imu_.block<3,3>(3, 3) = Eigen::Matrix3d::Identity();  // Linear acceleration part

        H_lidar_ = Eigen::MatrixXd::Zero(3, 10);
        H_lidar_.block<3,3>(0, 0) = Eigen::Matrix3d::Identity();  // Position part
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        if (!initialized_) {
            initializeWithIMU(msg);
            return;
        }

        ros::Time current_time = msg->header.stamp;
        double dt = (current_time - last_update_time_).toSec();
        last_update_time_ = current_time;

        if (dt <= 0) return;

        // Prediction step using IMU data
        predictState(dt, msg);
        predictCovariance(dt);

        // Update step with IMU measurements
        Eigen::VectorXd imu_measurement(6);
        imu_measurement <<
            msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
            msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

        updateState(imu_measurement, R_imu_, H_imu_);

        publishState(msg->header);
    }

    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        if (!initialized_) return;

        // Extract position estimate from LiDAR scan (simplified - in practice,
        // you'd use landmark detection or SLAM)
        Eigen::Vector3d position_estimate = extractPositionFromScan(*msg);

        // Update step with LiDAR measurements
        Eigen::VectorXd lidar_measurement(3);
        lidar_measurement << position_estimate.x(), position_estimate.y(), position_estimate.z();

        updateState(lidar_measurement, R_lidar_, H_lidar_);

        publishState(msg->header);
    }

    void initializeWithIMU(const sensor_msgs::Imu::ConstPtr& msg) {
        // Initialize orientation from IMU
        state_[6] = msg->orientation.w;  // qw
        state_[7] = msg->orientation.x;  // qx
        state_[8] = msg->orientation.y;  // qy
        state_[9] = msg->orientation.z;  // qz

        // Initialize angular velocity in state (for prediction model)
        state_[0] = 0;  // Initial x position
        state_[1] = 0;  // Initial y position
        state_[2] = 0;  // Initial z position
        state_[3] = 0;  // Initial x velocity
        state_[4] = 0;  // Initial y velocity
        state_[5] = 0;  // Initial z velocity

        initialized_ = true;
        last_update_time_ = msg->header.stamp;
    }

    void predictState(double dt, const sensor_msgs::Imu::ConstPtr& msg) {
        // Extract current state
        Eigen::Vector3d pos(state_[0], state_[1], state_[2]);
        Eigen::Vector3d vel(state_[3], state_[4], state_[5]);
        Eigen::Quaterniond quat(state_[6], state_[7], state_[8], state_[9]);
        quat.normalize();

        // Convert IMU acceleration from body frame to world frame
        Eigen::Matrix3d R = quat.toRotationMatrix();
        Eigen::Vector3d acc_body(
            msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z
        );
        Eigen::Vector3d acc_world = R * acc_body;

        // Update state using motion model
        pos += vel * dt + 0.5 * acc_world * dt * dt;
        vel += acc_world * dt;

        // Update quaternion with angular velocity (simplified integration)
        Eigen::Vector3d ang_vel(
            msg->angular_velocity.x,
            msg->angular_velocity.y,
            msg->angular_velocity.z
        );

        Eigen::Quaterniond quat_dot(
            0,
            ang_vel.x() * 0.5,
            ang_vel.y() * 0.5,
            ang_vel.z() * 0.5
        );
        quat = quat + quat_dot * quat * dt;
        quat.normalize();

        // Update state vector
        state_.segment<3>(0) = pos;
        state_.segment<3>(3) = vel;
        state_[6] = quat.w();
        state_[7] = quat.x();
        state_[8] = quat.y();
        state_[9] = quat.z();
    }

    void predictCovariance(double dt) {
        // Linearize the motion model to get state transition matrix F
        F_ = Eigen::MatrixXd::Identity(10, 10);

        // Simplified F matrix - in practice, you'd compute the full Jacobian
        // Position-velocity relationship
        F_.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

        // More accurate prediction would involve the full Jacobian of the motion model
        // For this example, we'll use the simplified approach

        // Predict covariance
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void updateState(const Eigen::VectorXd& measurement,
                    const Eigen::MatrixXd& R,
                    const Eigen::MatrixXd& H) {
        // Innovation
        Eigen::VectorXd y = measurement - H * state_;

        // Innovation covariance
        Eigen::MatrixXd S = H * P_ * H.transpose() + R;

        // Kalman gain
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        // Update state and covariance
        state_ = state_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(state_.size(), state_.size()) - K * H) * P_;
    }

    Eigen::Vector3d extractPositionFromScan(const sensor_msgs::LaserScan& scan) {
        // Simplified position extraction - in practice, you'd use landmark detection
        // or SLAM to get absolute position estimates
        // This is a placeholder that returns a relative position based on scan features

        std::vector<float> valid_ranges;
        for (float range : scan.ranges) {
            if (!std::isnan(range) && !std::isinf(range) &&
                range >= scan.range_min && range <= scan.range_max) {
                valid_ranges.push_back(range);
            }
        }

        if (valid_ranges.empty()) {
            return Eigen::Vector3d(0, 0, 0);
        }

        // Calculate center of detected obstacles as a simple position estimate
        float avg_range = std::accumulate(valid_ranges.begin(), valid_ranges.end(), 0.0) / valid_ranges.size();

        // Use the angle of the center beam for direction
        size_t center_idx = scan.ranges.size() / 2;
        float angle = scan.angle_min + center_idx * scan.angle_increment;

        return Eigen::Vector3d(
            avg_range * cos(angle),
            avg_range * sin(angle),
            0.0  // Assuming 2D operation for simplicity
        );
    }

    void publishState(const std_msgs::Header& header) {
        geometry_msgs::PoseWithCovarianceStamped pose_msg;
        pose_msg.header = header;
        pose_msg.header.frame_id = "map";

        // Fill position
        pose_msg.pose.pose.position.x = state_[0];
        pose_msg.pose.pose.position.y = state_[1];
        pose_msg.pose.pose.position.z = state_[2];

        // Fill orientation
        pose_msg.pose.pose.orientation.w = state_[6];
        pose_msg.pose.pose.orientation.x = state_[7];
        pose_msg.pose.pose.orientation.y = state_[8];
        pose_msg.pose.pose.orientation.z = state_[9];

        // Fill covariance (convert from P matrix to ROS format)
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                pose_msg.pose.covariance[i*6 + j] = P_(i, j);
            }
        }

        pose_pub_.publish(pose_msg);

        // Broadcast TF transform
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(state_[0], state_[1], state_[2]));
        transform.setRotation(tf::Quaternion(state_[7], state_[8], state_[9], state_[6]));
        tf_broadcaster_->sendTransform(
            tf::StampedTransform(transform, header.stamp, "map", "base_link"));
    }
};
```

### Unscented Kalman Filter (UKF) for Non-Linear Fusion

For highly non-linear systems, the Unscented Kalman Filter provides better performance:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import deque

class UKFSensorFusion:
    def __init__(self):
        rospy.init_node('ukf_sensor_fusion')

        # Subscribe to sensors
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.lidar_sub = rospy.Subscriber('/laser_scan', LaserScan, self.lidar_callback)

        # Publisher for fused state
        self.pose_pub = rospy.Publisher('/fused_pose_ukf', PoseWithCovarianceStamped, queue_size=10)

        # UKF parameters
        self.state_dim = 9  # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 1000  # Initial uncertainty

        # UKF weights and sigma points
        self.alpha = 1e-3  # Spread of sigma points
        self.kappa = 0     # Secondary scaling parameter
        self.beta = 2      # Prior knowledge of distribution (2 for Gaussian)

        self.lamb = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.gamma = np.sqrt(self.state_dim + self.lamb)

        # Process and measurement noise
        self.Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])  # Process noise
        self.R_imu = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])  # IMU noise [angular_vel, linear_acc]
        self.R_lidar = np.diag([0.1, 0.1, 0.1])  # LiDAR position noise

        self.initialized = False
        self.last_time = None

        rospy.loginfo("UKF Sensor Fusion initialized")

    def compute_sigma_points(self):
        """Compute sigma points for UKF"""
        # Ensure covariance matrix is positive semi-definite
        U = np.linalg.cholesky((self.state_dim + self.lamb) * self.covariance)

        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        sigma_points[0] = self.state  # Center point

        for i in range(self.state_dim):
            sigma_points[i + 1] = self.state + U[i, :]
            sigma_points[i + 1 + self.state_dim] = self.state - U[i, :]

        return sigma_points

    def state_prediction_model(self, state, dt, angular_vel, linear_acc):
        """Non-linear state prediction model"""
        # Extract state variables
        x, y, z, vx, vy, vz, roll, pitch, yaw = state

        # Convert angular velocity to orientation changes
        d_roll = angular_vel.x * dt
        d_pitch = angular_vel.y * dt
        d_yaw = angular_vel.z * dt

        # Update orientation
        new_roll = roll + d_roll
        new_pitch = pitch + d_pitch
        new_yaw = yaw + d_yaw

        # Convert acceleration from body frame to world frame
        # Using rotation matrix for coordinate transformation
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        world_acc = R @ np.array([linear_acc.x, linear_acc.y, linear_acc.z])

        # Update velocity
        new_vx = vx + world_acc[0] * dt
        new_vy = vy + world_acc[1] * dt
        new_vz = vz + (world_acc[2] - 9.81) * dt  # Subtract gravity

        # Update position
        new_x = x + vx * dt + 0.5 * world_acc[0] * dt**2
        new_y = y + vy * dt + 0.5 * world_acc[1] * dt**2
        new_z = z + vz * dt + 0.5 * (world_acc[2] - 9.81) * dt**2

        return np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz,
                        new_roll, new_pitch, new_yaw])

    def measurement_model_imu(self, state):
        """Measurement model for IMU (angular velocity and linear acceleration)"""
        # Extract orientation from state
        roll, pitch, yaw = state[6], state[7], state[8]

        # For this example, we'll return the state values directly
        # In practice, this would transform the state to expected IMU readings
        return np.array([0, 0, 0, 0, 0, 0])  # Placeholder

    def measurement_model_lidar(self, state):
        """Measurement model for LiDAR (position)"""
        # Return position measurements
        return state[:3]  # x, y, z position

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles to rotation matrix"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R

    def unscented_transform(self, sigma_points, noise_cov):
        """Apply unscented transform to propagate sigma points through non-linear function"""
        n = sigma_points.shape[1]
        weights_m = np.zeros(2 * n + 1)  # Mean weights
        weights_c = np.zeros(2 * n + 1)  # Covariance weights

        # Compute weights
        weights_m[0] = self.lamb / (n + self.lamb)
        weights_c[0] = self.lamb / (n + self.lamb) + (1 - self.alpha**2 + self.beta)

        for i in range(1, 2 * n + 1):
            weights_m[i] = weights_c[i] = 1.0 / (2 * (n + self.lamb))

        return weights_m, weights_c

    def imu_callback(self, msg):
        if not self.initialized:
            self.initialize_with_imu(msg)
            return

        current_time = rospy.Time.now()
        if self.last_time is None:
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt <= 0:
            return

        # Prediction step
        self.ukf_predict(msg, dt)

        # Update step with IMU measurements
        imu_measurement = np.array([
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ])

        self.ukf_update(imu_measurement, self.R_imu, self.measurement_model_imu)

        self.publish_state(current_time)

    def lidar_callback(self, msg):
        if not self.initialized:
            return

        # For this example, we'll extract a simple position estimate from the scan
        # In practice, you'd use landmark detection or SLAM
        position_estimate = self.extract_position_from_scan(msg)

        # Update step with LiDAR measurements
        self.ukf_update(position_estimate, self.R_lidar, self.measurement_model_lidar)

        current_time = msg.header.stamp
        self.publish_state(current_time)

    def ukf_predict(self, imu_msg, dt):
        """UKF prediction step"""
        # Get sigma points
        sigma_points = self.compute_sigma_points()

        # Propagate sigma points through non-linear model
        propagated_points = np.zeros_like(sigma_points)
        for i, point in enumerate(sigma_points):
            propagated_points[i] = self.state_prediction_model(
                point, dt,
                imu_msg.angular_velocity,
                imu_msg.linear_acceleration
            )

        # Compute predicted state and covariance
        weights_m, weights_c = self.unscented_transform(sigma_points, self.Q)

        # Predicted state
        predicted_state = np.zeros(self.state_dim)
        for i in range(len(sigma_points)):
            predicted_state += weights_m[i] * propagated_points[i]

        # Predicted covariance
        predicted_cov = np.zeros((self.state_dim, self.state_dim))
        for i in range(len(sigma_points)):
            diff = propagated_points[i] - predicted_state
            predicted_cov += weights_c[i] * np.outer(diff, diff)
        predicted_cov += self.Q

        self.state = predicted_state
        self.covariance = predicted_cov

    def ukf_update(self, measurement, measurement_noise, measurement_function):
        """UKF update step"""
        # Get sigma points
        sigma_points = self.compute_sigma_points()

        # Transform sigma points through measurement function
        measurement_sigma_points = np.zeros((len(sigma_points), len(measurement)))
        for i, point in enumerate(sigma_points):
            measurement_sigma_points[i] = measurement_function(point)

        # Compute predicted measurement
        weights_m, weights_c = self.unscented_transform(sigma_points, measurement_noise)

        predicted_measurement = np.zeros(len(measurement))
        for i in range(len(sigma_points)):
            predicted_measurement += weights_m[i] * measurement_sigma_points[i]

        # Compute measurement covariance
        measurement_cov = np.zeros((len(measurement), len(measurement)))
        for i in range(len(sigma_points)):
            diff = measurement_sigma_points[i] - predicted_measurement
            measurement_cov += weights_c[i] * np.outer(diff, diff)
        measurement_cov += measurement_noise

        # Compute cross-covariance
        cross_cov = np.zeros((self.state_dim, len(measurement)))
        for i in range(len(sigma_points)):
            state_diff = sigma_points[i] - self.state
            meas_diff = measurement_sigma_points[i] - predicted_measurement
            cross_cov += weights_c[i] * np.outer(state_diff, meas_diff)

        # Compute Kalman gain
        kalman_gain = cross_cov @ np.linalg.inv(measurement_cov)

        # Update state and covariance
        innovation = measurement - predicted_measurement
        self.state = self.state + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ measurement_cov @ kalman_gain.T

    def extract_position_from_scan(self, scan_msg):
        """Extract position estimate from LiDAR scan"""
        # This is a simplified example - in practice, you'd use landmark detection or SLAM
        # For demonstration, we'll return a placeholder position
        return np.array([0.0, 0.0, 0.0])  # Placeholder

    def initialize_with_imu(self, msg):
        """Initialize filter with IMU data"""
        # Extract orientation from IMU
        _, _, yaw = euler_from_quaternion([
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        ])

        self.state[6] = 0  # roll
        self.state[7] = 0  # pitch
        self.state[8] = yaw  # yaw

        self.initialized = True
        self.last_time = rospy.Time.now()

    def publish_state(self, timestamp):
        """Publish the fused state"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = self.state[0]
        pose_msg.pose.pose.position.y = self.state[1]
        pose_msg.pose.pose.position.z = self.state[2]

        # Convert Euler angles to quaternion
        quat = quaternion_from_euler(self.state[6], self.state[7], self.state[8])
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]

        # Copy covariance matrix to ROS format (flatten 6x6 matrix)
        for i in range(6):
            for j in range(6):
                idx = i * 6 + j
                if i < self.covariance.shape[0] and j < self.covariance.shape[1]:
                    pose_msg.pose.covariance[idx] = self.covariance[i, j]
                else:
                    pose_msg.pose.covariance[idx] = 0.0

        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        fusion = UKFSensorFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Particle Filter for Multi-Modal Fusion

Particle filters are effective for multi-modal distributions and non-Gaussian noise:

```cpp
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

class ParticleFilterFusion {
public:
    struct Particle {
        double x, y, z;
        double vx, vy, vz;
        double roll, pitch, yaw;
        double weight;

        Particle() : x(0), y(0), z(0), vx(0), vy(0), vz(0),
                     roll(0), pitch(0), yaw(0), weight(1.0) {}
    };

    ParticleFilterFusion(ros::NodeHandle& nh) : nh_(nh), rng_(std::random_device{}()) {
        imu_sub_ = nh_.subscribe("/imu/data", 100, &ParticleFilterFusion::imuCallback, this);
        lidar_sub_ = nh_.subscribe("/laser_scan", 10, &ParticleFilterFusion::lidarCallback, this);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/particle_pose", 10);

        // Initialize particles
        initializeParticles();

        ROS_INFO("Particle Filter Fusion initialized");
    }

private:
    ros::NodeHandle& nh_;
    ros::Subscriber imu_sub_, lidar_sub_;
    ros::Publisher pose_pub_;

    std::vector<Particle> particles_;
    std::mt19937 rng_;
    int num_particles_ = 1000;
    double resample_threshold_ = 0.5;

    void initializeParticles() {
        particles_.resize(num_particles_);

        std::normal_distribution<double> pos_dist(0.0, 0.5);  // Initial position uncertainty
        std::normal_distribution<double> vel_dist(0.0, 0.1);  // Initial velocity uncertainty
        std::normal_distribution<double> angle_dist(0.0, 0.1); // Initial angle uncertainty

        for (auto& particle : particles_) {
            particle.x = pos_dist(rng_);
            particle.y = pos_dist(rng_);
            particle.z = pos_dist(rng_);
            particle.vx = vel_dist(rng_);
            particle.vy = vel_dist(rng_);
            particle.vz = vel_dist(rng_);
            particle.roll = angle_dist(rng_);
            particle.pitch = angle_dist(rng_);
            particle.yaw = angle_dist(rng_);
            particle.weight = 1.0 / num_particles_;
        }
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        if (particles_.empty()) return;

        // Prediction step: update particles based on IMU input
        predictParticles(*msg);

        // Resample if needed
        if (calculateEffectiveSampleSize() < resample_threshold_ * num_particles_) {
            resampleParticles();
        }

        publishEstimate(msg->header);
    }

    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        if (particles_.empty()) return;

        // Update particle weights based on LiDAR likelihood
        updateWeightsWithLidar(*msg);

        // Resample if needed
        if (calculateEffectiveSampleSize() < resample_threshold_ * num_particles_) {
            resampleParticles();
        }

        publishEstimate(msg->header);
    }

    void predictParticles(const sensor_msgs::Imu& imu_msg) {
        static ros::Time last_time = ros::Time(0);
        if (last_time.is_zero()) {
            last_time = imu_msg.header.stamp;
            return;
        }

        double dt = (imu_msg.header.stamp - last_time).toSec();
        last_time = imu_msg.header.stamp;

        if (dt <= 0) return;

        // Process noise parameters
        std::normal_distribution<double> pos_noise(0.0, 0.01);
        std::normal_distribution<double> vel_noise(0.0, 0.05);
        std::normal_distribution<double> angle_noise(0.0, 0.001);

        for (auto& particle : particles_) {
            // Update position based on velocity
            particle.x += particle.vx * dt + pos_noise(rng_);
            particle.y += particle.vy * dt + pos_noise(rng_);
            particle.z += particle.vz * dt + pos_noise(rng_);

            // Update velocity based on acceleration
            particle.vx += imu_msg.linear_acceleration.x * dt + vel_noise(rng_);
            particle.vy += imu_msg.linear_acceleration.y * dt + vel_noise(rng_);
            particle.vz += (imu_msg.linear_acceleration.z - 9.81) * dt + vel_noise(rng_);  // Subtract gravity

            // Update orientation based on angular velocity
            particle.roll += imu_msg.angular_velocity.x * dt + angle_noise(rng_);
            particle.pitch += imu_msg.angular_velocity.y * dt + angle_noise(rng_);
            particle.yaw += imu_msg.angular_velocity.z * dt + angle_noise(rng_);
        }
    }

    void updateWeightsWithLidar(const sensor_msgs::LaserScan& scan) {
        // Calculate likelihood of each particle given the LiDAR scan
        // This is a simplified example - in practice, you'd implement scan matching
        // or likelihood fields for proper LiDAR update

        for (auto& particle : particles_) {
            // Calculate a simple likelihood based on expected measurements
            double likelihood = calculateLidarLikelihood(particle, scan);
            particle.weight *= likelihood;
        }

        // Normalize weights
        double total_weight = std::accumulate(particles_.begin(), particles_.end(), 0.0,
            [](double sum, const Particle& p) { return sum + p.weight; });

        if (total_weight > 0) {
            for (auto& particle : particles_) {
                particle.weight /= total_weight;
            }
        } else {
            // If all weights are zero, reset to uniform distribution
            for (auto& particle : particles_) {
                particle.weight = 1.0 / num_particles_;
            }
        }
    }

    double calculateLidarLikelihood(const Particle& particle, const sensor_msgs::LaserScan& scan) {
        // Simplified likelihood calculation - in practice, this would involve
        // comparing expected sensor readings with actual readings
        // For this example, we'll return a basic probability based on position uncertainty

        // If particle position is reasonable, give higher weight
        if (std::abs(particle.x) < 10.0 && std::abs(particle.y) < 10.0) {
            return 1.0;
        } else {
            return 0.1;  // Lower weight for particles far from origin
        }
    }

    void resampleParticles() {
        // Systematic resampling
        std::vector<Particle> new_particles;
        new_particles.reserve(num_particles_);

        // Calculate cumulative weights
        std::vector<double> cumulative_weights(num_particles_);
        cumulative_weights[0] = particles_[0].weight;
        for (size_t i = 1; i < particles_.size(); ++i) {
            cumulative_weights[i] = cumulative_weights[i-1] + particles_[i].weight;
        }

        // Generate random starting point
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0 / num_particles_);
        double start = uniform_dist(rng_);

        // Systematic sampling
        size_t index = 0;
        for (int i = 0; i < num_particles_; ++i) {
            double threshold = start + i * (1.0 / num_particles_);

            while (index < num_particles_ - 1 && cumulative_weights[index] < threshold) {
                ++index;
            }

            new_particles.push_back(particles_[index]);
        }

        particles_ = std::move(new_particles);

        // Reset weights to uniform
        for (auto& particle : particles_) {
            particle.weight = 1.0 / num_particles_;
        }
    }

    double calculateEffectiveSampleSize() {
        double sum_weights_sq = 0.0;
        for (const auto& particle : particles_) {
            sum_weights_sq += particle.weight * particle.weight;
        }
        return 1.0 / sum_weights_sq;
    }

    void publishEstimate(const std_msgs::Header& header) {
        // Calculate weighted mean of particles
        Particle mean_particle;
        double total_weight = 0.0;

        for (const auto& particle : particles_) {
            mean_particle.x += particle.x * particle.weight;
            mean_particle.y += particle.y * particle.weight;
            mean_particle.z += particle.z * particle.weight;
            mean_particle.vx += particle.vx * particle.weight;
            mean_particle.vy += particle.vy * particle.weight;
            mean_particle.vz += particle.vz * particle.weight;
            mean_particle.roll += particle.roll * particle.weight;
            mean_particle.pitch += particle.pitch * particle.weight;
            mean_particle.yaw += particle.yaw * particle.weight;
            total_weight += particle.weight;
        }

        // Normalize by total weight
        mean_particle.x /= total_weight;
        mean_particle.y /= total_weight;
        mean_particle.z /= total_weight;
        mean_particle.vx /= total_weight;
        mean_particle.vy /= total_weight;
        mean_particle.vz /= total_weight;
        mean_particle.roll /= total_weight;
        mean_particle.pitch /= total_weight;
        mean_particle.yaw /= total_weight;

        // Calculate covariance (simplified)
        geometry_msgs::PoseWithCovarianceStamped pose_msg;
        pose_msg.header = header;
        pose_msg.header.frame_id = "map";

        pose_msg.pose.pose.position.x = mean_particle.x;
        pose_msg.pose.pose.position.y = mean_particle.y;
        pose_msg.pose.pose.position.z = mean_particle.z;

        // Convert Euler angles to quaternion
        tf::Quaternion quat;
        quat.setRPY(mean_particle.roll, mean_particle.pitch, mean_particle.yaw);
        pose_msg.pose.pose.orientation.x = quat.x();
        pose_msg.pose.pose.orientation.y = quat.y();
        pose_msg.pose.pose.orientation.z = quat.z();
        pose_msg.pose.pose.orientation.w = quat.w();

        // Set covariance to identity for simplicity (in practice, calculate from particles)
        for (int i = 0; i < 36; ++i) {
            pose_msg.pose.covariance[i] = 0.0;
        }
        pose_msg.pose.covariance[0] = 0.1;   // x
        pose_msg.pose.covariance[7] = 0.1;   // y
        pose_msg.pose.covariance[14] = 0.1;  // z
        pose_msg.pose.covariance[21] = 0.01; // rx
        pose_msg.pose.covariance[28] = 0.01; // ry
        pose_msg.pose.covariance[35] = 0.01; // rz

        pose_pub_.publish(pose_msg);
    }
};
```

## Digital Twin-Specific Fusion Considerations

### Temporal Alignment and Synchronization

In digital twin applications, it's crucial to maintain temporal consistency between the physical system and its virtual counterpart:

```cpp
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseStamped.h>

class DigitalTwinFusion {
public:
    DigitalTwinFusion(ros::NodeHandle& nh) : nh_(nh) {
        // Use message filters for proper synchronization
        imu_sub_.reset(new message_filters::Subscriber<sensor_msgs::Imu>(
            nh_, "/imu/data", 100));
        lidar_sub_.reset(new message_filters::Subscriber<sensor_msgs::LaserScan>(
            nh_, "/laser_scan", 10));
        pose_sub_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(
            nh_, "/ground_truth_pose", 10));  // If available for digital twin validation

        // Synchronize with larger time window for digital twin applications
        sync_.reset(new Synchronizer(
            SyncPolicy(10),
            *imu_sub_, *lidar_sub_, *pose_sub_));
        sync_->registerCallback(boost::bind(&DigitalTwinFusion::synchronizedCallback, this, _1, _2, _3));

        // Publishers
        fused_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/digital_twin/estimated_pose", 10);
        state_error_pub_ = nh_.advertise<geometry_msgs::Vector3>("/digital_twin/state_error", 10);

        ROS_INFO("Digital Twin Fusion initialized with synchronization");
    }

private:
    ros::NodeHandle& nh_;
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Imu, sensor_msgs::LaserScan, geometry_msgs::PoseStamped> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Imu>> imu_sub_;
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::LaserScan>> lidar_sub_;
    boost::shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
    boost::shared_ptr<Synchronizer> sync_;

    ros::Publisher fused_pose_pub_;
    ros::Publisher state_error_pub_;

    void synchronizedCallback(const sensor_msgs::Imu::ConstPtr& imu_msg,
                             const sensor_msgs::LaserScan::ConstPtr& lidar_msg,
                             const geometry_msgs::PoseStamped::ConstPtr& truth_msg) {
        // Process synchronized sensor data for digital twin
        processFusion(imu_msg, lidar_msg, truth_msg);

        // Calculate error between estimated and true state (for validation)
        calculateStateError(truth_msg);
    }

    void processFusion(const sensor_msgs::Imu::ConstPtr& imu_msg,
                      const sensor_msgs::LaserScan::ConstPtr& lidar_msg,
                      const geometry_msgs::PoseStamped::ConstPtr& truth_msg) {
        // Implement your fusion algorithm here
        // This is where you'd call your EKF, UKF, or particle filter

        geometry_msgs::PoseStamped fused_pose;
        fused_pose.header = imu_msg->header;  // Use the synchronized timestamp
        fused_pose.header.frame_id = "map";

        // For this example, we'll just publish the IMU orientation and LiDAR position
        // In practice, you'd use your fusion algorithm
        fused_pose.pose.orientation = imu_msg->orientation;

        // Extract position from LiDAR (simplified)
        extractPositionFromLidar(*lidar_msg, fused_pose.pose.position);

        fused_pose_pub_.publish(fused_pose);
    }

    void extractPositionFromLidar(const sensor_msgs::LaserScan& scan,
                                 geometry_msgs::Point& position) {
        // Simplified position extraction - in practice, use landmark detection
        std::vector<float> valid_ranges;
        std::vector<float> angles;

        for (size_t i = 0; i < scan.ranges.size(); ++i) {
            float range = scan.ranges[i];
            if (!std::isnan(range) && !std::isinf(range) &&
                range >= scan.range_min && range <= scan.range_max) {
                valid_ranges.push_back(range);
                angles.push_back(scan.angle_min + i * scan.angle_increment);
            }
        }

        if (!valid_ranges.empty()) {
            // Calculate centroid of detected obstacles
            double sum_x = 0, sum_y = 0;
            for (size_t i = 0; i < valid_ranges.size(); ++i) {
                sum_x += valid_ranges[i] * cos(angles[i]);
                sum_y += valid_ranges[i] * sin(angles[i]);
            }

            position.x = sum_x / valid_ranges.size();
            position.y = sum_y / valid_ranges.size();
            position.z = 0;  // Assume 2D for simplicity
        }
    }

    void calculateStateError(const geometry_msgs::PoseStamped::ConstPtr& truth_msg) {
        // In a real digital twin, you'd compare your estimate with ground truth
        // For this example, we'll just publish a zero error
        geometry_msgs::Vector3 error_msg;
        error_msg.x = 0;
        error_msg.y = 0;
        error_msg.z = 0;

        state_error_pub_.publish(error_msg);
    }
};
```

## Adaptive Fusion for Changing Conditions

Digital twins operate in dynamic environments, requiring adaptive fusion strategies:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32

class AdaptiveFusion:
    def __init__(self):
        rospy.init_node('adaptive_fusion')

        # Subscribe to sensors
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.lidar_sub = rospy.Subscriber('/laser_scan', LaserScan, self.lidar_callback)

        # Publishers
        self.fused_pose_pub = rospy.Publisher('/adaptive_fused_pose', PoseWithCovarianceStamped, queue_size=10)
        self.confidence_pub = rospy.Publisher('/sensor_confidence', Float32, queue_size=10)

        # Adaptive parameters
        self.imu_trust = 0.7  # Initial trust in IMU
        self.lidar_trust = 0.7  # Initial trust in LiDAR
        self.adaptation_rate = 0.01  # How fast to adapt trust levels

        # Sensor quality tracking
        self.imu_quality_history = []
        self.lidar_quality_history = []
        self.max_history = 50

        # State estimation (simplified - in practice, use proper filter)
        self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.covariance = np.eye(6) * 0.1

        rospy.loginfo("Adaptive Fusion initialized")

    def imu_callback(self, msg):
        # Evaluate IMU quality
        imu_quality = self.evaluate_imu_quality(msg)
        self.imu_quality_history.append(imu_quality)
        if len(self.imu_quality_history) > self.max_history:
            self.imu_quality_history.pop(0)

        # Update trust based on recent quality
        avg_imu_quality = np.mean(self.imu_quality_history) if self.imu_quality_history else 0.5
        self.imu_trust = self.adapt_trust(self.imu_trust, avg_imu_quality)

        # Perform fusion with adaptive weights
        self.perform_adaptive_fusion('imu', msg)

    def lidar_callback(self, msg):
        # Evaluate LiDAR quality
        lidar_quality = self.evaluate_lidar_quality(msg)
        self.lidar_quality_history.append(lidar_quality)
        if len(self.lidar_quality_history) > self.max_history:
            self.lidar_quality_history.pop(0)

        # Update trust based on recent quality
        avg_lidar_quality = np.mean(self.lidar_quality_history) if self.lidar_quality_history else 0.5
        self.lidar_trust = self.adapt_trust(self.lidar_trust, avg_lidar_quality)

        # Perform fusion with adaptive weights
        self.perform_adaptive_fusion('lidar', msg)

    def evaluate_imu_quality(self, msg):
        """Evaluate the quality of IMU data"""
        # Check for unrealistic values
        ang_vel_norm = np.linalg.norm([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        lin_acc_norm = np.linalg.norm([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Quality score based on reasonableness of measurements
        quality = 1.0
        if ang_vel_norm > 10.0:  # Too high angular velocity
            quality *= 0.1
        elif ang_vel_norm > 5.0:
            quality *= 0.5

        if abs(lin_acc_norm - 9.81) > 5.0:  # Too far from gravity
            quality *= 0.5

        return quality

    def evaluate_lidar_quality(self, msg):
        """Evaluate the quality of LiDAR data"""
        # Count valid measurements
        valid_ranges = [r for r in msg.ranges if not (np.isnan(r) or np.isinf(r))]
        valid_ratio = len(valid_ranges) / len(msg.ranges) if msg.ranges else 0

        # Quality based on data completeness
        quality = min(1.0, valid_ratio * 2)  # Boost for high validity

        # Additional checks could include:
        # - Consistency with previous scans
        # - Detection of expected landmarks
        # - Noise level analysis

        return quality

    def adapt_trust(self, current_trust, quality):
        """Adapt trust level based on sensor quality"""
        # Increase trust if quality is high, decrease if low
        if quality > 0.7:
            current_trust = min(1.0, current_trust + self.adaptation_rate)
        elif quality < 0.3:
            current_trust = max(0.1, current_trust - self.adaptation_rate)
        else:
            # For medium quality, move slowly toward 0.7 (desired operating point)
            if current_trust < 0.7:
                current_trust = min(0.7, current_trust + self.adaptation_rate * 0.5)
            else:
                current_trust = max(0.7, current_trust - self.adaptation_rate * 0.5)

        return current_trust

    def perform_adaptive_fusion(self, sensor_type, sensor_data):
        """Perform fusion with adaptive weights based on sensor trust"""
        if sensor_type == 'imu':
            # Update state based on IMU with adaptive trust
            self.update_state_imu(sensor_data)
        elif sensor_type == 'lidar':
            # Update state based on LiDAR with adaptive trust
            self.update_state_lidar(sensor_data)

        # Publish fused state
        self.publish_fused_state()

        # Publish overall confidence
        avg_trust = (self.imu_trust + self.lidar_trust) / 2
        self.confidence_pub.publish(Float32(avg_trust))

    def update_state_imu(self, msg):
        """Update state estimate using IMU data with adaptive trust"""
        dt = 0.01  # Assume 100Hz IMU for this example

        # Extract measurements
        angular_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        linear_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        # Simple prediction (in practice, use proper integration)
        self.state[3:6] += linear_acc * dt  # Update velocities
        self.state[0:3] += self.state[3:6] * dt  # Update positions

        # Adjust covariance based on IMU trust
        trust_factor = 1.0 / max(0.1, self.imu_trust)  # Higher factor for lower trust
        self.covariance[3:6, 3:6] *= (1 + (1 - self.imu_trust) * 0.1)  # Increase uncertainty if low trust

    def update_state_lidar(self, msg):
        """Update state estimate using LiDAR data with adaptive trust"""
        # Extract position information from LiDAR
        # This is simplified - in practice, use landmark detection or scan matching
        position_estimate = self.extract_position_from_scan(msg)

        # Update position with adaptive trust
        if position_estimate is not None:
            # Weighted update based on trust
            position_weight = self.lidar_trust
            self.state[0:3] = (1 - position_weight) * self.state[0:3] + position_weight * position_estimate

            # Adjust position covariance based on LiDAR trust
            trust_factor = 1.0 / max(0.1, self.lidar_trust)
            self.covariance[0:3, 0:3] *= (1 - self.lidar_trust * 0.1)  # Decrease uncertainty if high trust

    def extract_position_from_scan(self, scan_msg):
        """Extract position estimate from LiDAR scan"""
        # Simplified extraction - in practice, use proper SLAM or landmark detection
        valid_ranges = []
        angles = []

        for i, range_val in enumerate(scan_msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)):
                if scan_msg.range_min <= range_val <= scan_msg.range_max:
                    valid_ranges.append(range_val)
                    angle = scan_msg.angle_min + i * scan_msg.angle_increment
                    angles.append(angle)

        if valid_ranges:
            # Calculate centroid of detected obstacles
            sum_x = sum(r * np.cos(a) for r, a in zip(valid_ranges, angles))
            sum_y = sum(r * np.sin(a) for r, a in zip(valid_ranges, angles))

            avg_x = sum_x / len(valid_ranges)
            avg_y = sum_y / len(valid_ranges)

            return np.array([avg_x, avg_y, 0.0])

        return None

    def publish_fused_state(self):
        """Publish the fused state estimate"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = self.state[0]
        pose_msg.pose.pose.position.y = self.state[1]
        pose_msg.pose.pose.position.z = self.state[2]

        # For this example, orientation is not updated
        pose_msg.pose.pose.orientation.w = 1.0

        # Copy covariance matrix
        for i in range(6):
            for j in range(6):
                idx = i * 6 + j
                if i < self.covariance.shape[0] and j < self.covariance.shape[1]:
                    pose_msg.pose.covariance[idx] = self.covariance[i, j]
                else:
                    pose_msg.pose.covariance[idx] = 0.0

        self.fused_pose_pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        fusion = AdaptiveFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Fusion Quality Assessment in Digital Twins

### Consistency Monitoring

```python
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray

class FusionQualityMonitor:
    def __init__(self):
        rospy.init_node('fusion_quality_monitor')

        # Subscribe to all sensor types and fused output
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.lidar_sub = rospy.Subscriber('/laser_scan', LaserScan, self.lidar_callback)
        self.fused_sub = rospy.Subscriber('/fused_pose', PoseWithCovarianceStamped, self.fused_callback)

        # Publisher for quality metrics
        self.quality_pub = rospy.Publisher('/fusion_quality_metrics', Float32MultiArray, queue_size=10)

        # Data storage for consistency checks
        self.imu_history = []
        self.lidar_history = []
        self.fused_history = []
        self.max_history = 100

        # Quality metrics
        self.consistency_score = 1.0
        self.divergence_score = 0.0
        self.confidence_score = 1.0

        rospy.loginfo("Fusion Quality Monitor initialized")

    def imu_callback(self, msg):
        # Store IMU data with timestamp
        self.imu_history.append({
            'timestamp': msg.header.stamp.to_sec(),
            'angular_velocity': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
            'linear_acceleration': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]),
            'orientation': np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        })

        if len(self.imu_history) > self.max_history:
            self.imu_history.pop(0)

    def lidar_callback(self, msg):
        # Store LiDAR data with timestamp
        self.lidar_history.append({
            'timestamp': msg.header.stamp.to_sec(),
            'ranges': np.array(msg.ranges)
        })

        if len(self.lidar_history) > self.max_history:
            self.lidar_history.pop(0)

    def fused_callback(self, msg):
        # Store fused state with timestamp
        self.fused_history.append({
            'timestamp': msg.header.stamp.to_sec(),
            'position': np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]),
            'orientation': np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                   msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]),
            'covariance': np.array(msg.pose.covariance).reshape(6, 6)
        })

        if len(self.fused_history) > self.max_history:
            self.fused_history.pop(0)

        # Calculate quality metrics
        self.calculate_consistency_metrics()
        self.calculate_divergence_metrics()
        self.calculate_confidence_metrics()

        # Publish quality metrics
        self.publish_quality_metrics()

    def calculate_consistency_metrics(self):
        """Calculate consistency between different sensor estimates"""
        if len(self.fused_history) < 2:
            return

        # Calculate consistency based on covariance and actual measurements
        latest_state = self.fused_history[-1]
        prev_state = self.fused_history[-2]

        # Position consistency check
        pos_diff = np.linalg.norm(latest_state['position'] - prev_state['position'])
        pos_uncertainty = np.sqrt(np.mean(np.diag(latest_state['covariance'][:3, :3])))

        if pos_uncertainty > 0:
            consistency_ratio = pos_diff / pos_uncertainty
            # Lower ratio indicates better consistency
            self.consistency_score = max(0.0, min(1.0, 2.0 - consistency_ratio))
        else:
            self.consistency_score = 1.0

    def calculate_divergence_metrics(self):
        """Calculate if the fusion is diverging"""
        if len(self.fused_history) < 10:
            return

        # Check if covariance is growing exponentially
        cov_trace_history = [np.trace(state['covariance']) for state in self.fused_history[-10:]]

        if len(set(cov_trace_history)) > 1:  # Not all values are the same
            # Calculate trend
            if len(cov_trace_history) > 1:
                trend = (cov_trace_history[-1] - cov_trace_history[0]) / len(cov_trace_history)
                # Higher trend indicates potential divergence
                self.divergence_score = min(1.0, trend * 100)  # Scale appropriately
        else:
            self.divergence_score = 0.0

    def calculate_confidence_metrics(self):
        """Calculate overall confidence in the fusion result"""
        if not self.fused_history:
            return

        latest_cov = self.fused_history[-1]['covariance']

        # Confidence is inversely related to uncertainty (determinant of covariance)
        cov_det = np.linalg.det(latest_cov)

        # Normalize based on expected covariance bounds
        if cov_det > 0:
            # Lower determinant = higher confidence
            self.confidence_score = max(0.0, min(1.0, 1.0 / (1 + cov_det * 0.1)))
        else:
            self.confidence_score = 0.5  # Default if determinant is 0 or negative

    def publish_quality_metrics(self):
        """Publish fusion quality metrics"""
        metrics_msg = Float32MultiArray()
        metrics_msg.data = [
            float(self.consistency_score),
            float(self.divergence_score),
            float(self.confidence_score)
        ]

        self.quality_pub.publish(metrics_msg)

        # Log quality status
        if self.confidence_score < 0.3:
            rospy.logwarn(f"Fusion quality LOW - Consistency: {self.consistency_score:.2f}, "
                         f"Divergence: {self.divergence_score:.2f}, Confidence: {self.confidence_score:.2f}")
        elif self.confidence_score < 0.7:
            rospy.loginfo_throttle(10.0, f"Fusion quality MODERATE - Consistency: {self.consistency_score:.2f}, "
                            f"Divergence: {self.divergence_score:.2f}, Confidence: {self.confidence_score:.2f}")
        else:
            rospy.loginfo_throttle(10.0, f"Fusion quality GOOD - Consistency: {self.consistency_score:.2f}, "
                            f"Divergence: {self.divergence_score:.2f}, Confidence: {self.confidence_score:.2f}")

if __name__ == '__main__':
    try:
        monitor = FusionQualityMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Best Practices for Digital Twin Sensor Fusion

### 1. Modular Architecture
- Design fusion components as modular, replaceable units
- Use plugin architectures to easily swap fusion algorithms
- Implement clear interfaces between components

### 2. Fault Tolerance
- Implement graceful degradation when sensors fail
- Use voting mechanisms for redundant sensors
- Monitor sensor health continuously

### 3. Computational Efficiency
- Optimize for real-time performance
- Use appropriate fusion frequency based on application needs
- Implement multi-threading where possible

### 4. Validation and Testing
- Implement comprehensive validation pipelines
- Use simulation to test edge cases
- Continuously validate against ground truth when available

## Next Steps

In the next section, we'll add comprehensive citations to sensor simulation research papers, providing academic foundations for the techniques covered in this chapter.

## Exercises

1. **Implementation Exercise**: Create a complete sensor fusion pipeline that combines LiDAR, camera, and IMU data using an Extended Kalman Filter, and evaluate its performance against individual sensors.

2. **Adaptation Exercise**: Implement an adaptive fusion system that adjusts its algorithm parameters based on changing environmental conditions.

3. **Digital Twin Exercise**: Design a fusion system specifically for digital twin applications that maintains synchronization with a physical system and provides real-time state estimation.
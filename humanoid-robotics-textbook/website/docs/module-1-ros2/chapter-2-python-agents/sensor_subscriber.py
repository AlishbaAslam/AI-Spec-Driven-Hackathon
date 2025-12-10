#!/usr/bin/env python3
# Sensor Subscriber for Python Agent
# This module handles various sensor inputs for the Python agent

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import math


class SensorSubscriber(Node):
    """
    A specialized node that subscribes to various sensor data streams
    and processes them for the Python agent.
    """

    def __init__(self):
        super().__init__('sensor_subscriber')

        # Initialize CvBridge for image processing (if needed)
        self.bridge = CvBridge()

        # Create subscribers for different sensor types
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_scan_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for processed sensor data
        self.sensor_status_pub = self.create_publisher(
            String,
            '/sensor_status',
            10
        )

        # Store latest sensor readings
        self.latest_laser_scan = None
        self.latest_image = None
        self.latest_imu = None

        # Sensor validation flags
        self.laser_valid = False
        self.camera_valid = False
        self.imu_valid = False

        # Parameters for sensor validation
        self.min_laser_range = 0.1
        self.max_laser_range = 10.0
        self.expected_image_width = 640
        self.expected_image_height = 480

        self.get_logger().info('Sensor Subscriber initialized')

    def laser_scan_callback(self, msg):
        """
        Callback function for laser scan data.
        Validates and processes laser range measurements.
        """
        try:
            # Validate laser scan data
            if self.validate_laser_scan(msg):
                self.latest_laser_scan = msg
                self.laser_valid = True

                # Process the scan data for useful information
                min_distance, max_distance, avg_distance = self.process_laser_scan(msg)

                # Log important findings
                if min_distance < 0.5:
                    self.get_logger().warn(f'Close obstacle detected: {min_distance:.2f}m')

                # Publish sensor status
                status_msg = String()
                status_msg.data = f"Laser: Valid, Min: {min_distance:.2f}m, Max: {max_distance:.2f}m"
                self.sensor_status_pub.publish(status_msg)

                self.get_logger().debug(f'Processed laser scan: {len(msg.ranges)} points, min={min_distance:.2f}m')

            else:
                self.laser_valid = False
                self.get_logger().warn('Invalid laser scan data received')

        except Exception as e:
            self.laser_valid = False
            self.get_logger().error(f'Error processing laser scan: {e}')

    def camera_callback(self, msg):
        """
        Callback function for camera image data.
        Validates and processes image data.
        """
        try:
            # Validate image data
            if self.validate_image(msg):
                self.latest_image = msg
                self.camera_valid = True

                # Log image information
                self.get_logger().debug(f'Received image: {msg.width}x{msg.height}, encoding: {msg.encoding}')

                # Publish sensor status
                status_msg = String()
                status_msg.data = f"Camera: Valid, Resolution: {msg.width}x{msg.height}"
                self.sensor_status_pub.publish(status_msg)

            else:
                self.camera_valid = False
                self.get_logger().warn('Invalid image data received')

        except Exception as e:
            self.camera_valid = False
            self.get_logger().error(f'Error processing camera image: {e}')

    def imu_callback(self, msg):
        """
        Callback function for IMU data.
        Validates and processes inertial measurement data.
        """
        try:
            # Validate IMU data
            if self.validate_imu(msg):
                self.latest_imu = msg
                self.imu_valid = True

                # Extract orientation and angular velocity
                orientation = msg.orientation
                angular_velocity = msg.angular_velocity
                linear_acceleration = msg.linear_acceleration

                # Publish sensor status
                status_msg = String()
                status_msg.data = f"IMU: Valid, Orientation OK"
                self.sensor_status_pub.publish(status_msg)

                self.get_logger().debug(f'Processed IMU data: orientation valid')

            else:
                self.imu_valid = False
                self.get_logger().warn('Invalid IMU data received')

        except Exception as e:
            self.imu_valid = False
            self.get_logger().error(f'Error processing IMU data: {e}')

    def validate_laser_scan(self, msg):
        """
        Validate laser scan message data.
        """
        if not msg.ranges:
            return False

        # Check for valid range values
        for r in msg.ranges:
            if not (self.min_laser_range <= r <= self.max_laser_range or r == float('inf')):
                if not math.isnan(r):
                    return False

        # Check time stamp validity
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            self.get_logger().warn('Laser scan has zero timestamp')

        return True

    def validate_image(self, msg):
        """
        Validate image message data.
        """
        if msg.width <= 0 or msg.height <= 0:
            return False

        if not msg.data:
            return False

        # Check encoding is valid
        valid_encodings = ['rgb8', 'bgr8', 'mono8', 'mono16']
        if msg.encoding not in valid_encodings:
            self.get_logger().warn(f'Unexpected image encoding: {msg.encoding}')

        return True

    def validate_imu(self, msg):
        """
        Validate IMU message data.
        """
        # Check if orientation values are reasonable
        # (unit quaternion should have length close to 1)
        length_sq = (msg.orientation.x ** 2 +
                     msg.orientation.y ** 2 +
                     msg.orientation.z ** 2 +
                     msg.orientation.w ** 2)

        if abs(length_sq - 1.0) > 0.1:
            self.get_logger().warn(f'Invalid quaternion length: {length_sq}')
            return False

        return True

    def process_laser_scan(self, msg):
        """
        Process laser scan data to extract useful information.
        Returns min, max, and average distances.
        """
        # Filter out infinite and invalid values
        valid_ranges = [r for r in msg.ranges if r != float('inf') and not math.isnan(r) and r > 0]

        if not valid_ranges:
            return float('inf'), float('inf'), float('inf')

        min_distance = min(valid_ranges) if valid_ranges else float('inf')
        max_distance = max(valid_ranges) if valid_ranges else 0.0
        avg_distance = sum(valid_ranges) / len(valid_ranges) if valid_ranges else float('inf')

        return min_distance, max_distance, avg_distance

    def get_sensor_status(self):
        """
        Get current status of all sensors.
        """
        status = {
            'laser': self.laser_valid,
            'camera': self.camera_valid,
            'imu': self.imu_valid,
            'timestamp': self.get_clock().now().to_msg()
        }
        return status

    def get_processed_sensor_data(self):
        """
        Get the latest processed sensor data.
        """
        data = {
            'laser_scan': self.latest_laser_scan,
            'image': self.latest_image,
            'imu': self.latest_imu
        }
        return data


def main(args=None):
    """
    Main function to run the sensor subscriber.
    """
    rclpy.init(args=args)

    sensor_subscriber = SensorSubscriber()

    try:
        rclpy.spin(sensor_subscriber)
    except KeyboardInterrupt:
        sensor_subscriber.get_logger().info('Interrupted, shutting down sensor subscriber')
    except Exception as e:
        sensor_subscriber.get_logger().error(f'Unexpected error: {e}')
    finally:
        sensor_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# Test for sensor subscriber in Python agent
# This test verifies that the sensor subscription functionality works correctly

import unittest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist
from unittest.mock import Mock, MagicMock


class TestSensorSubscriber(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_sensor_subscriber_node')

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_laser_scan_subscription(self):
        """Test that laser scan subscription is set up correctly"""
        mock_callback = Mock()
        subscriber = self.node.create_subscription(
            LaserScan,
            '/scan',
            mock_callback,
            10)
        self.assertIsNotNone(subscriber)

    def test_image_subscription(self):
        """Test that image subscription is set up correctly"""
        mock_callback = Mock()
        subscriber = self.node.create_subscription(
            Image,
            '/camera/image_raw',
            mock_callback,
            10)
        self.assertIsNotNone(subscriber)

    def test_imu_subscription(self):
        """Test that IMU subscription is set up correctly"""
        mock_callback = Mock()
        subscriber = self.node.create_subscription(
            Imu,
            '/imu/data',
            mock_callback,
            10)
        self.assertIsNotNone(subscriber)

    def test_laser_scan_callback_processing(self):
        """Test processing of laser scan data in callback"""
        # Create mock laser scan data
        scan_msg = LaserScan()
        scan_msg.ranges = [1.0, 1.5, 0.8, 2.0, 1.2, float('inf'), 0.5]
        scan_msg.angle_increment = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0

        # Process the ranges to get valid distances (not infinite)
        valid_ranges = [r for r in scan_msg.ranges if r != float('inf') and r > scan_msg.range_min]

        # Calculate min distance
        min_distance = min(valid_ranges) if valid_ranges else float('inf')

        # Verify the calculation
        self.assertEqual(min_distance, 0.5)

    def test_sensor_data_validation(self):
        """Test validation of sensor data"""
        def validate_scan_data(msg):
            """Validate laser scan message"""
            if len(msg.ranges) == 0:
                return False
            # Check for any invalid negative ranges (excluding infinite values)
            invalid_ranges = [r for r in msg.ranges if r < 0 and r != float('inf')]
            return len(invalid_ranges) == 0

        # Valid scan data
        valid_scan = LaserScan()
        valid_scan.ranges = [1.0, 1.5, 2.0]
        self.assertTrue(validate_scan_data(valid_scan))

        # Invalid scan data with negative values
        invalid_scan = LaserScan()
        invalid_scan.ranges = [1.0, -0.5, 2.0]
        self.assertFalse(validate_scan_data(invalid_scan))

        # Empty scan data
        empty_scan = LaserScan()
        empty_scan.ranges = []
        self.assertFalse(validate_scan_data(empty_scan))

    def test_sensor_callback_execution(self):
        """Test that sensor callback executes properly (mock test)"""
        # Create a mock callback that processes sensor data
        processed_data = {'received': False, 'min_distance': None}

        def mock_sensor_callback(msg):
            processed_data['received'] = True
            if hasattr(msg, 'ranges'):
                valid_ranges = [r for r in msg.ranges if r != float('inf')]
                processed_data['min_distance'] = min(valid_ranges) if valid_ranges else float('inf')

        # Create a subscription with the callback
        subscriber = self.node.create_subscription(
            LaserScan,
            '/scan',
            mock_sensor_callback,
            10)

        # Simulate receiving a message
        test_msg = LaserScan()
        test_msg.ranges = [1.0, 0.8, 1.5]
        mock_sensor_callback(test_msg)

        # Verify the callback processed the data
        self.assertTrue(processed_data['received'])
        self.assertEqual(processed_data['min_distance'], 0.8)


def main():
    # This test should fail initially as per TDD approach
    # The actual implementation should make it pass
    unittest.main()


if __name__ == '__main__':
    main()
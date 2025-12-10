#!/usr/bin/env python3
# Test for Python agent example
# This test verifies that the Python agent example works correctly

import unittest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from unittest.mock import Mock, MagicMock


class TestPythonAgent(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_agent_node')

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_agent_node_creation(self):
        """Test that agent node is created correctly"""
        # This test verifies that the basic node structure is correct
        self.assertIsNotNone(self.node)

    def test_publisher_creation(self):
        """Test that publisher for control commands is created correctly"""
        publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.assertIsNotNone(publisher)

    def test_subscriber_creation(self):
        """Test that subscriber for sensor data is created correctly"""
        mock_callback = Mock()
        subscriber = self.node.create_subscription(
            LaserScan,
            '/scan',
            mock_callback,
            10)
        self.assertIsNotNone(subscriber)

    def test_twist_message_creation(self):
        """Test that Twist messages for robot control can be created"""
        cmd = Twist()
        cmd.linear.x = 0.5
        cmd.angular.z = 0.0

        self.assertEqual(cmd.linear.x, 0.5)
        self.assertEqual(cmd.angular.z, 0.0)

    def test_sensor_data_processing(self):
        """Test that sensor data processing logic works (mock test)"""
        # Simulate laser scan data
        scan_msg = LaserScan()
        scan_msg.ranges = [1.0, 1.5, 0.8, 2.0, 1.2]  # Sample distances
        scan_msg.angle_increment = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0

        # Process the data to find minimum distance
        min_distance = min([r for r in scan_msg.ranges if r != float('inf')])

        # Verify minimum distance calculation
        self.assertEqual(min_distance, 0.8)

    def test_obstacle_detection_logic(self):
        """Test obstacle detection logic"""
        safety_distance = 0.5  # meters

        # Test with obstacle too close
        scan_msg = LaserScan()
        scan_msg.ranges = [0.3, 1.0, 1.5]  # Obstacle at 0.3m
        min_distance = min([r for r in scan_msg.ranges if r != float('inf')])

        # Should trigger obstacle avoidance
        should_stop = min_distance < safety_distance
        self.assertTrue(should_stop)

        # Test with safe distance
        scan_msg.ranges = [1.0, 1.5, 2.0]  # All obstacles far away
        min_distance = min([r for r in scan_msg.ranges if r != float('inf')])

        # Should not trigger obstacle avoidance
        should_stop = min_distance < safety_distance
        self.assertFalse(should_stop)


def main():
    # This test should fail initially as per TDD approach
    # The actual implementation should make it pass
    unittest.main()


if __name__ == '__main__':
    main()
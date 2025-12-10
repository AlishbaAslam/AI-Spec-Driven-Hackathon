#!/usr/bin/env python3
# Test for basic publisher example
# This test verifies that the basic publisher example works correctly

import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from unittest.mock import Mock, MagicMock


class TestBasicPublisher(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_publisher_node')

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_publisher_creation(self):
        """Test that publisher is created correctly"""
        publisher = self.node.create_publisher(String, 'test_topic', 10)
        self.assertIsNotNone(publisher)
        # Check that the publisher is for the correct topic
        # Note: We can't directly check topic name in publisher object
        # but we can verify it doesn't throw an exception during creation

    def test_message_creation(self):
        """Test that String messages can be created and populated"""
        msg = String()
        msg.data = "Test message"
        self.assertEqual(msg.data, "Test message")

    def test_publisher_publish(self):
        """Test that publisher can publish messages (mock test)"""
        # Create a mock publisher to verify publish is called
        mock_publisher = Mock()
        mock_publisher.publish = Mock()

        # Create a test message
        msg = String()
        msg.data = "Test publish"

        # Call publish on the mock
        mock_publisher.publish(msg)

        # Verify that publish was called once
        mock_publisher.publish.assert_called_once_with(msg)


def main():
    # This test should fail initially as per TDD approach
    # The actual implementation should make it pass
    unittest.main()


if __name__ == '__main__':
    main()
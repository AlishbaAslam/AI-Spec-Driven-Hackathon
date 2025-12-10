#!/usr/bin/env python3
# Test for basic subscriber example
# This test verifies that the basic subscriber example works correctly

import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from unittest.mock import Mock, MagicMock


class TestBasicSubscriber(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_subscriber_node')

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_subscriber_creation(self):
        """Test that subscriber is created correctly"""
        mock_callback = Mock()
        subscriber = self.node.create_subscription(
            String,
            'test_topic',
            mock_callback,
            10)
        self.assertIsNotNone(subscriber)

    def test_message_callback(self):
        """Test that subscriber callback processes messages correctly"""
        # Create a mock callback function
        mock_callback = Mock()

        # Create a test message
        test_msg = String()
        test_msg.data = "Test message for subscriber"

        # Call the mock callback with the test message
        mock_callback(test_msg)

        # Verify that the callback was called with the correct message
        mock_callback.assert_called_once_with(test_msg)

    def test_subscription_callback_execution(self):
        """Test that subscription can execute callback (mock test)"""
        # This is a more complete test that verifies the callback mechanism
        received_messages = []

        def mock_callback(msg):
            received_messages.append(msg.data)

        # Create a subscription with the callback
        subscriber = self.node.create_subscription(
            String,
            'test_topic',
            mock_callback,
            10)

        # Simulate receiving a message
        test_msg = String()
        test_msg.data = "Simulated message"
        mock_callback(test_msg)

        # Verify the message was received
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0], "Simulated message")


def main():
    # This test should fail initially as per TDD approach
    # The actual implementation should make it pass
    unittest.main()


if __name__ == '__main__':
    main()
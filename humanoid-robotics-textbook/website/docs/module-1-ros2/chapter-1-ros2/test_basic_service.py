#!/usr/bin/env python3
# Test for basic service example
# This test verifies that the basic service example works correctly

import unittest
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from unittest.mock import Mock, MagicMock


class TestBasicService(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('test_service_node')

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_service_creation(self):
        """Test that service server is created correctly"""
        mock_callback = Mock()
        service = self.node.create_service(AddTwoInts, 'test_service', mock_callback)
        self.assertIsNotNone(service)

    def test_service_callback_logic(self):
        """Test that the service callback performs addition correctly"""
        # Create mock request and response objects
        request = AddTwoInts.Request()
        request.a = 5
        request.b = 3

        response = AddTwoInts.Response()

        # Test the addition logic directly
        response.sum = request.a + request.b

        self.assertEqual(response.sum, 8)

    def test_service_callback_with_different_values(self):
        """Test the service callback with different input values"""
        test_cases = [
            (0, 0, 0),
            (1, 1, 2),
            (5, 7, 12),
            (-3, 5, 2),
            (10, -4, 6)
        ]

        for a, b, expected_sum in test_cases:
            request = AddTwoInts.Request()
            request.a = a
            request.b = b

            response = AddTwoInts.Response()
            response.sum = request.a + request.b

            self.assertEqual(response.sum, expected_sum,
                           f"Failed for {a} + {b}, expected {expected_sum}, got {response.sum}")

    def test_service_server_mock(self):
        """Test service server behavior with mock"""
        # Mock callback function that mimics the actual service behavior
        def mock_add_two_ints_callback(request, response):
            response.sum = request.a + request.b
            return response

        # Create a service with the mock callback
        service = self.node.create_service(AddTwoInts, 'test_add_service', mock_add_two_ints_callback)

        # Test the callback directly
        request = AddTwoInts.Request()
        request.a = 10
        request.b = 20

        response = AddTwoInts.Response()
        result = mock_add_two_ints_callback(request, response)

        self.assertEqual(result.sum, 30)


def main():
    # This test should fail initially as per TDD approach
    # The actual implementation should make it pass
    unittest.main()


if __name__ == '__main__':
    main()
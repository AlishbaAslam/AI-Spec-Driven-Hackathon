#!/usr/bin/env python3

"""
Basic ROS 2 service server example
This node provides a simple service that responds to requests.
"""

import rclpy
from rclpy.node import Node
# Note: For this example to work, you would need to define or use an existing service type
# For demonstration, we'll use a basic example that would work with a custom service
from example_interfaces.srv import AddTwoInts


class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    # Destroy the node explicitly
    minimal_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
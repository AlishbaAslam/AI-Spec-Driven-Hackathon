#!/usr/bin/env python3

"""
Basic ROS 2 service client example
This node calls a service to request computation.
"""

import sys
import rclpy
from rclpy.node import Node
# Note: For this example to work, you would need to define or use an existing service type
# For demonstration, we'll use a basic example that would work with a custom service
from example_interfaces.srv import AddTwoInts


class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))

    minimal_client.get_logger().info(
        f'Result of add_two_ints: for {sys.argv[1]} + {sys.argv[2]} = {response.sum}')

    # Destroy the node explicitly
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
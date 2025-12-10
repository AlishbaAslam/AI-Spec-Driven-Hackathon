# Common Python imports and ROS 2 initialization patterns
# This file demonstrates the standard setup patterns used in ROS 2 Python development

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time

# Standard initialization pattern
def initialize_ros():
    """Initialize ROS 2 and return the initialized context"""
    rclpy.init()
    return rclpy

# Node base class with common utilities
class BaseROSNode(Node):
    """Base class with common ROS 2 utilities and patterns"""

    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f'Node {node_name} initialized')

        # Common QoS profiles
        self.default_qos = QoSProfile(depth=10)
        self.reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.best_effort_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

    def safe_destroy_node(self):
        """Safely destroy the node and shut down ROS"""
        self.get_logger().info('Shutting down node...')
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

# Common publisher pattern
class CommonPublisher(BaseROSNode):
    def __init__(self, node_name, topic_name, msg_type=String, qos_profile=None):
        super().__init__(node_name)

        if qos_profile is None:
            qos_profile = self.default_qos

        self.publisher = self.create_publisher(msg_type, topic_name, qos_profile)
        self.get_logger().info(f'Publisher created for topic: {topic_name}')

# Common subscriber pattern
class CommonSubscriber(BaseROSNode):
    def __init__(self, node_name, topic_name, msg_type=String, callback=None, qos_profile=None):
        super().__init__(node_name)

        if qos_profile is None:
            qos_profile = self.default_qos

        if callback is None:
            callback = self.default_callback

        self.subscription = self.create_subscription(
            msg_type,
            topic_name,
            callback,
            qos_profile
        )
        self.get_logger().info(f'Subscriber created for topic: {topic_name}')

    def default_callback(self, msg):
        self.get_logger().info(f'Received message: {msg}')

# Timer pattern for periodic tasks
class TimedNode(BaseROSNode):
    def __init__(self, node_name, timer_period=1.0):
        super().__init__(node_name)

        self.timer_period = timer_period
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info(f'Timer created with period: {timer_period}s')

    def timer_callback(self):
        """Override this method in subclasses"""
        self.get_logger().info('Timer callback executed')

# Service client pattern
class ServiceClient(BaseROSNode):
    def __init__(self, node_name, service_name, service_type):
        super().__init__(node_name)

        self.client = self.create_client(service_type, service_name)

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Service {service_name} not available, waiting...')

        self.get_logger().info(f'Service client created for: {service_name}')

    def call_service(self, request):
        """Call the service asynchronously"""
        future = self.client.call_async(request)
        return future

# Parameter declaration pattern
class ParameterNode(BaseROSNode):
    def __init__(self, node_name):
        super().__init__(node_name)

        # Declare parameters with default values
        self.declare_parameter('param_name', 'default_value')
        self.declare_parameter('int_param', 42)
        self.declare_parameter('float_param', 3.14)

    def get_param(self, param_name, default_value=None):
        """Get parameter value with optional default"""
        param = self.get_parameter(param_name)
        if param.type == param.Type.NOT_SET and default_value is not None:
            return default_value
        return param.value

# Main execution pattern
def run_node(node_class, node_name, **kwargs):
    """Standard pattern to run a ROS 2 node"""
    rclpy.init()

    try:
        node = node_class(node_name, **kwargs)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.safe_destroy_node()

# Example usage patterns
if __name__ == '__main__':
    # Example 1: Simple publisher
    def example_publisher():
        rclpy.init()
        node = CommonPublisher('example_publisher', 'test_topic')

        msg = String()
        msg.data = 'Hello, ROS 2!'
        node.publisher.publish(msg)
        node.get_logger().info('Published message')

        node.safe_destroy_node()

    # Example 2: Simple subscriber
    def example_subscriber():
        rclpy.init()
        node = CommonSubscriber('example_subscriber', 'test_topic')

        # Run for a short time to receive messages
        rclpy.spin_once(node, timeout_sec=1.0)
        node.safe_destroy_node()

    # Example 3: Timed node
    def example_timed_node():
        class MyTimedNode(TimedNode):
            def timer_callback(self):
                self.get_logger().info('Custom timer callback')

        run_node(MyTimedNode, 'timed_node', timer_period=0.5)
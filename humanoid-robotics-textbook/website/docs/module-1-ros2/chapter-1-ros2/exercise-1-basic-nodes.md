---
title: Exercise 1 - Basic Nodes, Topics, and Services
sidebar_position: 4
---

# Exercise 1 - Basic Nodes, Topics, and Services

In this exercise, you'll create your first ROS 2 nodes, implement publisher/subscriber communication, and create a simple service. This will solidify your understanding of the fundamental ROS 2 concepts.

## Prerequisites

Before starting this exercise, make sure you have:

1. Completed the ROS 2 installation and environment setup
2. Familiarized yourself with the basic ROS 2 concepts
3. A working ROS 2 workspace

## Exercise Objectives

By the end of this exercise, you will:

1. Create a publisher node that sends custom messages
2. Create a subscriber node that receives and processes messages
3. Implement a service server and client
4. Run and test your nodes to observe communication

## Part 1: Publisher Node

Create a publisher node that publishes messages about robot status.

### Step 1: Create the Publisher Node

Create a file named `robot_status_publisher.py` in your workspace:

```python
#!/usr/bin/env python3
# Robot Status Publisher
# Publishes robot status information to a topic

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random


class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_publisher')

        # Create publisher for robot status
        self.publisher = self.create_publisher(String, 'robot_status', 10)

        # Create a timer to publish messages periodically
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Robot status options
        self.status_options = [
            "Operating normally",
            "Low battery",
            "Charging",
            "Maintenance required",
            "Idle"
        ]

        self.get_logger().info('Robot Status Publisher node initialized')

    def timer_callback(self):
        # Select a random status
        status = random.choice(self.status_options)

        # Create message
        msg = String()
        msg.data = f'Robot status: {status}'

        # Publish message
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    robot_status_publisher = RobotStatusPublisher()

    try:
        rclpy.spin(robot_status_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        robot_status_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 2: Run the Publisher

1. Save the file as `robot_status_publisher.py`
2. Make it executable: `chmod +x robot_status_publisher.py`
3. Source your ROS 2 environment: `source /opt/ros/humble/setup.bash`
4. Navigate to your workspace and source it: `cd ~/ros2_ws && source install/setup.bash`
5. Run the publisher: `python3 robot_status_publisher.py`

You should see messages being published every second.

## Part 2: Subscriber Node

Create a subscriber node that listens to the robot status topic.

### Step 1: Create the Subscriber Node

Create a file named `robot_status_subscriber.py`:

```python
#!/usr/bin/env python3
# Robot Status Subscriber
# Subscribes to robot status messages and processes them

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RobotStatusSubscriber(Node):
    def __init__(self):
        super().__init__('robot_status_subscriber')

        # Create subscriber for robot status
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Robot Status Subscriber node initialized')

    def status_callback(self, msg):
        self.get_logger().info(f'Received robot status: {msg.data}')

        # Process the status message
        if 'Low battery' in msg.data:
            self.get_logger().warn('Battery level is low! Consider charging.')
        elif 'Maintenance required' in msg.data:
            self.get_logger().error('Maintenance required! Robot needs attention.')
        elif 'Charging' in msg.data:
            self.get_logger().info('Robot is charging, normal operation paused.')


def main(args=None):
    rclpy.init(args=args)

    robot_status_subscriber = RobotStatusSubscriber()

    try:
        rclpy.spin(robot_status_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        robot_status_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 2: Run Both Publisher and Subscriber

1. In one terminal, run the publisher: `python3 robot_status_publisher.py`
2. In another terminal, source ROS 2 and your workspace, then run the subscriber: `python3 robot_status_subscriber.py`

You should see the subscriber receiving and processing messages from the publisher.

## Part 3: Service Implementation

Create a service that allows querying specific robot information.

### Step 1: Create the Service Server

Create a file named `robot_info_server.py`:

```python
#!/usr/bin/env python3
# Robot Info Service Server
# Provides robot information via a service

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger  # Using Trigger service for simplicity


class RobotInfoServer(Node):
    def __init__(self):
        super().__init__('robot_info_server')

        # Create a service
        self.srv = self.create_service(
            Trigger,
            'get_robot_info',
            self.get_robot_info_callback
        )

        # Sample robot information
        self.robot_info = {
            'name': 'ROSbot-X1',
            'model': 'H-2023',
            'firmware_version': '2.1.5',
            'operational_time': '42 days',
            'last_maintenance': '2023-11-15'
        }

        self.get_logger().info('Robot Info Service Server initialized')

    def get_robot_info_callback(self, request, response):
        self.get_logger().info('Received request for robot information')

        # Format the response
        info_str = "\n".join([f"{key}: {value}" for key, value in self.robot_info.items()])
        response.success = True
        response.message = info_str

        return response


def main(args=None):
    rclpy.init(args=args)

    robot_info_server = RobotInfoServer()

    try:
        rclpy.spin(robot_info_server)
    except KeyboardInterrupt:
        pass
    finally:
        robot_info_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 2: Create the Service Client

Create a file named `robot_info_client.py`:

```python
#!/usr/bin/env python3
# Robot Info Service Client
# Calls the robot info service to get robot information

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger


class RobotInfoClient(Node):
    def __init__(self):
        super().__init__('robot_info_client')

        # Create a client for the service
        self.cli = self.create_client(Trigger, 'get_robot_info')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = Trigger.Request()

    def send_request(self):
        # Call the service asynchronously
        self.future = self.cli.call_async(self.req)
        return self.future


def main(args=None):
    rclpy.init(args=args)

    robot_info_client = RobotInfoClient()

    # Send the request
    future = robot_info_client.send_request()

    try:
        # Wait for the response
        rclpy.spin_until_future_complete(robot_info_client, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                print(f"Robot Information:\n{response.message}")
            else:
                print(f"Service call failed: {response.message}")
        else:
            print('Service call failed')

    except KeyboardInterrupt:
        pass
    finally:
        robot_info_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Run the Service

1. In one terminal, run the service server: `python3 robot_info_server.py`
2. In another terminal, source ROS 2 and your workspace, then run the client: `python3 robot_info_client.py`

You should see the client receive and display the robot information from the server.

## Verification Steps

1. **Check active nodes**:
   ```bash
   ros2 node list
   ```

2. **Check active topics**:
   ```bash
   ros2 topic list
   ```

3. **Check active services**:
   ```bash
   ros2 service list
   ```

4. **Echo the robot status topic** (while publisher is running):
   ```bash
   ros2 topic echo /robot_status std_msgs/msg/String
   ```

## Troubleshooting

### Common Issues

1. **Nodes not communicating**: Make sure both terminals have sourced the ROS 2 environment and your workspace.

2. **Service not found**: Ensure the service server is running before calling the client.

3. **Permission errors**: Make sure your Python files are executable: `chmod +x filename.py`

4. **Package not found**: Make sure you're in the right directory and have sourced your workspace.

## Summary

In this exercise, you've:

1. Created publisher and subscriber nodes for asynchronous communication
2. Implemented a service server and client for synchronous communication
3. Learned how to run multiple nodes and observe their interaction
4. Practiced using ROS 2 command-line tools to inspect the system

These fundamental skills form the basis of all ROS 2 applications. Understanding the publish/subscribe and request/response patterns is crucial for building more complex robotic systems.
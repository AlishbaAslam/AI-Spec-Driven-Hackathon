---
title: Chapter 1 - Fundamentals of ROS 2 Nodes, Topics, and Services
sidebar_position: 1
---

# Chapter 1 - Fundamentals of ROS 2 Nodes, Topics, and Services

Welcome to the fundamentals of ROS 2! This chapter will introduce you to the core concepts that make up the Robot Operating System 2 (ROS 2) middleware.

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the difference between Nodes, Topics, and Services in ROS 2
- Create simple publisher/subscriber examples
- Understand the ROS 2 communication patterns
- Set up a basic ROS 2 workspace and environment

## Table of Contents
1. [Introduction to ROS 2](#introduction-to-ros-2)
2. [ROS 2 Nodes](#ros-2-nodes)
3. [ROS 2 Topics and Messages](#ros-2-topics-and-messages)
4. [ROS 2 Services](#ros-2-services)
5. [Hands-on Exercises](#hands-on-exercises)

## Introduction to ROS 2

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

Unlike traditional operating systems, ROS 2 is not an actual OS but rather a middleware that provides services designed for a heterogeneous computer cluster. It provides standard operating system services such as hardware abstraction, low-level device control, implementation of commonly used functionality, message-passing between processes, and package management.

### Key Features of ROS 2

- **Distributed computing**: Nodes can run on different machines and communicate over a network
- **Language independence**: Support for multiple programming languages (C++, Python, etc.)
- **Rich development tools**: Visualization, debugging, and profiling tools
- **Package management**: Easy distribution and reuse of robot software
- **Real-time support**: Better support for real-time systems compared to ROS 1
- **Improved security**: Built-in security features for safer robot operation

### The ROS 2 Ecosystem

ROS 2 consists of several key components that work together to provide a complete robot development environment:

- **DDS (Data Distribution Service)**: The underlying communication middleware
- **RMW (ROS Middleware)**: Abstraction layer over DDS implementations
- **RCL (ROS Client Libraries)**: Language-specific client libraries (rclpy, rclcpp)
- **Rosbags**: Tools for recording and playing back data
- **RViz**: 3D visualization tool for robot data
- **Gazebo**: Physics-based robot simulator
- **Launch**: System for starting multiple nodes at once
- **Parameters**: System for configuration management
- **Actions**: Framework for long-running tasks with feedback

## ROS 2 Nodes

A ROS 2 node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 program. They are organized in a graph structure and communicate with each other using messages.

### Creating a Node

In Python, a node is created by subclassing the `Node` class from `rclpy.node`. Here's a basic example:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle

Nodes in ROS 2 follow a specific lifecycle:
1. **Unconfigured**: Initial state after creation
2. **Inactive**: After configuration, ready to activate
3. **Active**: Running and processing data
4. **Finalized**: Node is shutting down

### Node Communication Overview

```
[Node A] ----publish----> [Topic] <----subscribe---- [Node B]
     |                                                 |
     |                                                 |
     +------publish----> [Topic 2] <----subscribe-----+
```

In this diagram, Node A publishes messages to a topic, which Node B subscribes to. This is the publish/subscribe pattern that forms the backbone of ROS 2 communication.

### Node Best Practices

- Each node should have a single, well-defined responsibility
- Use meaningful node names that reflect the node's function
- Properly handle node cleanup in the destructor
- Use logging to provide useful runtime information
- Consider error handling and recovery mechanisms

## ROS 2 Topics and Messages

Topics enable asynchronous communication between nodes using a publish/subscribe pattern. Messages are the data structures that are passed between nodes.

### Publisher/Subscriber Pattern

- **Publisher**: Node that sends messages to a topic
- **Subscriber**: Node that receives messages from a topic
- **Topic**: Named bus over which nodes exchange messages

### Topic Communication Diagram

```
                    Topic: /sensor_data
                              |
    [Sensor Node] ----publish----> [Message Buffer] <----subscribe---- [Processing Node]
         |                                                        |
         +--(LaserScan msg)                               (LaserScan msg)--+
         |                                                        |
    [Camera Node] ----publish----> [Message Buffer]              |
         |                         (different topic)             |
         +--(Image msg)                                         [Control Node]
```

This diagram illustrates how multiple nodes can publish to different topics, and other nodes can subscribe to the topics they're interested in. Each topic maintains its own message buffer.

### Quality of Service (QoS) Settings

Topics in ROS 2 support Quality of Service settings that define message delivery guarantees:

- **Reliability**: Reliable (all messages delivered) or Best Effort (some messages may be lost)
- **Durability**: Volatile (only new messages) or Transient Local (includes historical messages)
- **History**: Keep All (all messages) or Keep Last N (fixed number of messages)

### Example: Publisher Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Example: Subscriber Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

## ROS 2 Services

Services provide synchronous request/response communication between nodes. Unlike topics, services are blocking and ensure that a response is received for each request.

### Service Client/Server Pattern

- **Service Server**: Node that provides a service
- **Service Client**: Node that requests a service
- **Service**: Named interface that defines the request/response format

### Service Communication Diagram

```
    [Client Node]                           [Server Node]
         |                                        |
         |-------- Request (a=5, b=3) ----------->|
         |                                        |
         |<------- Response (sum=8) --------------|
         |                                        |
    (Blocking - waits for response)        (Processes request,
                                           sends response)
```

This diagram shows the synchronous nature of services: the client node blocks until it receives a response from the server node.

### When to Use Services vs Topics

| Service | Topic |
|---------|-------|
| Synchronous communication | Asynchronous communication |
| Request/response pattern | Continuous data stream |
| Blocking calls | Non-blocking |
| Guaranteed delivery | Delivery based on QoS |
| Good for: Commands, queries, configuration | Good for: Sensor data, status updates, logs |

### Example: Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response
```

### Example: Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
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
```

## Hands-on Exercises

Now that you've learned the fundamentals, it's time to practice with hands-on exercises. Complete the following exercises to reinforce your understanding:

1. Create a basic publisher and subscriber pair that communicate on a custom topic
2. Implement a simple service that performs a mathematical operation
3. Run both the publisher/subscriber and service examples to observe communication

Continue to the next sections for detailed instructions on each exercise.
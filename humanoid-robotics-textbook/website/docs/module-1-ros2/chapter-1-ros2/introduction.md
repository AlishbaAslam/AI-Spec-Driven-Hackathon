---
title: Introduction to ROS 2 Concepts
sidebar_position: 3
---

# Introduction to ROS 2 Concepts

This introduction covers the foundational concepts of ROS 2 (Robot Operating System 2) that you'll need to understand before diving deeper into the educational module. ROS 2 is a flexible framework for writing robot software, providing services designed for a heterogeneous computer cluster.

## What is ROS 2?

ROS 2 is the second generation of the Robot Operating System. Unlike traditional operating systems, ROS 2 is not an actual OS but rather a middleware that provides services designed for robot applications. It provides standard operating system services such as:

- Hardware abstraction
- Low-level device control
- Implementation of commonly used functionality
- Message-passing between processes
- Package management

### Key Improvements over ROS 1

ROS 2 addresses several limitations of the original ROS:

- **Real-time support**: Better support for real-time systems
- **Multi-robot systems**: Improved support for multiple robots
- **Security**: Built-in security features
- **Quality of Service (QoS)**: Configurable message delivery guarantees
- **Cross-platform support**: Better Windows and Mac support
- **DDS-based**: Uses Data Distribution Service (DDS) for communication

## Core Architecture

### Nodes

A **node** is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 program. They are organized in a graph structure and communicate with each other using messages.

Key characteristics of nodes:
- Each node runs a single-threaded event loop by default
- Nodes can be written in different programming languages (C++, Python, etc.)
- Nodes are organized in a distributed system
- Nodes can be started and stopped independently

### Topics and Messages

**Topics** enable asynchronous communication between nodes using a publish/subscribe pattern. **Messages** are the data structures that are passed between nodes.

- **Publisher**: A node that sends messages to a topic
- **Subscriber**: A node that receives messages from a topic
- **Topic**: A named bus over which nodes exchange messages

The publish/subscribe pattern is asynchronous:
- Publishers don't know who is subscribing
- Subscribers don't know who is publishing
- Communication is decoupled in time and space

### Services

**Services** provide synchronous request/response communication between nodes. Unlike topics, services are blocking and ensure that a response is received for each request.

- **Service Server**: A node that provides a service
- **Service Client**: A node that requests a service
- **Service**: A named interface that defines the request/response format

Services are useful when you need guaranteed communication and a response to your request.

### Actions

**Actions** are another communication pattern that extends services for long-running tasks. They provide feedback during execution and the ability to cancel requests.

- **Action Server**: Handles action requests and provides feedback
- **Action Client**: Sends action requests and receives feedback
- **Goal**: The requested action
- **Feedback**: Progress updates during execution
- **Result**: The final outcome

## Quality of Service (QoS)

QoS settings allow you to configure message delivery guarantees for topics:

- **Reliability**: Reliable (all messages delivered) or best-effort (some messages may be lost)
- **Durability**: Volatile (only new messages) or transient-local (includes historical messages)
- **History**: Keep-all (all messages) or keep-last (fixed number of messages)
- **Depth**: Number of messages to keep in history (when using keep-last)

QoS is crucial for configuring communication behavior based on your application's requirements.

## Package and Workspace Structure

### Packages

A **package** is the basic building unit in ROS 2. It contains:

- Source code
- Data files
- Configuration files
- Build instructions
- Dependencies

### Workspaces

A **workspace** is a directory containing one or more packages. The typical structure is:

```
workspace_folder/
├── src/
│   ├── package_1/
│   ├── package_2/
│   └── ...
├── build/
├── install/
└── log/
```

- `src/`: Source code for packages
- `build/`: Build artifacts
- `install/`: Installation directory after building
- `log/`: Build logs

## Client Libraries

ROS 2 provides client libraries for different programming languages:

- **rclcpp**: C++ client library
- **rclpy**: Python client library (used throughout this module)
- **rclrs**: Rust client library
- **rclc**: C client library
- **rcljava**: Java client library

## Communication Patterns Summary

| Pattern | Type | Use Case | Characteristics |
|---------|------|----------|-----------------|
| Topics | Publish/Subscribe | Continuous data streams | Asynchronous, many-to-many |
| Services | Request/Response | Single requests | Synchronous, blocking |
| Actions | Goal-Based | Long-running tasks | Feedback + cancellation |

## ROS 2 Ecosystem

ROS 2 includes many tools and libraries:

- **RViz**: 3D visualization tool
- **Gazebo**: Robot simulation environment
- **rqt**: GUI tools for introspection
- **rosbag**: Data recording and playback
- **ros2cli**: Command-line tools (ros2 run, ros2 launch, etc.)

## Next Steps

Now that you understand the fundamental concepts of ROS 2, you're ready to dive into:

1. Creating your first ROS 2 nodes
2. Implementing publisher/subscriber patterns
3. Working with services and actions
4. Building complete robot applications

The following chapters will guide you through practical implementation of these concepts with hands-on exercises.
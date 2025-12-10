---
title: Cognitive Planning - Using LLMs to Translate Natural Language into ROS 2 Actions
sidebar_label: Cognitive Planning
description: Learn how to use Large Language Models for cognitive planning in robotics, translating natural language into ROS 2 actions
---

# Cognitive Planning: Using LLMs to Translate Natural Language into ROS 2 Actions

## Introduction

This chapter explores cognitive planning in robotics, focusing on how to use Large Language Models (LLMs) to translate natural language commands into ROS 2 actions. Cognitive planning is a critical component of Vision-Language-Action (VLA) systems that bridges human language with robotic capabilities.

## Technical Background

### Understanding Cognitive Planning in Robotics

Cognitive planning in robotics involves creating high-level plans from natural language commands and translating them into executable robotic actions. This process requires understanding both the semantics of human language and the capabilities of robotic systems.

### ROS 2 for Robotic Applications

ROS 2 (Robot Operating System 2) is an open-source set of software libraries and tools for building robot applications. It provides drivers, algorithms, and developer tools that enable communication between different robotic components.

Key components of ROS 2 relevant to cognitive planning:
- **Topics**: Asynchronous message passing between nodes
- **Services**: Synchronous request/response communication
- **Actions**: Goal-oriented communication with feedback and status updates
- **Parameters**: Configuration management for nodes

### Integration with LLMs

The integration of LLMs with ROS 2 systems enables natural language interaction with robots. Projects like ROSA (ROS Agent) by NASA JPL demonstrate how LLMs can be used to interact with ROS systems through natural language queries.

## Implementation Guide

### Setting up ROS 2 Environment

First, ensure you have ROS 2 installed (Humble Hawksbill or later recommended):

```bash
# Install ROS 2 dependencies
sudo apt update
sudo apt install ros-humble-desktop
```

### Basic LLM Integration with ROS 2

Here's a basic example of how to integrate an LLM with ROS 2 for cognitive planning:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI  # Example with OpenAI, but can use other LLMs

class CognitivePlannerNode(Node):
    def __init__(self):
        super().__init__('cognitive_planner')

        # Initialize LLM client
        self.llm_client = OpenAI()  # Configure with your API key

        # Create publisher for robot commands
        self.command_publisher = self.create_publisher(String, 'robot_commands', 10)

        # Create subscriber for natural language commands
        self.command_subscriber = self.create_subscription(
            String,
            'natural_language_commands',
            self.command_callback,
            10
        )

        self.get_logger().info('Cognitive Planner Node initialized')

    def command_callback(self, msg):
        """Process natural language command and translate to ROS 2 action."""
        natural_language_command = msg.data
        self.get_logger().info(f'Received command: {natural_language_command}')

        # Use LLM to translate natural language to robot action
        robot_action = self.translate_to_action(natural_language_command)

        if robot_action:
            # Publish the robot action
            command_msg = String()
            command_msg.data = robot_action
            self.command_publisher.publish(command_msg)
            self.get_logger().info(f'Published robot action: {robot_action}')
        else:
            self.get_logger().warn('Could not translate command to robot action')

    def translate_to_action(self, natural_language):
        """Use LLM to translate natural language to robot action."""
        prompt = f"""
        You are a cognitive planning system for a robot.
        Convert the following natural language command into a robot action.
        Only respond with the action in the format "ACTION_TYPE:PARAMETER".

        Available actions:
        - MOVE_FORWARD:distance_in_meters
        - MOVE_BACKWARD:distance_in_meters
        - TURN_LEFT:angle_in_degrees
        - TURN_RIGHT:angle_in_degrees
        - GRIPPER_OPEN:
        - GRIPPER_CLOSE:
        - ARM_MOVE_TO:position_name

        Natural language command: "{natural_language}"

        Robot action:
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Or any other model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )

            action = response.choices[0].message.content.strip()
            return action
        except Exception as e:
            self.get_logger().error(f'Error calling LLM: {e}')
            return None

def main(args=None):
    rclpy.init(args=args)
    cognitive_planner = CognitivePlannerNode()

    try:
        rclpy.spin(cognitive_planner)
    except KeyboardInterrupt:
        pass
    finally:
        cognitive_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Code Examples

### Pseudocode for Cognitive Planning Pipeline

```
1. Receive natural language command
2. Preprocess command (clean, normalize)
3. Use LLM to parse command and generate plan
4. Validate plan against robot capabilities
5. Execute plan through ROS 2 actions
6. Monitor execution and provide feedback
```

### ROS 2 Action Client Example

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import json

class CognitivePlanningExecutor(Node):
    def __init__(self):
        super().__init__('cognitive_planning_executor')

        # Create action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create subscription for high-level commands
        self.command_sub = self.create_subscription(
            String,
            'high_level_commands',
            self.execute_command,
            10
        )

    def execute_command(self, msg):
        """Execute high-level command by converting to ROS 2 actions."""
        command_data = json.loads(msg.data)

        if command_data['action'] == 'navigate_to':
            self.navigate_to_location(command_data['location'])
        elif command_data['action'] == 'pick_object':
            self.pick_object(command_data['object'])
        # Add more action handlers as needed

    def navigate_to_location(self, location_name):
        """Navigate to a predefined location."""
        # Wait for action server
        self.nav_client.wait_for_server()

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose = self.get_location_pose(location_name)

        # Send goal
        self.nav_client.send_goal_async(goal_msg)
```

## Practical Examples

### Example 1: Natural Language Navigation

A cognitive planning system that allows users to give navigation commands in natural language like "Go to the kitchen" or "Navigate to the charging station."

### Example 2: Task Planning with LLM Integration

A system that takes high-level commands like "Clean the living room" and breaks them down into a sequence of robotic actions.

## Exercises

1. Implement a cognitive planning system that can handle at least 5 different types of robot commands.
2. Extend the system to handle multi-step commands like "Go to the kitchen and bring me a cup."
3. Add error handling and recovery mechanisms to the cognitive planning pipeline.
4. Implement a feedback system that allows the robot to report task completion to the user.

## Summary

This chapter covered the fundamentals of cognitive planning in robotics, focusing on how to use LLMs to translate natural language into ROS 2 actions. We explored the technical background, implementation patterns, and practical examples of cognitive planning systems. The next chapter will integrate these concepts in a capstone project.
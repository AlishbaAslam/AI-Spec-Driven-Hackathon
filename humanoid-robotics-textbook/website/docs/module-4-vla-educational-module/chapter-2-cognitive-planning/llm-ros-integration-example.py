"""
Example implementation of LLM integration with ROS 2 for cognitive planning.
This demonstrates how to use Large Language Models to translate natural language
into ROS 2 actions for robotic systems.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import JointState

import json
import openai
from typing import Dict, Any, List

class LLMROS2CognitivePlanner(Node):
    """
    A ROS 2 node that uses an LLM to translate natural language commands
    into ROS 2 actions for robotic systems.
    """

    def __init__(self):
        super().__init__('llm_ros2_cognitive_planner')

        # Initialize OpenAI client (or any other LLM client)
        # Note: You'll need to set your API key
        # openai.api_key = "your-api-key-here"
        self.llm_client = openai.OpenAI()

        # Create publisher for robot commands
        self.command_publisher = self.create_publisher(
            String,
            'robot_commands',
            QoSProfile(depth=10)
        )

        # Create subscriber for natural language commands
        self.command_subscriber = self.create_subscription(
            String,
            'natural_language_commands',
            self.natural_language_callback,
            QoSProfile(depth=10)
        )

        # Create action client for navigation
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Create subscriber for robot state
        self.robot_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.robot_state_callback,
            QoSProfile(depth=10)
        )

        # Store robot state
        self.robot_state = {}

        self.get_logger().info('LLM-ROS2 Cognitive Planner initialized')

    def natural_language_callback(self, msg: String):
        """
        Process natural language command and translate to ROS 2 actions.

        Args:
            msg (String): ROS message containing natural language command
        """
        natural_language_command = msg.data
        self.get_logger().info(f'Received natural language command: {natural_language_command}')

        # Generate action plan using LLM
        action_plan = self.generate_action_plan(natural_language_command)

        if action_plan:
            self.get_logger().info(f'Generated action plan: {json.dumps(action_plan, indent=2)}')

            # Execute the action plan
            self.execute_action_plan(action_plan)
        else:
            self.get_logger().warn('Could not generate action plan from command')

    def generate_action_plan(self, natural_language: str) -> Dict[str, Any]:
        """
        Use LLM to generate an action plan from natural language command.

        Args:
            natural_language (str): Natural language command

        Returns:
            Dict[str, Any]: Action plan as a dictionary
        """
        # Define available ROS 2 actions and services
        system_prompt = """
        You are a cognitive planning system for a ROS 2 robot.
        Convert natural language commands into structured action plans.
        Each action should have a type and parameters compatible with ROS 2.

        Available action types:
        - NAVIGATE_TO_POSE: Navigate to a specific location with x, y, theta coordinates
        - MOVE_ARM_JOINTS: Move robot arm to specific joint positions
        - GRIPPER_CONTROL: Control the gripper (open/close)
        - SPEAK_TEXT: Make the robot speak text
        - WAIT: Wait for a specified duration

        Format your response as a JSON object with an 'actions' array:
        {
          "actions": [
            {
              "type": "NAVIGATE_TO_POSE",
              "params": {"x": 1.0, "y": 2.0, "theta": 0.0}
            }
          ]
        }

        Be precise with coordinates and parameters.
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can use other models too
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Command: {natural_language}"}
                ],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # Parse the response
            action_plan_str = response.choices[0].message.content.strip()
            action_plan = json.loads(action_plan_str)

            return action_plan

        except Exception as e:
            self.get_logger().error(f'Error generating action plan: {e}')
            return {}

    def execute_action_plan(self, action_plan: Dict[str, Any]):
        """
        Execute the action plan by publishing to appropriate ROS 2 topics/services/actions.

        Args:
            action_plan (Dict[str, Any]): The action plan to execute
        """
        if 'actions' not in action_plan:
            self.get_logger().warn('No actions found in action plan')
            return

        for action in action_plan['actions']:
            action_type = action.get('type')
            params = action.get('params', {})

            self.get_logger().info(f'Executing action: {action_type} with params: {params}')

            if action_type == 'NAVIGATE_TO_POSE':
                self.execute_navigation_action(params)
            elif action_type == 'MOVE_ARM_JOINTS':
                self.execute_arm_action(params)
            elif action_type == 'GRIPPER_CONTROL':
                self.execute_gripper_action(params)
            elif action_type == 'SPEAK_TEXT':
                self.execute_speak_action(params)
            elif action_type == 'WAIT':
                self.execute_wait_action(params)
            else:
                self.get_logger().warn(f'Unknown action type: {action_type}')

    def execute_navigation_action(self, params: Dict[str, Any]):
        """
        Execute navigation action using Nav2.

        Args:
            params (Dict[str, Any]): Navigation parameters (x, y, theta)
        """
        # Wait for navigation action server
        self.nav_client.wait_for_server()

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set pose from parameters
        goal_msg.pose.pose.position.x = params.get('x', 0.0)
        goal_msg.pose.pose.position.y = params.get('y', 0.0)
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta (yaw) to quaternion
        theta = params.get('theta', 0.0)
        from math import sin, cos
        sin_half_theta = sin(theta / 2.0)
        cos_half_theta = cos(theta / 2.0)
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = sin_half_theta
        goal_msg.pose.pose.orientation.w = cos_half_theta

        # Send goal
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.navigation_goal_response_callback)

    def navigation_goal_response_callback(self, future):
        """
        Callback for navigation goal response.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            return

        self.get_logger().info('Navigation goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        """
        Callback for navigation result.
        """
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        self.publish_command_result('navigation_completed')

    def execute_arm_action(self, params: Dict[str, Any]):
        """
        Execute arm movement action.

        Args:
            params (Dict[str, Any]): Arm movement parameters
        """
        # This would publish to joint trajectory controller
        # For demonstration, we'll just log the action
        self.get_logger().info(f'Moving arm to joint positions: {params}')
        self.publish_command_result('arm_moved')

    def execute_gripper_action(self, params: Dict[str, Any]):
        """
        Execute gripper control action.

        Args:
            params (Dict[str, Any]): Gripper control parameters
        """
        # This would publish to gripper controller
        action = params.get('action', 'open')
        self.get_logger().info(f'Controlling gripper: {action}')
        self.publish_command_result(f'gripper_{action}')

    def execute_speak_action(self, params: Dict[str, Any]):
        """
        Execute speech action.

        Args:
            params (Dict[str, Any]): Speech parameters
        """
        text = params.get('text', '')
        self.get_logger().info(f'Speaking: {text}')
        self.publish_command_result('spoken')

    def execute_wait_action(self, params: Dict[str, Any]):
        """
        Execute wait action.

        Args:
            params (Dict[str, Any]): Wait parameters
        """
        duration = params.get('duration', 1.0)
        self.get_logger().info(f'Waiting for {duration} seconds')
        # In a real implementation, this would use a timer
        self.publish_command_result('wait_completed')

    def robot_state_callback(self, msg: JointState):
        """
        Update robot state from joint state messages.

        Args:
            msg (JointState): Joint state message
        """
        self.robot_state = {
            'position': dict(zip(msg.name, msg.position)),
            'velocity': dict(zip(msg.name, msg.velocity)),
            'effort': dict(zip(msg.name, msg.effort))
        }

    def publish_command_result(self, result: str):
        """
        Publish command result to robot_commands topic.

        Args:
            result (str): Result of the command execution
        """
        result_msg = String()
        result_msg.data = result
        self.command_publisher.publish(result_msg)


def main(args=None):
    """
    Main function to run the LLM-ROS2 Cognitive Planner node.
    """
    rclpy.init(args=args)

    cognitive_planner = LLMROS2CognitivePlanner()

    try:
        rclpy.spin(cognitive_planner)
    except KeyboardInterrupt:
        cognitive_planner.get_logger().info('Node interrupted by user')
    finally:
        cognitive_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
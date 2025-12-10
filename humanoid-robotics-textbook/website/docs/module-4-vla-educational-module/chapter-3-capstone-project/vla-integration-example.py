"""
Complete Vision-Language-Action (VLA) system for an autonomous humanoid robot.
This integrates voice processing (Whisper), cognitive planning (LLMs),
and robot control (ROS 2) into a unified system.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile

from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from builtin_interfaces.msg import Duration

import whisper
import openai
import json
import numpy as np
from typing import Dict, Any, Optional
import threading
import time

class VLASystemNode(Node):
    """
    Complete Vision-Language-Action system node that integrates:
    - Vision: Processing visual input
    - Language: Processing natural language with LLMs
    - Action: Executing robotic actions through ROS 2
    """

    def __init__(self):
        super().__init__('vla_system')

        # Initialize Whisper model for voice processing
        self.get_logger().info('Loading Whisper model...')
        self.whisper_model = whisper.load_model("base")
        self.get_logger().info('Whisper model loaded')

        # Initialize OpenAI client for cognitive planning
        # Note: You'll need to set your API key
        # openai.api_key = "your-api-key-here"
        self.llm_client = openai.OpenAI()

        # Initialize internal state
        self.robot_state = {}
        self.vision_data = {}
        self.current_task = None

        # Create publishers
        self.action_publisher = self.create_publisher(
            String,
            'robot_actions',
            QoSProfile(depth=10)
        )

        # Create subscribers
        self.voice_subscriber = self.create_subscription(
            String,
            'voice_commands',
            self.voice_command_callback,
            QoSProfile(depth=10)
        )

        self.vision_subscriber = self.create_subscription(
            Image,
            'camera_image',
            self.vision_callback,
            QoSProfile(depth=10)
        )

        self.robot_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.robot_state_callback,
            QoSProfile(depth=10)
        )

        # Create action clients
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Task execution thread
        self.task_execution_thread = None
        self.task_queue = []
        self.task_queue_lock = threading.Lock()

        # Timer for processing task queue
        self.task_timer = self.create_timer(0.1, self.process_task_queue)

        self.get_logger().info('VLA System initialized successfully')

    def voice_command_callback(self, msg: String):
        """
        Process voice command through the complete VLA pipeline.

        Args:
            msg (String): ROS message containing voice command
        """
        command_text = msg.data
        self.get_logger().info(f'Received voice command: {command_text}')

        # Process through VLA pipeline
        action_plan = self.vla_pipeline(command_text)

        if action_plan:
            self.get_logger().info(f'Generated action plan: {json.dumps(action_plan, indent=2)}')

            # Add to task queue for execution
            with self.task_queue_lock:
                self.task_queue.append(action_plan)

    def vision_callback(self, msg: Image):
        """
        Process visual input from robot's camera.

        Args:
            msg (Image): ROS image message
        """
        # In a real implementation, this would process the image
        # For this example, we'll just store some metadata
        self.vision_data = {
            'height': msg.height,
            'width': msg.width,
            'encoding': msg.encoding,
            'timestamp': self.get_clock().now().nanoseconds
        }

        self.get_logger().info(f'Processed visual input: {msg.height}x{msg.width}')

    def robot_state_callback(self, msg: JointState):
        """
        Update robot state from joint state messages.

        Args:
            msg (JointState): Joint state message
        """
        self.robot_state = {
            'position': dict(zip(msg.name, msg.position)),
            'velocity': dict(zip(msg.name, msg.velocity)),
            'effort': dict(zip(msg.name, msg.effort)),
            'timestamp': self.get_clock().now().nanoseconds
        }

    def vla_pipeline(self, natural_language_command: str) -> Optional[Dict[str, Any]]:
        """
        Complete VLA pipeline: Vision + Language + Action.

        Args:
            natural_language_command (str): Natural language command

        Returns:
            Optional[Dict[str, Any]]: Action plan or None if failed
        """
        try:
            # Step 1: Process language with LLM
            action_plan = self.cognitive_planning(natural_language_command)

            # Step 2: Integrate vision data if available
            if self.vision_data:
                action_plan = self.integrate_vision_data(action_plan)

            # Step 3: Validate action plan
            if self.validate_action_plan(action_plan):
                return action_plan
            else:
                self.get_logger().warn('Action plan validation failed')
                return None

        except Exception as e:
            self.get_logger().error(f'Error in VLA pipeline: {e}')
            return None

    def cognitive_planning(self, natural_language: str) -> Dict[str, Any]:
        """
        Use LLM to generate action plan from natural language.

        Args:
            natural_language (str): Natural language command

        Returns:
            Dict[str, Any]: Action plan
        """
        # System prompt for the VLA system
        system_prompt = """
        You are a cognitive planning system for a Vision-Language-Action (VLA) robot.
        Generate detailed action plans from natural language commands.
        Consider both the language command and any available visual information.

        Available action types:
        - NAVIGATE_TO_LOCATION: Navigate to a specific location
        - DETECT_OBJECT: Detect and locate objects in the environment
        - PICK_OBJECT: Pick up an object
        - PLACE_OBJECT: Place an object at a location
        - MOVE_ARM: Move the robot arm to a position
        - SPEAK: Make the robot speak
        - WAIT: Wait for a duration

        Parameters for each action:
        - NAVIGATE_TO_LOCATION: {"location": "kitchen", "x": float, "y": float}
        - DETECT_OBJECT: {"object_type": "cup", "color": "red"}
        - PICK_OBJECT: {"object_id": "obj123"}
        - PLACE_OBJECT: {"location": "table", "x": float, "y": float}
        - MOVE_ARM: {"joint_positions": [float, ...]}
        - SPEAK: {"text": "message"}
        - WAIT: {"duration": float}

        Format your response as a JSON object with an 'actions' array:
        {
          "actions": [
            {
              "type": "NAVIGATE_TO_LOCATION",
              "params": {"location": "kitchen", "x": 1.0, "y": 2.0}
            }
          ],
          "context": {
            "visual_info": "description of visual scene",
            "robot_state": "current robot state"
          }
        }
        """

        # Include visual context if available
        visual_context = self.vision_data.get('encoding', 'no visual data available')

        user_prompt = f"""
        Natural language command: "{natural_language}"
        Current visual context: "{visual_context}"
        Current robot state: "{json.dumps(self.robot_state.get('position', {}))}"

        Generate action plan:
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",  # Using a more capable model for complex VLA planning
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            action_plan_str = response.choices[0].message.content.strip()
            action_plan = json.loads(action_plan_str)

            return action_plan

        except Exception as e:
            self.get_logger().error(f'Error in cognitive planning: {e}')
            return {"actions": []}

    def integrate_vision_data(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate visual data into the action plan.

        Args:
            action_plan (Dict[str, Any]): Original action plan

        Returns:
            Dict[str, Any]: Updated action plan with visual integration
        """
        # Add visual context to the action plan
        if 'context' not in action_plan:
            action_plan['context'] = {}

        action_plan['context']['visual_data'] = self.vision_data
        action_plan['context']['timestamp'] = self.get_clock().now().nanoseconds

        # Potentially modify actions based on visual data
        # This is a simplified example - in practice, this would be more complex
        for action in action_plan.get('actions', []):
            if action['type'] == 'DETECT_OBJECT':
                # Use visual data to refine object detection parameters
                visual_objects = self.extract_visual_objects()
                action['params']['detected_objects'] = visual_objects

        return action_plan

    def extract_visual_objects(self) -> list:
        """
        Extract objects from visual data (simplified for this example).

        Returns:
            list: List of detected objects
        """
        # In a real implementation, this would run object detection
        # For this example, return some mock data
        return [
            {"id": "cup_001", "type": "cup", "color": "red", "position": [1.2, 0.8, 0.5]},
            {"id": "bottle_002", "type": "bottle", "color": "blue", "position": [1.5, 0.9, 0.6]}
        ]

    def validate_action_plan(self, action_plan: Dict[str, Any]) -> bool:
        """
        Validate the action plan for safety and feasibility.

        Args:
            action_plan (Dict[str, Any]): Action plan to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(action_plan, dict):
            return False

        if 'actions' not in action_plan:
            return False

        actions = action_plan['actions']
        if not isinstance(actions, list):
            return False

        # Check if all action types are supported
        supported_actions = [
            'NAVIGATE_TO_LOCATION',
            'DETECT_OBJECT',
            'PICK_OBJECT',
            'PLACE_OBJECT',
            'MOVE_ARM',
            'SPEAK',
            'WAIT'
        ]

        for action in actions:
            if not isinstance(action, dict):
                return False

            if 'type' not in action:
                return False

            if action['type'] not in supported_actions:
                self.get_logger().warn(f'Unsupported action type: {action["type"]}')
                return False

        return True

    def process_task_queue(self):
        """
        Process tasks in the execution queue.
        """
        if not self.task_queue:
            return

        with self.task_queue_lock:
            if not self.task_queue:
                return

            # Get the next task
            task = self.task_queue.pop(0)

        # Execute the task in a separate thread to avoid blocking
        self.task_execution_thread = threading.Thread(
            target=self.execute_action_plan,
            args=(task,)
        )
        self.task_execution_thread.start()

    def execute_action_plan(self, action_plan: Dict[str, Any]):
        """
        Execute the action plan sequentially.

        Args:
            action_plan (Dict[str, Any]): Action plan to execute
        """
        actions = action_plan.get('actions', [])

        for i, action in enumerate(actions):
            self.get_logger().info(f'Executing action {i+1}/{len(actions)}: {action["type"]}')

            success = self.execute_single_action(action)
            if not success:
                self.get_logger().error(f'Action failed: {action}')
                break

            # Small delay between actions
            time.sleep(0.1)

        self.get_logger().info('Action plan execution completed')

    def execute_single_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a single action.

        Args:
            action (Dict[str, Any]): Action to execute

        Returns:
            bool: True if successful, False otherwise
        """
        action_type = action.get('type')
        params = action.get('params', {})

        if action_type == 'NAVIGATE_TO_LOCATION':
            return self.execute_navigation_action(params)
        elif action_type == 'DETECT_OBJECT':
            return self.execute_detection_action(params)
        elif action_type == 'PICK_OBJECT':
            return self.execute_pick_action(params)
        elif action_type == 'PLACE_OBJECT':
            return self.execute_place_action(params)
        elif action_type == 'MOVE_ARM':
            return self.execute_arm_action(params)
        elif action_type == 'SPEAK':
            return self.execute_speak_action(params)
        elif action_type == 'WAIT':
            return self.execute_wait_action(params)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_navigation_action(self, params: Dict[str, Any]) -> bool:
        """
        Execute navigation action.

        Args:
            params (Dict[str, Any]): Navigation parameters

        Returns:
            bool: True if successful
        """
        try:
            # Wait for navigation server
            self.nav_client.wait_for_server()

            # Create goal message
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

            # Set position
            goal_msg.pose.pose.position.x = params.get('x', 0.0)
            goal_msg.pose.pose.position.y = params.get('y', 0.0)
            goal_msg.pose.pose.position.z = 0.0

            # Set orientation (simplified)
            goal_msg.pose.pose.orientation.w = 1.0

            # Send goal
            future = self.nav_client.send_goal_async(goal_msg)
            # Note: In a real implementation, you'd wait for the result
            # For this example, we'll just return True

            self.get_logger().info(f'Navigation goal sent to ({params.get("x", 0.0)}, {params.get("y", 0.0)})')
            return True

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False

    def execute_detection_action(self, params: Dict[str, Any]) -> bool:
        """
        Execute object detection action.

        Args:
            params (Dict[str, Any]): Detection parameters

        Returns:
            bool: True if successful
        """
        obj_type = params.get('object_type', 'unknown')
        color = params.get('color', 'any')

        self.get_logger().info(f'Detecting {color} {obj_type}')
        # In a real implementation, this would run object detection
        # For this example, we'll use the mock data from extract_visual_objects()
        detected_objects = self.extract_visual_objects()

        # Filter by type and color if specified
        if obj_type != 'unknown':
            detected_objects = [obj for obj in detected_objects if obj['type'] == obj_type]
        if color != 'any':
            detected_objects = [obj for obj in detected_objects if obj['color'] == color]

        self.get_logger().info(f'Detected {len(detected_objects)} objects: {detected_objects}')
        return True

    def execute_pick_action(self, params: Dict[str, Any]) -> bool:
        """
        Execute pick object action.

        Args:
            params (Dict[str, Any]): Pick parameters

        Returns:
            bool: True if successful
        """
        obj_id = params.get('object_id', 'unknown')
        self.get_logger().info(f'Picking object: {obj_id}')
        # In a real implementation, this would control the robot's gripper/arm
        return True

    def execute_place_action(self, params: Dict[str, Any]) -> bool:
        """
        Execute place object action.

        Args:
            params (Dict[str, Any]): Place parameters

        Returns:
            bool: True if successful
        """
        location = params.get('location', 'unknown')
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        self.get_logger().info(f'Placing object at {location} ({x}, {y})')
        # In a real implementation, this would control the robot's gripper/arm
        return True

    def execute_arm_action(self, params: Dict[str, Any]) -> bool:
        """
        Execute arm movement action.

        Args:
            params (Dict[str, Any]): Arm movement parameters

        Returns:
            bool: True if successful
        """
        joint_positions = params.get('joint_positions', [])
        self.get_logger().info(f'Moving arm to positions: {joint_positions}')
        # In a real implementation, this would publish to joint trajectory controller
        return True

    def execute_speak_action(self, params: Dict[str, Any]) -> bool:
        """
        Execute speak action.

        Args:
            params (Dict[str, Any]): Speak parameters

        Returns:
            bool: True if successful
        """
        text = params.get('text', '')
        self.get_logger().info(f'Speaking: {text}')
        # In a real implementation, this would use text-to-speech
        return True

    def execute_wait_action(self, params: Dict[str, Any]) -> bool:
        """
        Execute wait action.

        Args:
            params (Dict[str, Any]): Wait parameters

        Returns:
            bool: True if successful
        """
        duration = params.get('duration', 1.0)
        self.get_logger().info(f'Waiting for {duration} seconds')
        # In a real implementation, this would use a timer
        # For this example, we'll just sleep in the thread
        time.sleep(duration)
        return True


def main(args=None):
    """
    Main function to run the VLA system node.
    """
    rclpy.init(args=args)

    vla_system = VLASystemNode()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        vla_system.get_logger().info('VLA System interrupted by user')
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
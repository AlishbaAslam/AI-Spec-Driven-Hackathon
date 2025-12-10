# Data Model: The Robotic Nervous System (ROS 2)

## ROS 2 Node Entity
**Description**: A process that performs computation in the ROS 2 system, implementing communication with other nodes
**Fields**:
- node_name: string (unique identifier for the node)
- node_namespace: string (optional namespace for the node)
- publishers: list of Publisher objects (list of topics this node publishes to)
- subscribers: list of Subscriber objects (list of topics this node subscribes to)
- services: list of Service objects (list of services this node provides)
- clients: list of Client objects (list of services this node calls)

## ROS 2 Publisher Entity
**Description**: An object that sends messages to a specific topic
**Fields**:
- topic_name: string (name of the topic to publish to)
- message_type: string (type of message being published)
- qos_profile: QoSProfile object (quality of service settings)

## ROS 2 Subscriber Entity
**Description**: An object that receives messages from a specific topic
**Fields**:
- topic_name: string (name of the topic to subscribe to)
- message_type: string (type of message being received)
- callback_function: function (function to handle incoming messages)
- qos_profile: QoSProfile object (quality of service settings)

## ROS 2 Service Entity
**Description**: An object that provides a synchronous request/response communication pattern
**Fields**:
- service_name: string (name of the service)
- service_type: string (type of service request/response)
- callback_function: function (function to handle service requests)

## ROS 2 Client Entity
**Description**: An object that makes requests to a ROS 2 service
**Fields**:
- service_name: string (name of the service to call)
- service_type: string (type of service request/response)

## URDF Robot Model Entity
**Description**: An XML-based description of a robot including kinematics and dynamics
**Fields**:
- robot_name: string (name of the robot)
- links: list of Link objects (physical components of the robot)
- joints: list of Joint objects (connections between links)
- materials: list of Material objects (visual properties)
- gazebo_plugins: list of GazeboPlugin objects (simulation-specific extensions)

## URDF Link Entity
**Description**: A physical component of a robot (e.g., body, arm, wheel)
**Fields**:
- link_name: string (name of the link)
- visual: Visual object (visual representation)
- collision: Collision object (collision properties)
- inertial: Inertial object (mass and inertia properties)

## URDF Joint Entity
**Description**: Connection between two links defining their relative motion
**Fields**:
- joint_name: string (name of the joint)
- joint_type: string (type: revolute, continuous, prismatic, fixed, etc.)
- parent_link: string (name of parent link)
- child_link: string (name of child link)
- origin: Pose object (position and orientation of joint)

## Python Agent Entity
**Description**: A software component written in Python that performs intelligent behaviors and communicates via ROS 2
**Fields**:
- agent_name: string (name of the agent)
- ros_nodes: list of ROS 2 Node objects (ROS 2 nodes the agent controls)
- behavior_functions: list of function objects (functions implementing agent behaviors)
- sensor_subscriptions: list of Subscriber objects (sensors the agent monitors)
- actuator_publishers: list of Publisher objects (actuators the agent controls)

## QoSProfile Entity
**Description**: Quality of Service settings that define message delivery guarantees
**Fields**:
- reliability: string (reliable or best-effort)
- durability: string (volatile or transient-local)
- history: string (keep-all or keep-last)
- depth: integer (number of messages to keep in history)
# Quickstart: The Robotic Nervous System (ROS 2)

## Prerequisites

Before starting with the ROS 2 module, ensure you have:

1. **ROS 2 Humble Hawksbill** installed on your system (Linux recommended)
2. **Python 3.8+**
3. **Basic Python programming knowledge**
4. **Docker** (optional, for containerized examples)

## Environment Setup

### 1. Install ROS 2 Humble Hawksbill

Follow the official installation guide: https://docs.ros.org/en/humble/Installation.html

For Ubuntu:
```bash
# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions
```

### 2. Source ROS 2 Environment

```bash
source /opt/ros/humble/setup.bash
```

To make this permanent, add to your `.bashrc`:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 3. Create a ROS 2 Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Running the Examples

### Chapter 1: Basic Publisher/Subscriber

1. Navigate to the examples directory
2. Create a basic publisher:

```python
# basic_publisher.py
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

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. In another terminal, run the publisher:
```bash
python3 basic_publisher.py
```

4. In a third terminal, run the subscriber:
```bash
python3 basic_subscriber.py
```

### Chapter 2: Python Agent Integration

1. Run the Python agent example:
```bash
python3 python_agent.py
```

This agent will subscribe to sensor data and publish control commands.

### Chapter 3: URDF Modeling

1. Visualize the simple robot URDF:
```bash
ros2 run rviz2 rviz2
```

2. Load the URDF file and visualize the robot model.

## Testing Your Setup

To verify your ROS 2 installation is working:

```bash
# Check ROS 2 installation
ros2 --version

# List available topics
ros2 topic list

# Check if basic ROS 2 commands work
ros2 run demo_nodes_cpp talker
```

## Troubleshooting

- If ROS 2 commands are not found, ensure you've sourced the ROS 2 environment
- If Python packages are missing, install them with: `pip3 install [package-name]`
- For network issues with ROS 2, check ROS_DOMAIN_ID environment variable
- If examples don't run, ensure all dependencies are installed with: `rosdep install --from-paths src --ignore-src -r -y`
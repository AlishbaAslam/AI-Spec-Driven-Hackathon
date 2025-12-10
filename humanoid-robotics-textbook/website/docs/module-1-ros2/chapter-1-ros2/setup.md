---
title: ROS 2 Installation and Environment Setup
sidebar_position: 2
---

# ROS 2 Installation and Environment Setup

This guide will help you set up your ROS 2 environment for the educational module. Follow these steps to ensure your system is properly configured for ROS 2 development.

## Prerequisites

Before starting with the ROS 2 module, ensure you have:

1. **Ubuntu 22.04** (recommended) or compatible Linux distribution
2. **Python 3.8+** (Python 3.10 recommended for ROS 2 Humble)
3. **Basic Python programming knowledge**
4. **Git** for version control
5. **Docker** (optional, for containerized examples)

## Environment Setup

### 1. Install ROS 2 Humble Hawksbill

We'll be using ROS 2 Humble Hawksbill, which is the latest LTS (Long Term Support) version at the time of writing. Follow the official installation guide: https://docs.ros.org/en/humble/Installation.html

For Ubuntu 22.04:

```bash
# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions
sudo apt install -y python3-rosdep python3-vcstool # Recommended packages
```

### 2. Source ROS 2 Environment

After installation, you need to source the ROS 2 environment to use ROS 2 commands:

```bash
source /opt/ros/humble/setup.bash
```

To make this permanent, add to your `.bashrc`:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

You can verify the installation by checking the ROS 2 version:

```bash
ros2 --version
```

### 3. Create a ROS 2 Workspace

A workspace is where you'll develop your ROS 2 packages:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

The `src` directory is where you'll place your source code packages. The `colcon build` command compiles your packages, and `source install/setup.bash` makes them available in your current terminal session.

## Python Dependencies

For Python development with ROS 2, you may need additional packages:

```bash
pip3 install rclpy  # This should already be installed with ROS 2
pip3 install numpy
pip3 install matplotlib  # For visualization
pip3 install opencv-python  # For computer vision (optional)
```

## Testing Your Setup

To verify your ROS 2 installation is working:

```bash
# Check ROS 2 installation
ros2 --version

# List available topics (should work even without any nodes running)
ros2 topic list

# Check if basic ROS 2 commands work
# Note: This will run indefinitely, use Ctrl+C to stop
# ros2 run demo_nodes_cpp talker
```

## Troubleshooting

### Common Issues

1. **ROS 2 commands not found**: Make sure you've sourced the ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Python packages missing**: Install with pip3:
   ```bash
   pip3 install [package-name]
   ```

3. **Network issues with ROS 2**: Check ROS_DOMAIN_ID environment variable:
   ```bash
   echo $ROS_DOMAIN_ID
   # Default is 0, but you can set it to avoid conflicts:
   # export ROS_DOMAIN_ID=2
   ```

4. **If examples don't run**: Ensure all dependencies are installed with:
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

### ROS 2 Environment Variables

Some useful environment variables you might want to set:

```bash
# Set ROS domain ID to avoid conflicts on shared networks
export ROS_DOMAIN_ID=2

# Set log level
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=INFO

# Enable ROS 2 introspection tools
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## Next Steps

Once your environment is set up, you can proceed to the next sections of this module:

1. Chapter 1: Fundamentals of ROS 2 Nodes, Topics, and Services
2. Chapter 2: Integrating Python Agents with ROS 2 via rclpy
3. Chapter 3: Modeling Humanoid Robots with URDF

Each chapter includes hands-on exercises that will help you practice the concepts learned.
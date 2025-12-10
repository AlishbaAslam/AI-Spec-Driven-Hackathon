---
title: Gazebo Setup Guide for Physics Simulation
sidebar_position: 3
---

# Gazebo Setup Guide for Physics Simulation

This guide provides comprehensive instructions for installing and setting up Gazebo specifically for physics simulation in digital twin applications. We'll cover the installation process, configuration, and verification steps needed to create realistic physics simulations.

## Prerequisites

Before installing Gazebo, ensure your system meets the following requirements:

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS, Windows 10/11 (with WSL2), or macOS 10.14+
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB available space
- **Graphics**: OpenGL 2.1+ capable GPU with dedicated VRAM
- **Internet**: Required for package downloads

### Software Prerequisites
- Basic command-line knowledge
- Git installed on your system
- ROS 2 Humble Hawksbill (for ROS integration)

## Ubuntu Installation (Recommended)

### Step 1: Add Gazebo Package Repository

First, add the Gazebo package repository to your system:

```bash
# Add the OSRF keys
sudo apt update && sudo apt install wget
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Update package list
sudo apt update
```

### Step 2: Install Gazebo Garden

Install the latest stable version of Gazebo (Garden):

```bash
sudo apt install gz-garden
```

### Step 3: Install ROS 2 Integration Packages

For ROS 2 Humble integration:

```bash
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-ros2-controllers
```

### Step 4: Set Up Environment Variables

Add Gazebo and ROS 2 to your bash environment:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source /usr/share/gz/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Install Additional Dependencies

Install additional packages for enhanced functionality:

```bash
sudo apt install ros-humble-xacro
sudo apt install ros-humble-joint-state-publisher
sudo apt install ros-humble-robot-state-publisher
sudo apt install ros-humble-teleop-twist-keyboard
```

## Windows Installation (Using WSL2)

### Step 1: Install WSL2 and Ubuntu 22.04

1. Open PowerShell as Administrator and run:
   ```powershell
   wsl --install Ubuntu-22.04
   ```

2. Complete the Ubuntu setup by following the prompts

### Step 2: Install X Server for GUI Applications

1. Download and install VcXsrv or X410
2. Configure X Server to allow connections from network interfaces
3. Set the DISPLAY environment variable in Ubuntu:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   echo "export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0" >> ~/.bashrc
   ```

### Step 3: Follow Ubuntu Installation Steps

Follow the Ubuntu installation steps (1-5) within your WSL2 Ubuntu terminal.

## macOS Installation

### Step 1: Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Gazebo Garden

```bash
brew install osrf/simulation/gz-garden
```

### Step 3: Install ROS 2 via Docker

For macOS, we recommend using Docker for ROS 2:

```bash
# Install Docker Desktop for Mac
# Then pull the ROS 2 Humble image
docker pull osrf/ros:humble-desktop-full

# Create a script to run ROS 2 commands
echo '#!/bin/bash
docker run -it --rm \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --network=host \
  osrf/ros:humble-desktop-full
' > ~/run_ros.sh

chmod +x ~/run_ros.sh
```

## Verification and Testing

### Step 1: Verify Installation

Test that Gazebo is properly installed:

```bash
gz sim --version
```

You should see the Gazebo version information.

### Step 2: Test Basic Simulation

Launch a simple simulation to verify everything works:

```bash
gz sim -v 4 shapes.sdf
```

This should open the Gazebo GUI with a simple scene containing various shapes.

### Step 3: Test ROS Integration

Create a simple test to verify ROS integration:

1. Open a new terminal and source ROS:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Check available topics:
   ```bash
   ros2 topic list
   ```

3. If running a Gazebo simulation, you should see topics like `/clock`, `/tf`, etc.

## Configuration for Digital Twin Applications

### Performance Configuration

For optimal performance in digital twin applications, configure these settings:

1. **Time Step Configuration**:
   Create a custom physics configuration in your world file:
   ```xml
   <physics type="ode">
     <max_step_size>0.001</max_step_size>
     <real_time_factor>1.0</real_time_factor>
     <real_time_update_rate>1000</real_time_update_rate>
   </physics>
   ```

2. **Graphics Settings**:
   - For development: Use full graphics for visualization
   - For deployment: Consider using OGRE's reduced quality settings
   - Adjust shadow quality based on performance requirements

### Recommended Settings for Digital Twins

1. **Accuracy vs. Performance**:
   - Development: Smaller time steps (0.001s) for accuracy
   - Deployment: Larger time steps (0.01s) for performance
   - Use `real_time_factor` to control simulation speed

2. **Collision Detection**:
   - Use simple geometries (boxes, cylinders, spheres) for collision
   - Reserve complex meshes for visual representation only
   - Consider using bounding boxes for initial collision detection

## Troubleshooting Common Issues

### Issue: "Failed to create OpenGL context"
**Symptoms**: Gazebo fails to start with OpenGL errors
**Solutions**:
- Update your graphics drivers
- Ensure your GPU supports OpenGL 2.1+
- On WSL2, verify X-server is properly configured

### Issue: "No display found" on Linux
**Symptoms**: GUI applications fail to launch
**Solutions**:
- Ensure X11 server is running
- Check DISPLAY environment variable
- For headless systems, use `gz sim -s` for server-only mode

### Issue: "Library not found" errors
**Symptoms**: Missing library errors when launching Gazebo
**Solutions**:
- Run `sudo apt update && sudo apt upgrade`
- Verify proper environment setup with `source /usr/share/gz/setup.bash`
- Check that all dependencies are installed

### Issue: Poor simulation performance
**Symptoms**: Low real-time factor, lagging simulation
**Solutions**:
- Reduce physics update rate
- Simplify collision meshes
- Close other applications to free up resources
- Check CPU and GPU usage

## Customizing Your Setup

### Creating a Robot Package

Create a dedicated package for your digital twin robot:

```bash
# Create a workspace
mkdir -p ~/gazebo_digital_twin_ws/src
cd ~/gazebo_digital_twin_ws

# Create a robot description package
cd src
ros2 pkg create --build-type ament_cmake digital_twin_robot_description
```

### Setting Up Model Paths

Add your custom models to Gazebo's model path:

```bash
# Create model directory
mkdir -p ~/.gazebo/models/my_digital_twin_robot

# Add to environment (add to ~/.bashrc)
echo 'export GAZEBO_MODEL_PATH="~/.gazebo/models:$GAZEBO_MODEL_PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Testing Your Setup

### Basic Functionality Test

Run through these steps to ensure everything is working:

1. Launch Gazebo:
   ```bash
   gz sim
   ```

2. Verify you can open the GUI and select different worlds

3. Test command line usage:
   ```bash
   gz sim -s shapes.sdf
   ```

4. Verify ROS integration (in another terminal):
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 topic list
   ```

### Digital Twin Specific Test

Create a simple test to verify your setup is ready for digital twin applications:

1. Create a simple world file `digital_twin_test.sdf`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="digital_twin_test">
       <include>
         <uri>model://ground_plane</uri>
       </include>
       <include>
         <uri>model://sun</uri>
       </include>

       <!-- Add a simple robot -->
       <model name="test_robot">
         <pose>0 0 0.2 0 0 0</pose>
         <link name="chassis">
           <visual name="visual">
             <geometry>
               <box><size>0.5 0.3 0.2</size></box>
             </geometry>
           </visual>
           <collision name="collision">
             <geometry>
               <box><size>0.5 0.3 0.2</size></box>
             </geometry>
           </collision>
           <inertial>
             <mass>1.0</mass>
             <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
           </inertial>
         </link>
       </model>
     </world>
   </sdf>
   ```

2. Test the world:
   ```bash
   gz sim -v 4 digital_twin_test.sdf
   ```

## Next Steps

With Gazebo properly installed and configured, you're ready to move on to creating your first simulation environment. The next section will guide you through building custom worlds and configuring physics properties for realistic simulation.
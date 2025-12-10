---
title: Gazebo Plugins and Custom Environments Code Examples
sidebar_position: 9
---

# Gazebo Plugins and Custom Environments Code Examples

This section provides practical code examples for creating custom Gazebo plugins and building specialized simulation environments. These examples demonstrate advanced techniques for extending Gazebo's capabilities in digital twin applications.

## Overview

Gazebo plugins allow you to customize simulation behavior, add new sensor types, implement custom physics, and create specialized environments. This section covers:

- Model plugins for custom robot behaviors
- World plugins for environment modifications
- Sensor plugins for custom data processing
- System plugins for global simulation control

## Setting Up the Development Environment

Before creating plugins, set up your development workspace:

```bash
# Create a catkin workspace for plugins
mkdir -p ~/gazebo_plugins_ws/src
cd ~/gazebo_plugins_ws/src
catkin_init_workspace

# Create a package for your plugins
catkin_create_pkg digital_twin_gazebo_plugins
  roscpp std_msgs geometry_msgs sensor_msgs
  gazebo_ros_pkgs gazebo_msgs

cd ~/gazebo_plugins_ws
catkin_make
source devel/setup.bash
```

## Model Plugins

Model plugins are attached to specific models and can modify their behavior.

### Example 1: Simple Moving Platform Plugin

```cpp
/*
 * MovingPlatformPlugin.cpp
 * A plugin that moves a model in a circular path
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math.hh>

namespace gazebo
{
  class MovingPlatformPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      // Store the model pointer
      this->model = _model;

      // Get parameters from SDF
      if (_sdf->HasElement("radius"))
        this->radius = _sdf->Get<double>("radius");
      else
        this->radius = 1.0;

      if (_sdf->HasElement("speed"))
        this->speed = _sdf->Get<double>("speed");
      else
        this->speed = 0.5;

      // Initialize variables
      this->angle = 0.0;

      // Connect to the world update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&MovingPlatformPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Update angle based on time
      this->angle += this->speed * 0.001; // Assuming 1000Hz update rate

      // Calculate new position
      double x = this->radius * cos(this->angle);
      double y = this->radius * sin(this->angle);

      // Set model position
      ignition::math::Pose3d newPose(x, y, 0.5, 0, 0, this->angle);
      this->model->SetWorldPose(newPose);
    }

    private: physics::ModelPtr model;
    private: double radius;
    private: double speed;
    private: double angle;
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(MovingPlatformPlugin)
}
```

**SDF Usage**:
```xml
<model name="moving_platform">
  <pose>0 0 0.5 0 0 0</pose>
  <link name="link">
    <visual name="visual">
      <geometry>
        <box><size>1 1 0.1</size></box>
      </geometry>
    </visual>
    <collision name="collision">
      <geometry>
        <box><size>1 1 0.1</size></box>
      </geometry>
    </collision>
    <inertial>
      <mass>1.0</mass>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <plugin name="moving_platform_plugin" filename="libMovingPlatformPlugin.so">
    <radius>2.0</radius>
    <speed>1.0</speed>
  </plugin>
</model>
```

### Example 2: Robot Following Plugin

```cpp
/*
 * RobotFollowerPlugin.cpp
 * A plugin that makes one robot follow another
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math.hh>

namespace gazebo
{
  class RobotFollowerPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;

      // Get the name of the target robot
      if (_sdf->HasElement("target_robot"))
        this->targetRobotName = _sdf->Get<std::string>("target_robot");
      else
      {
        gzerr << "No target_robot specified\n";
        return;
      }

      // Get follow parameters
      if (_sdf->HasElement("follow_distance"))
        this->followDistance = _sdf->Get<double>("follow_distance");
      else
        this->followDistance = 1.0;

      // Get physics world
      this->world = this->model->GetWorld();

      // Connect to update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&RobotFollowerPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Get target robot
      physics::ModelPtr targetRobot = this->world->ModelByName(this->targetRobotName);
      if (!targetRobot)
      {
        gzerr << "Target robot '" << this->targetRobotName << "' not found\n";
        return;
      }

      // Get positions
      ignition::math::Pose3d robotPose = this->model->WorldPose();
      ignition::math::Pose3d targetPose = targetRobot->WorldPose();

      // Calculate desired position (behind target)
      double desiredX = targetPose.Pos().X() - this->followDistance * cos(targetPose.Rot().Yaw());
      double desiredY = targetPose.Pos().Y() - this->followDistance * sin(targetPose.Rot().Yaw());

      // Move towards desired position
      ignition::math::Vector3d desiredPos(desiredX, desiredY, robotPose.Pos().Z());
      ignition::math::Vector3d currentPos = robotPose.Pos();

      ignition::math::Vector3d direction = desiredPos - currentPos;
      direction.Z() = 0; // Keep at same height

      if (direction.Length() > 0.1) // Only move if significantly far away
      {
        direction.Normalize();
        ignition::math::Vector3d newVel = direction * 0.5; // Speed of 0.5 m/s
        this->model->SetLinearVel(newVel);
      }
    }

    private: physics::ModelPtr model;
    private: physics::WorldPtr world;
    private: std::string targetRobotName;
    private: double followDistance;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(RobotFollowerPlugin)
}
```

## World Plugins

World plugins affect the entire simulation world.

### Example 3: Dynamic Weather Plugin

```cpp
/*
 * DynamicWeatherPlugin.cpp
 * A plugin that changes weather conditions over time
 */

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class DynamicWeatherPlugin : public WorldPlugin
  {
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      this->world = _world;

      // Get parameters
      if (_sdf->HasElement("cycle_duration"))
        this->cycleDuration = _sdf->Get<double>("cycle_duration");
      else
        this->cycleDuration = 60.0; // seconds

      // Initialize timer
      this->lastUpdate = 0;

      // Connect to pre-update event
      this->updateConnection = event::Events::ConnectPreRender(
          std::bind(&DynamicWeatherPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      double currentTime = this->world->SimTime().Double();

      // Update weather every few seconds
      if (currentTime - this->lastUpdate > 5.0)
      {
        this->UpdateWeather(currentTime);
        this->lastUpdate = currentTime;
      }
    }

    private: void UpdateWeather(double _time)
    {
      // Calculate weather phase (0 to 1)
      double phase = fmod(_time, this->cycleDuration) / this->cycleDuration;

      // Calculate weather parameters based on phase
      double cloudiness = 0.2 + 0.8 * sin(phase * 2 * M_PI);
      double windSpeed = 0.5 + 2.0 * sin(phase * 2 * M_PI + M_PI/2);

      // Apply weather effects (simplified)
      sdf::ElementPtr atmosphere = this->world->SDF()->GetElement("atmosphere");
      if (atmosphere)
      {
        // In a real implementation, you would modify atmosphere properties
        // This is a simplified example
        gzmsg << "Weather phase: " << phase
              << ", Cloudiness: " << cloudiness
              << ", Wind: " << windSpeed << "\n";
      }
    }

    private: physics::WorldPtr world;
    private: double cycleDuration;
    private: double lastUpdate;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_WORLD_PLUGIN(DynamicWeatherPlugin)
}
```

## Sensor Plugins

Sensor plugins process data from Gazebo sensors.

### Example 4: Custom Camera Processing Plugin

```cpp
/*
 * CustomCameraPlugin.cpp
 * A plugin that processes camera data for digital twin applications
 */

#include <gazebo/gazebo.hh>
#include <gazebo/sensors/CameraSensor.hh>
#include <gazebo/sensors/SensorManager.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gazebo
{
  class CustomCameraPlugin : public SensorPlugin
  {
    public: virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Get the camera sensor
      this->cameraSensor = std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);
      if (!this->cameraSensor)
      {
        gzerr << "CustomCameraPlugin requires a camera sensor\n";
        return;
      }

      // Get the camera
      this->camera = this->cameraSensor->Camera();
      if (!this->camera)
      {
        gzerr << "Camera is null\n";
        return;
      }

      // Connect to the camera's new frame event
      this->newFrameConnection = this->camera->ConnectNewImageFrame(
          std::bind(&CustomCameraPlugin::OnNewFrame, this,
                   std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3, std::placeholders::_4,
                   std::placeholders::_5));

      // Get parameters
      if (_sdf->HasElement("process_frequency"))
        this->processFrequency = _sdf->Get<int>("process_frequency");
      else
        this->processFrequency = 10; // Hz

      this->frameCount = 0;
      this->processInterval = 1000 / this->processFrequency; // milliseconds
    }

    private: void OnNewFrame(const unsigned char *_image,
                            unsigned int _width, unsigned int _height,
                            unsigned int _depth, const std::string &_format)
    {
      this->frameCount++;

      // Process every N frames based on desired frequency
      if (this->frameCount % this->processInterval == 0)
      {
        // Convert Gazebo image to OpenCV Mat
        cv::Mat frame;
        if (_format == "R8G8B8")
        {
          frame = cv::Mat(_height, _width, CV_8UC3, (void *)_image);
        }
        else if (_format == "L8")
        {
          frame = cv::Mat(_height, _width, CV_8UC1, (void *)_image);
        }
        else
        {
          gzerr << "Unsupported image format: " << _format << "\n";
          return;
        }

        // Apply custom processing for digital twin (example: edge detection)
        cv::Mat processedFrame;
        cv::Canny(frame, processedFrame, 50, 150);

        // In a real implementation, you might:
        // - Analyze the image for specific features
        // - Send processed data to ROS topics
        // - Store data for digital twin synchronization
        // - Apply machine learning models

        gzdbg << "Processed frame " << this->frameCount
              << " with size " << _width << "x" << _height << "\n";
      }
    }

    private: sensors::CameraSensorPtr cameraSensor;
    private: rendering::CameraPtr camera;
    private: event::ConnectionPtr newFrameConnection;
    private: int processFrequency;
    private: int processInterval;
    private: int frameCount;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomCameraPlugin)
}
```

## Custom Environment Examples

### Example 5: Smart Factory Environment

Create a complete smart factory environment with various interactive elements:

**smart_factory.world**:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="smart_factory">
    <!-- Include default elements -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Factory floor -->
    <model name="factory_floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>20 15 0.1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>20 15 0.1</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Assembly station -->
    <model name="assembly_station">
      <pose>5 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="base">
        <collision name="collision">
          <geometry>
            <box><size>2 1 0.8</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1 0.8</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Conveyor belt -->
      <model name="conveyor_belt">
        <pose>0 0 0.9 0 0 0</pose>
        <link name="belt">
          <visual name="visual">
            <geometry>
              <box><size>1.8 0.8 0.05</size></box>
            </geometry>
            <material>
              <ambient>0.3 0.3 0.3 1</ambient>
              <diffuse>0.3 0.3 0.3 1</diffuse>
            </material>
          </visual>
        </link>

        <!-- Conveyor belt plugin -->
        <plugin name="conveyor_plugin" filename="libConveyorPlugin.so">
          <speed>0.2</speed>
          <width>0.8</width>
        </plugin>
      </model>
    </model>

    <!-- Quality control station -->
    <model name="quality_station">
      <pose>-5 2 0.5 0 0 0</pose>
      <static>true</static>
      <link name="base">
        <collision name="collision">
          <geometry>
            <box><size>1.5 1 0.8</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1.5 1 0.8</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Quality control camera -->
      <model name="qc_camera">
        <pose>0 0.6 1.2 0 0 0</pose>
        <link name="camera_link">
          <visual name="visual">
            <geometry>
              <box><size>0.1 0.1 0.1</size></box>
            </geometry>
          </visual>
          <sensor name="qc_camera_sensor" type="camera">
            <camera>
              <horizontal_fov>1.047</horizontal_fov>
              <image>
                <width>640</width>
                <height>480</height>
              </image>
              <clip>
                <near>0.1</near>
                <far>10</far>
              </clip>
            </camera>
            <plugin name="qc_camera_plugin" filename="libCustomCameraPlugin.so"/>
          </sensor>
        </link>
      </model>
    </model>

    <!-- Mobile robot -->
    <model name="mobile_robot">
      <pose>-2 -3 0.2 0 0 0</pose>
      <include>
        <uri>model://diff_drive_robot</uri>
      </include>
    </model>

    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- World plugin for factory operations -->
    <plugin name="factory_controller" filename="libFactoryControllerPlugin.so">
      <station_count>3</station_count>
      <product_cycle_time>30.0</product_cycle_time>
    </plugin>
  </world>
</sdf>
```

## Building and Using Plugins

### CMakeLists.txt for Plugin Package

```cmake
cmake_minimum_required(VERSION 3.5)
project(digital_twin_gazebo_plugins)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find packages
find_package(gazebo REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  gazebo_ros_pkgs
  gazebo_msgs
)

# Set Gazebo C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link directories
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_LIBRARY_PATH ${GAZEBO_LIBRARY_DIRS})

# Add plugin libraries
add_library(MovingPlatformPlugin SHARED MovingPlatformPlugin.cpp)
target_link_libraries(MovingPlatformPlugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})

add_library(RobotFollowerPlugin SHARED RobotFollowerPlugin.cpp)
target_link_libraries(RobotFollowerPlugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})

add_library(DynamicWeatherPlugin SHARED DynamicWeatherPlugin.cpp)
target_link_libraries(DynamicWeatherPlugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})

add_library(CustomCameraPlugin SHARED CustomCameraPlugin.cpp)
target_link_libraries(CustomCameraPlugin
  ${GAZEBO_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

# Install plugins
install(TARGETS
  MovingPlatformPlugin
  RobotFollowerPlugin
  DynamicWeatherPlugin
  CustomCameraPlugin
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

## Plugin Best Practices

### 1. Error Handling
Always check for null pointers and handle errors gracefully:

```cpp
// Good practice: Check for null pointers
if (!this->model)
{
  gzerr << "Model is null in plugin\n";
  return;
}

// Good practice: Validate SDF parameters
if (_sdf->HasElement("parameter"))
{
  double param = _sdf->Get<double>("parameter");
  if (param <= 0)
  {
    gzerr << "Invalid parameter value: " << param << "\n";
    param = 1.0; // Use default
  }
}
```

### 2. Performance Considerations
Optimize for real-time performance:

```cpp
// Process data at appropriate intervals
if (this->updateCount % this->processInterval == 0)
{
  this->ProcessData();
}
```

### 3. Memory Management
Properly manage memory in long-running simulations:

```cpp
// Clear temporary data periodically
if (this->frameCount % 1000 == 0)
{
  this->temporaryData.clear();
}
```

## Troubleshooting Common Issues

### Issue: Plugin not loading
**Symptoms**: Plugin fails to load with "filename not found" error
**Solutions**:
- Ensure plugin is built and installed in the correct location
- Check that GAZEBO_PLUGIN_PATH includes your plugin directory
- Verify the plugin filename matches the registered name

### Issue: Segmentation fault in plugin
**Symptoms**: Gazebo crashes when plugin is active
**Solutions**:
- Check for null pointer dereferences
- Ensure proper initialization order
- Use debug builds to identify the exact location of the crash

### Issue: Performance degradation
**Symptoms**: Simulation runs slowly when plugin is active
**Solutions**:
- Reduce update frequency
- Optimize algorithms
- Use efficient data structures

## Next Steps

With these plugin examples, you can create sophisticated custom behaviors for your digital twin environments. The next section will cover troubleshooting techniques for common Gazebo issues.
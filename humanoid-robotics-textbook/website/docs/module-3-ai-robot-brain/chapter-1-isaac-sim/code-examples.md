---
sidebar_position: 8
title: "Code Examples for Isaac Sim"
---

# Code Examples for Isaac Sim

## Overview
This document provides practical code examples for implementing Isaac Sim features for robotics simulation and synthetic data generation. These examples demonstrate how to create environments, configure sensors, and generate synthetic data programmatically.

## Learning Objectives
After reviewing these examples, you will be able to:
- Create Isaac Sim environments programmatically
- Configure and control sensors for data generation
- Implement custom simulation behaviors
- Integrate with ROS/ROS2 systems
- Apply best practices for Isaac Sim programming

## Prerequisites
- Isaac Sim installed and running
- Python programming knowledge
- Basic understanding of robotics concepts

## Basic Environment Setup

### 1. Initialize Isaac Sim World
```python
# Initialize Isaac Sim World
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np

# Create world instance
world = World(stage_units_in_meters=1.0)

# Set physics parameters
world.scene.add_default_ground_plane()

# Optional: Set gravity
world.set_physics_dt(1.0/60.0, substeps=1)
```

### 2. Create a Simple Robot Environment
```python
# Create a simple environment with a robot
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, Sdf

def create_simple_robot_environment():
    # Create a basic robot model
    robot = world.scene.add(
        Robot(
            prim_path="/World/Robot",
            name="simple_robot",
            usd_path=f"{get_assets_root_path()}/Isaac/Robots/TurtleBot3/turtlebot3.usd",
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
    )

    # Add obstacles to the environment
    create_prim(
        prim_path="/World/Obstacle1",
        prim_type="Cube",
        position=np.array([1.0, 1.0, 0.5]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        scale=np.array([0.5, 0.5, 0.5])
    )

    create_prim(
        prim_path="/World/Obstacle2",
        prim_type="Sphere",
        position=np.array([-1.0, -1.0, 0.5]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        scale=np.array([0.3, 0.3, 0.3])
    )

    return robot

# Create the environment
robot = create_simple_robot_environment()
```

## Sensor Configuration Examples

### 3. Configure RGB and Depth Cameras
```python
from omni.isaac.sensor import Camera
import carb

def setup_camera_sensor():
    # Create camera sensor
    camera = Camera(
        prim_path="/World/Robot/Camera",
        frequency=30,
        resolution=(640, 480),
        position=np.array([0.2, 0.0, 0.8]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )

    # Add data types to capture
    camera.add_data_to_frame("rgb")
    camera.add_data_to_frame("depth")
    camera.add_data_to_frame("semantic_segmentation")

    # Configure camera properties
    camera.post_process_lights_enabled = True

    return camera

# Set up the camera
camera = setup_camera_sensor()
```

### 4. Configure LiDAR Sensor
```python
from omni.isaac.sensor import RotatingLidarSensor
from omni.isaac.range_sensor import attach_lidar_sensor

def setup_lidar_sensor():
    # Create LiDAR sensor
    lidar_sensor = RotatingLidarSensor(
        prim_path="/World/Robot/LiDAR",
        translation=np.array([0.0, 0.0, 0.8]),
        configuration=RotatingLidarSensor.default_sensor_config()
    )

    # Configure LiDAR parameters
    lidar_config = lidar_sensor.get_sensor_config()
    lidar_config.rotation_rate = 10  # RPM
    lidar_config.number_of_channels = 16
    lidar_config.points_per_second = 240000
    lidar_config.horizontal_resolution = 0.18  # degrees
    lidar_config.vertical_resolution = 2.0  # degrees
    lidar_config.horizontal_lasers = 1875  # points per revolution
    lidar_config.vertical_lasers = 16
    lidar_config.range = 25.0  # Maximum range in meters

    return lidar_sensor

# Set up the LiDAR
lidar = setup_lidar_sensor()
```

### 5. Configure IMU Sensor
```python
from omni.isaac.sensor import IMU
import numpy as np

def setup_imu_sensor():
    # Create IMU sensor
    imu = IMU(
        prim_path="/World/Robot/IMU",
        name="imu_sensor",
        position=np.array([0.0, 0.0, 0.5]),
        frequency=100  # 100 Hz update rate
    )

    return imu

# Set up the IMU
imu = setup_imu_sensor()
```

## Synthetic Data Generation

### 6. Capture and Process Synthetic Data
```python
import cv2
import numpy as np
import os
from PIL import Image

def capture_synthetic_data(camera, output_dir="synthetic_data"):
    """Capture synthetic data from camera sensor"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/rgb", exist_ok=True)
    os.makedirs(f"{output_dir}/depth", exist_ok=True)
    os.makedirs(f"{output_dir}/segmentation", exist_ok=True)

    frame_count = 0

    # Run simulation and capture data
    for i in range(100):  # Capture 100 frames
        world.step(render=True)

        if i % 10 == 0:  # Capture every 10th frame
            # Get frame data
            frame = camera.get_frame()

            if "rgb" in frame:
                # Save RGB image
                rgb_image = frame["rgb"]
                rgb_pil = Image.fromarray(rgb_image, mode="RGBA")
                rgb_pil = rgb_pil.convert("RGB")  # Convert RGBA to RGB
                rgb_pil.save(f"{output_dir}/rgb/frame_{frame_count:04d}.jpg")

            if "depth" in frame:
                # Save depth map
                depth_data = frame["depth"]
                # Normalize depth for visualization
                depth_normalized = ((depth_data - depth_data.min()) /
                                   (depth_data.max() - depth_data.min()) * 255).astype(np.uint8)
                depth_image = Image.fromarray(depth_normalized)
                depth_image.save(f"{output_dir}/depth/frame_{frame_count:04d}.png")

            if "semantic_segmentation" in frame:
                # Save semantic segmentation
                seg_data = frame["semantic_segmentation"]["data"]
                seg_image = Image.fromarray(seg_data.astype(np.uint8))
                seg_image.save(f"{output_dir}/segmentation/frame_{frame_count:04d}.png")

            frame_count += 1

            print(f"Captured frame {frame_count}")

# Run the data capture
capture_synthetic_data(camera)
```

### 7. Domain Randomization Example
```python
import random
import numpy as np

def apply_domain_randomization():
    """Apply domain randomization to environment"""

    # Randomize lighting
    def randomize_lighting():
        # Get existing lights
        stage = get_current_stage()
        light_paths = ["/World/DistantLight", "/World/DomeLight"]

        for light_path in light_paths:
            light_prim = stage.GetPrimAtPath(light_path)
            if light_prim:
                # Randomize intensity
                intensity = random.uniform(300, 800)
                # Randomize color temperature
                color_temp = random.uniform(5000, 8000)

                # Apply randomization (conceptual - actual implementation depends on light type)
                print(f"Applied randomization to {light_path}")

    # Randomize materials
    def randomize_materials():
        # Define material ranges
        material_properties = {
            "albedo_range": [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]],  # [min, max]
            "roughness_range": [0.1, 0.9],
            "metallic_range": [0.0, 0.2]
        }

        # Apply randomization to objects
        print("Applied material randomization")

    # Randomize object positions
    def randomize_object_positions():
        # Get all objects in scene
        # Move them to random positions within bounds
        print("Applied object position randomization")

    # Execute randomization
    randomize_lighting()
    randomize_materials()
    randomize_object_positions()

# Apply domain randomization
apply_domain_randomization()
```

## Advanced Environment Creation

### 8. Create Warehouse Environment Programmatically
```python
def create_warehouse_environment():
    """Create a warehouse-like environment"""

    # Create floor
    floor = world.scene.add_ground_plane(
        "x",
        size=20.0,
        color=np.array([0.7, 0.7, 0.7]),
        albedo_texture_path=None
    )

    # Create walls
    wall_height = 3.0
    wall_thickness = 0.2

    # Back wall
    create_prim(
        prim_path="/World/Wall_Back",
        prim_type="Cube",
        position=np.array([0.0, -10.0, wall_height/2]),
        scale=np.array([20.0, wall_thickness, wall_height])
    )

    # Front wall
    create_prim(
        prim_path="/World/Wall_Front",
        prim_type="Cube",
        position=np.array([0.0, 10.0, wall_height/2]),
        scale=np.array([20.0, wall_thickness, wall_height])
    )

    # Side walls
    create_prim(
        prim_path="/World/Wall_Left",
        prim_type="Cube",
        position=np.array([-10.0, 0.0, wall_height/2]),
        scale=np.array([wall_thickness, 20.0, wall_height])
    )

    create_prim(
        prim_path="/World/Wall_Right",
        prim_type="Cube",
        position=np.array([10.0, 0.0, wall_height/2]),
        scale=np.array([wall_thickness, 20.0, wall_height])
    )

    # Create shelves
    shelf_width = 1.5
    shelf_depth = 0.8
    shelf_height = 2.0

    # Create shelf rows
    for row in range(4):
        for col in range(3):
            x_pos = -6.0 + col * 4.0
            y_pos = -8.0 + row * 4.0

            # Create shelf
            create_prim(
                prim_path=f"/World/Shelf_{row}_{col}",
                prim_type="Cube",
                position=np.array([x_pos, y_pos, shelf_height/2]),
                scale=np.array([shelf_width, shelf_depth, shelf_height]),
                color=np.array([0.8, 0.6, 0.2])  # Wood color
            )

    print("Warehouse environment created successfully")

# Create warehouse environment
create_warehouse_environment()
```

## ROS Integration Examples

### 9. Isaac Sim to ROS Bridge Configuration
```python
# Example of ROS bridge configuration
import rclpy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

def setup_ros_bridge():
    """Setup ROS bridge for Isaac Sim integration"""

    # Initialize ROS node
    rclpy.init()
    node = rclpy.create_node('isaac_sim_ros_bridge')

    # Create publishers
    rgb_pub = node.create_publisher(Image, '/camera/rgb/image_raw', 10)
    depth_pub = node.create_publisher(Image, '/camera/depth/image_raw', 10)
    cam_info_pub = node.create_publisher(CameraInfo, '/camera/rgb/camera_info', 10)

    def camera_callback(frame_data):
        """Callback function for camera data"""
        if "rgb" in frame_data:
            # Convert Isaac Sim RGB to ROS Image
            rgb_msg = convert_isaac_rgb_to_ros(frame_data["rgb"])
            rgb_pub.publish(rgb_msg)

        if "depth" in frame_data:
            # Convert Isaac Sim depth to ROS Image
            depth_msg = convert_isaac_depth_to_ros(frame_data["depth"])
            depth_pub.publish(depth_msg)

    # Setup camera callback
    camera.add_event_callback("on_frame", lambda frame: camera_callback(frame))

    return node

def convert_isaac_rgb_to_ros(isaac_rgb):
    """Convert Isaac Sim RGB data to ROS Image message"""
    # Implementation would convert Isaac format to ROS Image
    pass

def convert_isaac_depth_to_ros(isaac_depth):
    """Convert Isaac Sim depth data to ROS Image message"""
    # Implementation would convert Isaac format to ROS Image
    pass

# Setup ROS bridge (conceptual)
# ros_node = setup_ros_bridge()
```

## Performance Optimization Examples

### 10. Efficient Data Pipeline
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

class EfficientDataPipeline:
    def __init__(self, camera, output_dir="efficient_pipeline_data"):
        self.camera = camera
        self.output_dir = output_dir
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        self.frame_count = 0

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)

    def process_frame_async(self, frame_data, frame_num):
        """Process frame asynchronously"""
        # Process RGB data
        if "rgb" in frame_data:
            rgb_image = frame_data["rgb"]
            rgb_pil = Image.fromarray(rgb_image, mode="RGBA")
            rgb_pil = rgb_pil.convert("RGB")
            rgb_pil.save(f"{self.output_dir}/rgb/frame_{frame_num:04d}.jpg")

        # Process depth data
        if "depth" in frame_data:
            depth_data = frame_data["depth"]
            depth_normalized = ((depth_data - depth_data.min()) /
                               (depth_data.max() - depth_data.min()) * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_normalized)
            depth_image.save(f"{self.output_dir}/depth/frame_{frame_num:04d}.png")

    def capture_data_efficiently(self, total_frames=100, capture_interval=5):
        """Capture data with efficient pipeline"""
        frame_count = 0

        for i in range(total_frames):
            world.step(render=True)

            if i % capture_interval == 0:  # Capture every nth frame
                frame = self.camera.get_frame()

                # Submit processing to thread pool
                future = self.executor.submit(
                    self.process_frame_async,
                    frame,
                    frame_count
                )

                frame_count += 1
                print(f"Submitted frame {frame_count} for processing")

        # Shutdown executor
        self.executor.shutdown(wait=True)
        print("Data capture completed")

# Use efficient pipeline
# efficient_pipeline = EfficientDataPipeline(camera)
# efficient_pipeline.capture_data_efficiently()
```

## Error Handling and Validation

### 11. Data Quality Validation
```python
def validate_synthetic_data_quality(data_batch):
    """Validate the quality of synthetic data"""

    validation_results = {
        "image_quality": True,
        "annotation_accuracy": True,
        "completeness": True,
        "consistency": True
    }

    # Check for proper exposure in RGB images
    def check_image_exposure(images):
        for img in images:
            mean_brightness = np.mean(img)
            if mean_brightness < 20 or mean_brightness > 235:  # Too dark or too bright
                return False
        return True

    # Check depth data validity
    def check_depth_validity(depth_maps):
        for depth in depth_maps:
            valid_pixels = np.count_nonzero(~np.isnan(depth) & ~np.isinf(depth))
            total_pixels = depth.size
            if valid_pixels / total_pixels < 0.8:  # Less than 80% valid pixels
                return False
        return True

    # Check annotation completeness
    def check_annotation_completeness(annotations):
        for ann in annotations:
            if not ann or len(ann) == 0:
                return False
        return True

    # Run validation checks
    validation_results["image_quality"] = check_image_exposure(data_batch.get("rgb", []))
    validation_results["annotation_accuracy"] = check_annotation_completeness(data_batch.get("annotations", []))
    validation_results["completeness"] = check_depth_validity(data_batch.get("depth", []))

    # Overall quality score
    quality_score = sum(validation_results.values()) / len(validation_results)

    return quality_score, validation_results

# Example usage
# quality_score, results = validate_synthetic_data_quality(captured_data)
# print(f"Data quality score: {quality_score:.2f}")
```

## Complete Example: Autonomous Robot Navigation Simulation

### 12. Complete Simulation Example
```python
def run_complete_navigation_simulation():
    """Complete example of autonomous navigation simulation"""

    print("Setting up complete navigation simulation...")

    # 1. Initialize world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # 2. Create robot
    robot = world.scene.add(
        Robot(
            prim_path="/World/Robot",
            name="turtlebot_navigation",
            usd_path=f"{get_assets_root_path()}/Isaac/Robots/TurtleBot3/turtlebot3.usd",
            position=np.array([0.0, 0.0, 0.1]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
    )

    # 3. Create obstacles
    for i in range(5):
        create_prim(
            prim_path=f"/World/Obstacle_{i}",
            prim_type="Cylinder",
            position=np.array([
                random.uniform(-3.0, 3.0),
                random.uniform(-3.0, 3.0),
                0.5
            ]),
            scale=np.array([0.3, 0.3, 1.0]),
            color=np.array([0.8, 0.2, 0.2])
        )

    # 4. Add sensors
    camera = Camera(
        prim_path="/World/Robot/Camera",
        frequency=30,
        resolution=(640, 480),
        position=np.array([0.2, 0.0, 0.8]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    camera.add_data_to_frame("rgb")

    # 5. Simulation loop
    world.reset()

    for i in range(1000):  # Run for 1000 steps
        world.step(render=True)

        # Simple navigation logic
        if i % 100 == 0:  # Every 100 steps
            # Get current position
            current_pos = robot.get_world_pos()
            print(f"Step {i}: Robot at position {current_pos}")

            # Apply simple movement command
            robot.apply_wheel_actions(
                np.array([1.0, 1.0])  # Move forward
            )

        # Capture data periodically
        if i % 50 == 0:
            frame = camera.get_frame()
            if "rgb" in frame:
                # Process camera data for navigation
                pass

    print("Navigation simulation completed!")

# Run complete example
# run_complete_navigation_simulation()
```

## Best Practices and Tips

### 13. Performance Tips
```python
# Performance optimization tips as code examples

# 1. Use appropriate physics dt
world.set_physics_dt(1.0/60.0, substeps=1)  # Balance accuracy and performance

# 2. Optimize rendering settings for data generation
carb.settings.get_settings().set("/app/renderer/milliseconds", 16.67)  # 60 FPS

# 3. Use appropriate sensor frequencies
# High frequency for control (100Hz+)
# Medium frequency for navigation (10-30Hz)
# Low frequency for mapping (1-5Hz)

# 4. Optimize scene complexity
# Use Level of Detail (LOD) where possible
# Implement occlusion culling
# Reduce poly count where detail isn't needed

# 5. Memory management for large datasets
def process_large_datasets_in_chunks(data, chunk_size=1000):
    """Process large datasets in chunks to manage memory"""
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        # Process chunk
        yield process_chunk(chunk)
```

## Troubleshooting Common Issues

### 14. Common Error Solutions
```python
# Common issues and solutions

def troubleshoot_common_issues():
    """Common troubleshooting examples"""

    # Issue 1: Camera not capturing data
    try:
        camera.get_frame()
    except Exception as e:
        print(f"Camera error: {e}")
        # Solution: Verify camera configuration
        print("Verify camera is properly configured and positioned")

    # Issue 2: Physics simulation instability
    try:
        world.step(render=True)
    except Exception as e:
        print(f"Physics error: {e}")
        # Solution: Check mass properties and scales
        print("Verify all objects have appropriate mass and scale")

    # Issue 3: Memory issues with large datasets
    try:
        # Process large dataset
        pass
    except MemoryError:
        print("Memory error - process data in smaller chunks")
        # Solution: Implement streaming/chunking
        print("Use data generators and process in batches")

# Call troubleshooting
troubleshoot_common_issues()
```

## Next Steps and Further Learning

### 15. Extending Examples
```python
# Ideas for extending the examples

# 1. Add more complex sensors
# - Thermal cameras
# - Event-based cameras
# - Multi-modal sensors

# 2. Implement advanced scenarios
# - Dynamic environments
# - Multi-robot scenarios
# - Adversarial conditions

# 3. Integration with AI frameworks
# - Direct PyTorch integration
# - TensorFlow compatibility
# - Custom training loops

# 4. Advanced domain randomization
# - Weather conditions
# - Time of day variations
# - Seasonal changes
```

## Exercise: Implement Your Own Environment
Try implementing the following:

1. Create a custom environment with specific lighting conditions
2. Add multiple sensor types to your robot
3. Implement domain randomization for your specific use case
4. Generate a small dataset and validate its quality
5. Integrate with a simple ROS node for data processing

These code examples provide a foundation for building sophisticated Isaac Sim environments and synthetic data generation pipelines. Each example builds upon the previous ones, allowing you to create increasingly complex and realistic simulation environments for robotics development and AI training.
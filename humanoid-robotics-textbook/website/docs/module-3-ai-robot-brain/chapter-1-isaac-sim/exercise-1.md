---
sidebar_position: 6
title: "Exercise 1: Basic Isaac Sim Environment and Synthetic Data Generation"
---

# Exercise 1: Basic Isaac Sim Environment and Synthetic Data Generation

## Overview
This hands-on exercise will guide you through creating a basic Isaac Sim environment and generating your first synthetic dataset. You'll create a simple indoor scene, configure sensors, and generate annotated data suitable for AI model training.

## Learning Objectives
After completing this exercise, you will be able to:
- Create a basic indoor environment in Isaac Sim
- Configure RGB and depth sensors for data collection
- Generate a small synthetic dataset with annotations
- Validate the quality of generated data
- Export data in a standard format

## Prerequisites
- Isaac Sim installed and running
- Completed the environment creation and synthetic data generation tutorials
- Basic understanding of robotics and AI concepts
- Estimated time to complete: 45-60 minutes

## Setup Requirements
1. **Environment**: Isaac Sim application running
2. **Tools**: Isaac Sim interface, text editor for configuration files
3. **Data**: No additional data files required (all created during exercise)

## Exercise Steps

### Step 1: Create Basic Indoor Environment
**Task**: Create a simple indoor environment with walls, floor, and obstacles
1. Launch Isaac Sim and create a new stage
2. Create a 5m x 5m floor using a plane primitive
3. Add 4 walls around the perimeter (each 5m long, 3m high)
4. Place 3-5 simple objects (cubes, spheres, cylinders) as obstacles
5. Add appropriate materials to surfaces (floor, walls, objects)

**Expected Result**: A simple rectangular room with obstacles inside
**Verification**: You should see a complete indoor environment in the viewport

### Step 2: Configure Camera Sensors
**Task**: Add and configure RGB and depth sensors
1. Add an RGB camera at a height of 1.5m (simulating robot-mounted camera)
2. Configure the camera with:
   - Resolution: 640x480 pixels
   - Field of view: 60 degrees
   - Frame rate: 30 FPS
3. Enable depth data collection alongside RGB
4. Position the camera to have a clear view of the environment

**Expected Result**: Camera positioned in the scene with both RGB and depth data enabled
**Verification**: You should see the camera's view in Isaac Sim and be able to access both RGB and depth data

### Step 3: Add Semantic Annotations
**Task**: Assign semantic labels to objects in the scene
1. Create a semantic schema for your scene
2. Assign labels to different objects:
   - Floor: "floor" (class ID: 1)
   - Walls: "wall" (class ID: 2)
   - Obstacles: "obstacle" (class ID: 3)
   - Camera: "sensor" (class ID: 4)
3. Verify that each object has the correct semantic label

**Expected Result**: All objects in the scene have semantic labels assigned
**Verification**: You should be able to generate semantic segmentation data for the scene

### Step 4: Generate Sample Dataset
**Task**: Capture a small dataset of images with annotations
1. Configure your environment for data collection:
   - Set up a simple trajectory for the camera (or use a static position)
   - Enable data collection for RGB, depth, and semantic segmentation
   - Configure the capture rate (e.g., 1 frame per second)
2. Generate 20-30 frames of data
3. Move the camera to different positions to capture varied perspectives

**Expected Result**: A dataset of 20-30 frames with RGB, depth, and semantic segmentation
**Verification**: You should have captured multiple frames with all required data types

### Step 5: Validate Data Quality
**Task**: Check the quality and completeness of your generated data
1. Examine the RGB images for proper exposure and focus
2. Verify depth maps have correct depth values
3. Check semantic segmentation masks align with objects
4. Ensure all frames have complete annotation data

**Expected Result**: All captured frames have high-quality data with proper annotations
**Verification**: Each frame should contain valid RGB, depth, and segmentation data

### Step 6: Export Dataset
**Task**: Export your dataset in a standard format
1. Create a directory for your dataset: `isaac_sim_exercise_1_dataset`
2. Export data in a structured format:
   - RGB images in `rgb/` folder
   - Depth maps in `depth/` folder
   - Semantic segmentation in `segmentation/` folder
   - Annotations in `annotations.json`
3. Include metadata file with capture parameters

**Expected Result**: Well-organized dataset directory with all data and annotations
**Verification**: You should have a complete dataset ready for AI training

## Troubleshooting Tips
- **Issue**: Camera not capturing depth data
  **Solution**: Verify that depth data is enabled in the camera configuration
- **Issue**: Semantic segmentation not working
  **Solution**: Check that semantic labels are properly assigned to objects
- **Issue**: Slow performance during data capture
  **Solution**: Reduce scene complexity or lower capture resolution
- **Issue**: Missing annotations in exported data
  **Solution**: Verify all data types are enabled before capture

## Challenge Questions
1. How would you modify the environment to make it more challenging for a perception model to train on?
2. What domain randomization techniques could you apply to make the dataset more diverse?
3. How would you extend this exercise to include LiDAR data generation?
4. What metrics would you use to evaluate the quality of your synthetic dataset?

## Success Criteria
To successfully complete this exercise, you must:
- [ ] Create a basic indoor environment with walls, floor, and obstacles
- [ ] Configure RGB and depth sensors with proper parameters
- [ ] Assign semantic labels to all objects in the scene
- [ ] Generate 20-30 frames of synthetic data with annotations
- [ ] Validate the quality of the generated data
- [ ] Export the dataset in an organized format
- [ ] Document the capture parameters and environment configuration

## Sample Solution Code
```python
# Example script for data generation (conceptual)
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np

# Initialize world
world = World(stage_units_in_meters=1.0)

# Create environment
floor = world.scene.add_ground_plane("x", 5.0, 0.0, 1.0, "visual_materials")
# Add walls and obstacles...

# Create camera
camera = Camera(
    prim_path="/World/Camera",
    frequency=30,
    resolution=(640, 480),
    position=np.array([0.0, 0.0, 1.5])
)

# Enable data collection
camera.add_data_to_frame("rgb")
camera.add_data_to_frame("depth")
camera.add_data_to_frame("semantic_segmentation")

# Generate data
for frame in range(30):
    world.step(render=True)
    data = camera.get_frame()
    save_frame_data(data, f"frame_{frame:03d}")
```

## Next Steps
After completing this exercise, proceed to:
- Experiment with different environment configurations
- Try generating data with domain randomization
- Explore more complex sensor configurations
- Move to Chapter 2 to learn about Isaac ROS integration

## Additional Resources
- [Isaac Sim Data Generation Tutorial](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_synthetic_data.html)
- [Semantic Segmentation in Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_semantic.html)
- [Camera Configuration Guide](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_cameras.html)
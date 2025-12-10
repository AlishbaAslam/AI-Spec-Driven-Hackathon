---
sidebar_position: 3
title: "Environment Creation in Isaac Sim"
---

# Environment Creation in Isaac Sim

## Overview
This tutorial guides you through creating photorealistic environments in Isaac Sim. You'll learn to build realistic scenes with accurate lighting, materials, and physics properties that can be used for robot simulation and synthetic data generation.

## Learning Objectives
After completing this section, you will be able to:
- Create and configure realistic 3D environments
- Set up proper lighting and materials for photorealistic rendering
- Configure physics properties for accurate simulation
- Import and place objects in the simulation environment
- Optimize environments for performance

## Prerequisites
- Isaac Sim installed and running
- Basic understanding of 3D modeling concepts
- Completed the installation and setup guide

## Creating Your First Environment

### Step 1: Initialize a New Stage
1. Launch Isaac Sim
2. Create a new stage by selecting **File → New Stage**
3. You'll see a default stage with a ground plane and basic lighting

### Step 2: Understanding the Scene Hierarchy
The scene hierarchy in Isaac Sim follows the Universal Scene Description (USD) format:
```
World/
├── GroundPlane/
├── Lights/
│   ├── DistantLight/
│   └── DomeLight/
└── Cameras/
    └── ViewportCamera/
```

### Step 3: Adding Basic Geometry
1. In the **Create** menu, select **Primitive → Cube**
2. Position the cube in the scene using the transform gizmo
3. Adjust the cube's properties in the **Property** window:
   - **Size**: Set to desired dimensions
   - **Position**: Adjust X, Y, Z coordinates
   - **Rotation**: Set orientation

## Lighting Setup for Photorealism

### Types of Lights in Isaac Sim
Isaac Sim supports several light types for realistic rendering:

#### Distant Light (Sun)
```python
# Example of creating a distant light via scripting
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage

# Create distant light
create_prim(
    prim_path="/World/DistantLight",
    prim_type="DistantLight",
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1],
    attributes={
        "inputs:intensity": 500,
        "inputs:angle": 0.5,
        "inputs:color": [0.9, 0.9, 0.8]
    }
)
```

#### Dome Light (Environment)
The dome light provides ambient lighting and environment reflections:
1. Select **Create → Light → Dome Light**
2. Configure the dome texture for realistic environment lighting
3. Adjust intensity and color temperature

#### Rect Light (Artificial)
For indoor scenes with artificial lighting:
1. Select **Create → Light → Rect Light**
2. Position and orient the light appropriately
3. Set color and intensity for the desired effect

### Lighting Best Practices
- Use a combination of distant and dome lighting for outdoor scenes
- Add local lights for indoor environments
- Match lighting conditions to your real-world application
- Consider the time of day and weather conditions

## Material and Surface Properties

### Creating Realistic Materials
1. Open the **Material** window (**Window → Material → Material Browser**)
2. Create a new material by right-clicking in the browser
3. Configure material properties:
   - **Albedo**: Base color of the surface
   - **Roughness**: Surface micro-geometry
   - **Metallic**: Metallic properties
   - **Normal**: Surface detail mapping

### Material Examples for Robotics Environments
#### Floor Materials
```usd
# Example floor material configuration
{
    "albedo": [0.8, 0.8, 0.8],  # Light gray for visibility
    "roughness": 0.8,           # Non-reflective for floor
    "metallic": 0.0,            # Non-metallic
    "normal": "floor_normal_map.png"
}
```

#### Wall Materials
```usd
# Example wall material configuration
{
    "albedo": [0.7, 0.7, 0.7],  # Neutral gray
    "roughness": 0.9,           # Matte finish
    "metallic": 0.0,            # Non-metallic
    "specular": 0.5             # Some reflectivity
}
```

## Physics Configuration

### Setting Physics Properties
For objects to interact realistically in simulation:

1. **Collision Meshes**: Ensure objects have proper collision geometry
   - Use convex hulls for simple objects
   - Use mesh colliders for complex geometry
   - Adjust collision margins as needed

2. **Mass Properties**: Configure mass, center of mass, and inertia
   - Set realistic densities for materials
   - Ensure proper mass distribution
   - Consider the actual robot's weight for scale

3. **Friction and Restitution**:
   - Static friction: Resistance to initial movement
   - Dynamic friction: Resistance during sliding
   - Restitution: Bounciness (0 = no bounce, 1 = perfectly elastic)

### Physics Materials
Create physics materials for consistent surface properties:
```python
# Example physics material configuration
from omni.physx.scripts.physicsUtils import *

# Define physics material properties
material_params = {
    "staticFriction": 0.5,
    "dynamicFriction": 0.4,
    "restitution": 0.1,
    "friction_combine_mode": "average",
    "restitution_combine_mode": "average"
}
```

## Importing Custom Assets

### Supported File Formats
Isaac Sim supports various asset formats:
- **USD**: Native format, recommended
- **FBX**: Common interchange format
- **OBJ**: Simple geometry format
- **GLTF/GLB**: Modern format with materials

### Asset Preparation Guidelines
1. **Scale**: Ensure assets are properly scaled (1 unit = 1 meter)
2. **Centering**: Center objects at origin [0, 0, 0] where appropriate
3. **UV Mapping**: Proper UV coordinates for textures
4. **LOD**: Consider Level of Detail for performance

### Importing Process
1. **File → Import** or drag and drop files
2. Configure import settings:
   - **Scale factor**: Adjust if asset units differ
   - **Import materials**: Choose whether to import materials
   - **Import animations**: If applicable

## Creating a Sample Warehouse Environment

Let's create a simple warehouse environment as an example:

### Environment Layout
```
Warehouse Layout (10m x 10m):
┌─────────────────────────────────┐
│    [Shelf]      [Shelf]         │
│    [Shelf]      [Shelf]         │
│                                 │
│  [Robot Start]                  │
│                                 │
│    [Shelf]      [Shelf]         │
│    [Shelf]      [Shelf]         │
└─────────────────────────────────┘
```

### Step-by-Step Creation
1. **Create Ground Plane**
   - Create a 10m x 10m rectangle
   - Apply warehouse floor material
   - Configure physics properties (high friction, low restitution)

2. **Add Lighting**
   - Distant light for overhead lighting
   - Intensity: 500, Color: warm white
   - Dome light for ambient illumination

3. **Create Shelves**
   - Use primitive cubes for basic shelf structure
   - Position 4 shelves in a 2x2 grid
   - Apply metal/wood materials as appropriate
   - Configure collision properties for interaction

4. **Add Obstacles**
   - Place various objects on shelves
   - Add dynamic objects that the robot might encounter
   - Configure physics properties appropriately

## Performance Optimization

### Level of Detail (LOD)
- Use simplified geometry for distant objects
- Reduce polygon count where detail isn't needed
- Implement texture streaming for large environments

### Occlusion Culling
- Hide objects not visible to cameras
- Use occluders to block rendering of hidden geometry
- Implement frustum culling for camera views

### Texture Optimization
- Use compressed texture formats
- Implement texture atlasing where possible
- Adjust texture resolution based on viewing distance

## Quality Validation

### Visual Quality Checks
- Verify lighting looks realistic
- Check material properties are appropriate
- Ensure shadows and reflections are correct
- Validate color accuracy

### Physics Quality Checks
- Test object interactions are realistic
- Verify friction and collision properties
- Check for physics artifacts or instabilities
- Ensure simulation runs at expected frame rates

## Troubleshooting Common Issues

### Lighting Issues
- **Problem**: Dark or overly bright scenes
- **Solution**: Check light intensities and exposure settings
- **Alternative**: Adjust camera exposure or dome light intensity

### Physics Issues
- **Problem**: Objects falling through surfaces
- **Solution**: Verify collision meshes and physics properties
- **Check**: Ensure proper scale and mass properties

### Performance Issues
- **Problem**: Low frame rates in complex scenes
- **Solution**: Optimize geometry and reduce draw calls
- **Alternative**: Adjust rendering quality settings

## Advanced Environment Features

### Procedural Environment Generation
For large-scale environments, consider procedural generation:
```python
# Example procedural shelf placement
import random

def create_warehouse_layout(rows, cols, spacing=2.0):
    for i in range(rows):
        for j in range(cols):
            x_pos = j * spacing - (cols * spacing / 2)
            y_pos = i * spacing - (rows * spacing / 2)

            # Create shelf at position
            create_shelf(x_pos, y_pos, random_height())
```

### Dynamic Environment Elements
- Moving obstacles
- Changing lighting conditions
- Interactive elements
- Weather effects (if supported)

## Next Steps
After creating your environment:
- Test with robot models to verify functionality
- Generate synthetic sensor data
- Optimize for your specific use case
- Document environment parameters for consistency

## Additional Resources
- [Isaac Sim Environment Tutorials](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_basic_env.html)
- [USD Documentation](https://graphics.pixar.com/usd/release/wp_usd.html)
- [Material Creation Guide](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_materials.html)

## Exercise
Create a simple indoor environment with at least:
- 1 floor with appropriate material
- 4 walls with different materials
- 3 obstacles of varying sizes and materials
- Proper lighting setup
- Physics properties configured for realistic interaction
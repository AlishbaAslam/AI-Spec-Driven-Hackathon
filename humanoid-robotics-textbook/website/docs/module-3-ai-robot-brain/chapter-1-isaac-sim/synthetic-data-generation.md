---
sidebar_position: 4
title: "Synthetic Data Generation"
---

# Synthetic Data Generation in Isaac Sim

## Overview
This section covers synthetic data generation techniques in Isaac Sim, which is crucial for AI model training. Synthetic data generation allows you to create large, diverse datasets with accurate annotations that can be used to train perception models without the need for expensive real-world data collection.

## Learning Objectives
After completing this section, you will be able to:
- Configure sensors for synthetic data collection
- Generate diverse datasets with proper annotations
- Apply domain randomization techniques
- Export synthetic data in various formats
- Validate synthetic data quality for AI training

## Prerequisites
- Environment creation knowledge
- Understanding of sensor types and configurations
- Basic knowledge of AI/ML concepts

## Understanding Synthetic Data Generation

### Why Synthetic Data?
Synthetic data generation offers several advantages:
- **Cost Effective**: No need for expensive real-world data collection
- **Safety**: Train models without real-world risks
- **Diversity**: Generate rare or dangerous scenarios
- **Annotations**: Perfect ground truth available automatically
- **Control**: Precise control over environmental conditions

### Types of Synthetic Data
1. **RGB Images**: Color camera data for visual perception
2. **Depth Maps**: Depth information for 3D understanding
3. **Point Clouds**: LiDAR simulation data
4. **Semantic Segmentation**: Pixel-level object classification
5. **Instance Segmentation**: Individual object identification
6. **Bounding Boxes**: Object detection annotations
7. **Sensor Fusion**: Combined data from multiple sensors

## Sensor Configuration for Data Generation

### Camera Sensors
Configuring cameras for optimal synthetic data generation:

#### Basic Camera Setup
```python
from omni.isaac.sensor import Camera
import numpy as np

# Create RGB camera
rgb_camera = Camera(
    prim_path="/World/Robot/Camera",
    frequency=30,  # Hz
    resolution=(640, 480),
    position=np.array([0.0, 0.0, 0.5]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0])
)

# Configure for data collection
rgb_camera.add_data_to_frame("rgb")
rgb_camera.add_data_to_frame("depth")
rgb_camera.add_data_to_frame("semantic_segmentation")
```

#### Advanced Camera Properties
```yaml
camera_properties:
  resolution: [1280, 720]  # Higher resolution for detail
  fov: 60.0  # Field of view in degrees
  focal_length: 24.0  # Focal length in mm
  clipping_range: [0.1, 100.0]  # Near and far clipping
  sensor_horizontal_size: 36.0  # Sensor size for depth of field
  focus_distance: 10.0  # Focus distance in meters
```

### LiDAR Sensors
Configuring LiDAR for synthetic point cloud generation:

```python
from omni.isaac.sensor import RotatingLidarSensor

# Create 360-degree LiDAR
lidar_sensor = RotatingLidarSensor(
    prim_path="/World/Robot/LiDAR",
    translation=np.array([0.0, 0.0, 0.8]),
    configuration=RotatingLidarSensor.default_sensor_config()
)

# Configure LiDAR parameters
lidar_config = {
    "rotation_rate": 10,  # RPM
    "number_of_channels": 16,
    "points_per_second": 240000,
    "horizontal_resolution": 0.18,  # degrees
    "vertical_resolution": 2.0,  # degrees
    "horizontal_lasers": 1875,  # points per revolution
    "vertical_lasers": 16,
    "range": 25.0  # Maximum range in meters
}
```

### Multi-Sensor Configuration
For comprehensive synthetic datasets:

```python
# Multi-sensor robot configuration
robot_sensors = {
    "front_camera": {
        "type": "rgb",
        "resolution": [1280, 720],
        "fov": 90.0,
        "position": [0.3, 0.0, 0.8]
    },
    "left_camera": {
        "type": "rgb",
        "resolution": [640, 480],
        "fov": 60.0,
        "position": [0.1, 0.2, 0.8]
    },
    "rear_camera": {
        "type": "rgb",
        "resolution": [640, 480],
        "fov": 120.0,
        "position": [-0.3, 0.0, 0.8]
    },
    "lidar": {
        "type": "lidar",
        "range": 25.0,
        "channels": 32,
        "rpm": 10
    },
    "imu": {
        "type": "imu",
        "update_rate": 100
    }
}
```

## Domain Randomization Techniques

### What is Domain Randomization?
Domain randomization is a technique that increases the diversity of synthetic data by randomly varying environmental parameters to improve model generalization.

### Lighting Randomization
```python
import random

def randomize_lighting():
    # Randomize distant light properties
    light_intensity = random.uniform(300, 800)
    light_color_temp = random.uniform(5000, 8000)  # Kelvin
    light_direction = [
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(-1, 1)
    ]

    # Randomize dome light environment
    dome_envs = ["indoor_warehouse", "outdoor_day", "outdoor_night", "indoor_office"]
    selected_env = random.choice(dome_envs)

    return {
        "intensity": light_intensity,
        "color_temp": light_color_temp,
        "direction": light_direction,
        "environment": selected_env
    }
```

### Material Randomization
```python
def randomize_materials():
    # Randomize surface properties
    materials = {
        "floor": {
            "albedo": [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)],
            "roughness": random.uniform(0.1, 0.9),
            "metallic": random.uniform(0.0, 0.2),
            "texture": random.choice(["concrete", "tile", "wood", "metal"])
        },
        "objects": {
            "albedo_range": [[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]],
            "roughness_range": [0.2, 0.8],
            "metallic_range": [0.0, 0.5]
        }
    }
    return materials
```

### Object Placement Randomization
```python
def randomize_object_placement(env_bounds, num_objects):
    objects = []
    for i in range(num_objects):
        obj = {
            "position": [
                random.uniform(env_bounds["x_min"], env_bounds["x_max"]),
                random.uniform(env_bounds["y_min"], env_bounds["y_max"]),
                random.uniform(env_bounds["z_min"], env_bounds["z_max"])
            ],
            "rotation": [
                random.uniform(0, 360),  # degrees
                random.uniform(0, 360),
                random.uniform(0, 360)
            ],
            "scale": random.uniform(0.5, 2.0),
            "type": random.choice(["box", "cylinder", "sphere", "capsule"])
        }
        objects.append(obj)
    return objects
```

## Data Annotation Generation

### Semantic Segmentation
Isaac Sim automatically generates semantic segmentation data:

```python
# Configure semantic segmentation
from omni.isaac.synthetic_utils import SemanticSchema

# Assign semantic labels to objects
def assign_semantic_labels():
    semantic_schema = SemanticSchema()

    # Define semantic classes
    classes = {
        "robot": 1,
        "obstacle": 2,
        "floor": 3,
        "wall": 4,
        "shelf": 5,
        "person": 6
    }

    # Apply labels to objects
    for obj_name, class_id in classes.items():
        obj_prim = get_prim_at_path(f"/World/{obj_name}")
        if obj_prim:
            semantic_schema.add_label(obj_prim, class_id, obj_name)
```

### Instance Segmentation
For individual object identification:

```python
def generate_instance_masks():
    # Each unique object gets its own instance ID
    instance_data = {}

    # Iterate through all objects in scene
    for i, obj in enumerate(get_all_objects_in_scene()):
        instance_id = i + 1  # Instance IDs start from 1
        instance_data[obj.name] = {
            "instance_id": instance_id,
            "class": get_object_class(obj),
            "bbox": compute_bounding_box(obj)
        }

    return instance_data
```

### 3D Bounding Boxes
Generating 3D bounding box annotations:

```python
def generate_3d_bounding_boxes():
    boxes_3d = []

    for obj in get_all_objects_in_scene():
        bbox_3d = {
            "center": get_object_center(obj),
            "size": get_object_size(obj),
            "rotation": get_object_rotation(obj),
            "class": get_object_class(obj),
            "instance_id": get_instance_id(obj)
        }
        boxes_3d.append(bbox_3d)

    return boxes_3d
```

## Data Export and Formats

### Export Configuration
```yaml
export_config:
  format: "coco"  # COCO, KITTI, or custom
  output_directory: "/path/to/dataset"
  include_rgb: true
  include_depth: true
  include_segmentation: true
  include_bounding_boxes: true
  include_3d_annotations: true
  annotation_format: "json"  # JSON, XML, or text
  compression: "png"  # For images
  quality: 95  # Image quality percentage
```

### COCO Format Export
```python
def export_to_coco_format(data, output_path):
    coco_format = {
        "info": {
            "description": "Synthetic Robotics Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "AI-Robot Brain Module",
            "date_created": "2025-12-10"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Populate COCO format with synthetic data
    # ... implementation details ...

    # Save to file
    import json
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
```

### KITTI Format for Autonomous Driving
```python
def export_to_kitti_format(data, output_path):
    # Create KITTI-style directory structure
    os.makedirs(f"{output_path}/image_2", exist_ok=True)  # RGB images
    os.makedirs(f"{output_path}/label_2", exist_ok=True)  # Labels
    os.makedirs(f"{output_path}/velodyne", exist_ok=True) # Point clouds

    # Export each frame
    for frame_idx, frame_data in enumerate(data):
        # Save image
        cv2.imwrite(f"{output_path}/image_2/{frame_idx:06d}.png", frame_data["rgb"])

        # Save labels in KITTI format
        with open(f"{output_path}/label_2/{frame_idx:06d}.txt", 'w') as f:
            for obj in frame_data["objects"]:
                line = f"{obj['class']} -1 -1 -10 {obj['bbox'][0]} {obj['bbox'][1]} {obj['bbox'][2]} {obj['bbox'][3]} -1 -1 -1 -1000 -1000 -1000 -10\n"
                f.write(line)
```

## Quality Validation

### Data Quality Metrics
```python
def validate_synthetic_data_quality(data_batch):
    quality_metrics = {
        "image_quality": check_image_quality(data_batch["rgb"]),
        "annotation_accuracy": verify_annotations(data_batch),
        "diversity_score": measure_dataset_diversity(data_batch),
        "realism_score": assess_realism(data_batch),
        "completeness": check_annotation_completeness(data_batch)
    }

    # Return overall quality assessment
    overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
    return overall_quality, quality_metrics
```

### Validation Checks
1. **Image Quality**: Check for proper exposure, focus, and artifacts
2. **Annotation Accuracy**: Verify bounding boxes align with objects
3. **Data Completeness**: Ensure all required annotations exist
4. **Consistency**: Check for temporal consistency in sequences
5. **Realism**: Compare synthetic data characteristics to real data

## Performance Optimization

### Batch Processing
```python
def generate_data_in_batches(total_samples, batch_size=100):
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)

        # Randomize environment for batch
        randomize_environment()

        # Generate samples for batch
        for i in range(batch_start, batch_end):
            capture_data_sample(i)

        # Save batch to disk
        save_batch(batch_start, batch_end)
```

### Parallel Generation
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_data_generation(num_processes=4):
    # Create multiple Isaac Sim instances
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            future = executor.submit(generate_data_worker, i)
            futures.append(future)

        # Collect results
        results = [future.result() for future in futures]
        return results
```

## Best Practices

### 1. Progressive Complexity
- Start with simple environments
- Gradually increase complexity
- Validate at each step

### 2. Consistent Annotation Standards
- Use consistent class definitions
- Maintain annotation quality
- Document annotation guidelines

### 3. Metadata Management
- Track all environmental parameters
- Record sensor configurations
- Document randomization ranges

### 4. Validation and Testing
- Test models on real data when possible
- Use subset of synthetic data for validation
- Monitor for domain shift issues

## Troubleshooting

### Common Issues and Solutions
1. **Poor Image Quality**: Check lighting configuration and camera settings
2. **Missing Annotations**: Verify semantic schema is properly configured
3. **Performance Issues**: Optimize scene complexity and sensor settings
4. **Inconsistent Data**: Check randomization bounds and seed management

## Next Steps
After generating synthetic data:
- Train perception models using the synthetic dataset
- Validate model performance on real-world data
- Iterate on environment design based on results
- Expand dataset with additional scenarios

## Additional Resources
- [Isaac Sim Synthetic Data Generation Guide](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_synthetic_data.html)
- [Domain Randomization Best Practices](https://research.nvidia.com/labs/toronto-ai/DomainRandomization/)
- [COCO Dataset Format Specification](https://cocodataset.org/#format-data)

## Exercise
Create a synthetic dataset with:
- 1000 RGB images with random lighting
- Semantic segmentation annotations for 5 object classes
- Bounding box annotations for all objects
- Depth maps for 3D understanding
- Export in COCO format for AI training
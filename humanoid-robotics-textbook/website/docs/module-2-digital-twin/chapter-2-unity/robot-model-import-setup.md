---
title: Robot Model Import and Setup Tutorial
sidebar_position: 3
---

# Robot Model Import and Setup Tutorial

This tutorial provides comprehensive guidance on importing and setting up robot models in Unity for digital twin applications. You'll learn how to bring 3D robot models into Unity, configure them for visualization, and prepare them for integration with robotics systems.

## Overview

Importing and properly configuring robot models is a critical step in creating effective digital twin visualizations. This tutorial covers the complete process from model preparation to Unity setup, ensuring optimal performance and visual quality.

## Prerequisites

Before starting this tutorial, you should have:
- Unity Editor installed with robotics packages
- 3D robot model files (FBX, OBJ, or similar format)
- Basic understanding of Unity interface
- Knowledge of robot kinematics (helpful but not required)

## Robot Model Preparation

### Model Format Requirements

Unity supports several 3D model formats:
- **FBX** (Recommended): Industry standard, supports animations and materials
- **OBJ**: Simple format, good for static models
- **DAE**: Collada format, supports complex scenes
- **3DS**: Legacy format, basic support

### Model Optimization Guidelines

Before importing, optimize your robot model:

1. **Polygon Count**: Keep under 50,000 triangles for real-time performance
2. **Texture Resolution**: Use 1024x1024 or 2048x2048 max for most parts
3. **UV Mapping**: Ensure proper UV coordinates for texture application
4. **Scale**: Model should be in meters for proper physics integration
5. **Hierarchy**: Organize parts logically (base, arms, sensors, etc.)

### Example Robot Model Structure

For a typical wheeled robot:
```
Robot_Base
├── Chassis
├── Wheel_Left
├── Wheel_Right
├── Sensor_Camera
├── Sensor_LiDAR
└── Manipulator_Arm (if applicable)
```

## Importing Models into Unity

### Step 1: Prepare Your Project

1. Create a new Unity project or open an existing one
2. Create a folder structure:
   - `Assets/Models/Robots/`
   - `Assets/Materials/Robots/`
   - `Assets/Prefabs/Robots/`

### Step 2: Import Robot Model

1. Place your robot model file in the `Assets/Models/Robots/` folder
2. Unity will automatically import and process the model
3. Select the model in the Project window to view import settings

### Step 3: Configure Import Settings

In the Import Settings panel:

**Model Tab:**
- **Scale Factor**: Set to 1 (if model is already in meters)
- **Mesh Compression**: Medium (balance quality/performance)
- **Read/Write enabled**: Check if you need runtime mesh manipulation
- **Optimize Mesh**: Enable for better performance
- **Import Visibility**: Enable to preserve visibility settings
- **Import Cameras**: Enable if model includes cameras
- **Import Lights**: Enable if model includes lights

**Rig Tab:**
- **Animation Type**: Select based on your needs:
  - None: For static models
  - Legacy: For simple animations
  - Generic: For humanoid or generic rigs
  - Humanoid: For humanoid robots

**Animation Tab:**
- **Import Animation**: Enable if model has animations
- **Generate Animations**: Select "In Root Motion" or "None" as appropriate

## Setting Up Robot Components

### Basic Robot Setup

Create a basic robot prefab with essential components:

```csharp
using UnityEngine;

public class RobotModelSetup : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "DigitalTwinRobot";
    public float robotScale = 1.0f;

    [Header("Visual Components")]
    public Transform chassis;
    public Transform[] wheels;
    public Transform[] sensors;

    [Header("Physical Properties")]
    public float mass = 10.0f;
    public Vector3 centerOfMass = Vector3.zero;

    void Start()
    {
        SetupRobotComponents();
    }

    void SetupRobotComponents()
    {
        // Configure rigidbody for physics (optional for visualization only)
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }

        rb.mass = mass;
        rb.centerOfMass = centerOfMass;
        rb.isKinematic = true; // For visualization, disable physics simulation

        // Configure wheels if they exist
        ConfigureWheels();

        // Setup sensors
        ConfigureSensors();
    }

    void ConfigureWheels()
    {
        if (wheels == null) return;

        foreach (Transform wheel in wheels)
        {
            if (wheel != null)
            {
                // Add visual effects or markers for wheels
                AddWheelVisuals(wheel);
            }
        }
    }

    void ConfigureSensors()
    {
        if (sensors == null) return;

        foreach (Transform sensor in sensors)
        {
            if (sensor != null)
            {
                // Add sensor visual indicators
                AddSensorVisuals(sensor);
            }
        }
    }

    void AddWheelVisuals(Transform wheel)
    {
        // Add rotation indicators or visual effects
        // This could be particle systems, trails, or simple markers
    }

    void AddSensorVisuals(Transform sensor)
    {
        // Add visual indicators for sensors (camera frustums, LiDAR beams, etc.)
        // This helps visualize sensor coverage in the digital twin
    }
}
```

### Hierarchical Setup for Articulated Robots

For robots with multiple joints and degrees of freedom:

```csharp
using UnityEngine;

public class ArticulatedRobotSetup : MonoBehaviour
{
    [System.Serializable]
    public class JointInfo
    {
        public string jointName;
        public Transform jointTransform;
        public JointType jointType;
        public float minAngle = -90f;
        public float maxAngle = 90f;
        public float currentAngle = 0f;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    public JointInfo[] joints;

    [Header("Kinematics")]
    public Transform baseLink;
    public Transform endEffector;

    void Start()
    {
        InitializeJoints();
    }

    void InitializeJoints()
    {
        foreach (JointInfo joint in joints)
        {
            if (joint.jointTransform != null)
            {
                // Store initial position/rotation for reference
                joint.currentAngle = GetJointAngle(joint);
            }
        }
    }

    float GetJointAngle(JointInfo joint)
    {
        // Calculate current joint angle based on joint type
        switch (joint.jointType)
        {
            case JointType.Revolute:
                // For revolute joints, extract rotation angle
                return joint.jointTransform.localEulerAngles.z;
            case JointType.Prismatic:
                // For prismatic joints, extract position
                return joint.jointTransform.localPosition.x;
            default:
                return 0f;
        }
    }

    public void SetJointAngle(int jointIndex, float angle)
    {
        if (jointIndex < 0 || jointIndex >= joints.Length)
            return;

        JointInfo joint = joints[jointIndex];
        angle = Mathf.Clamp(angle, joint.minAngle, joint.maxAngle);

        switch (joint.jointType)
        {
            case JointType.Revolute:
                joint.jointTransform.localRotation =
                    Quaternion.Euler(0, 0, angle);
                break;
            case JointType.Prismatic:
                Vector3 pos = joint.jointTransform.localPosition;
                pos.x = angle;
                joint.jointTransform.localPosition = pos;
                break;
        }

        joint.currentAngle = angle;
    }
}
```

## Material and Texture Setup

### Basic Material Configuration

Apply appropriate materials to robot parts:

1. **Select the robot model** in the Hierarchy or Scene view
2. **In the Inspector**, expand the mesh components
3. **Select each material** and assign appropriate textures
4. **Adjust material properties** for realistic appearance

### Recommended Material Settings

For robot visualization:

```csharp
using UnityEngine;

public class RobotMaterialSetup : MonoBehaviour
{
    [Header("Material Presets")]
    public Material chassisMaterial;
    public Material wheelMaterial;
    public Material sensorMaterial;
    public Material cableMaterial;

    [Header("Visual Effects")]
    public bool enableSpecular = true;
    public float specularIntensity = 0.5f;
    public bool enableEmission = false;
    public Color emissionColor = Color.black;

    void Start()
    {
        ApplyRobotMaterials();
    }

    void ApplyRobotMaterials()
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>();

        foreach (Renderer renderer in renderers)
        {
            if (renderer.name.Contains("Chassis") || renderer.name.Contains("Base"))
            {
                renderer.material = CreateOrAssignMaterial(chassisMaterial, "Chassis");
            }
            else if (renderer.name.Contains("Wheel"))
            {
                renderer.material = CreateOrAssignMaterial(wheelMaterial, "Wheel");
            }
            else if (renderer.name.Contains("Sensor") || renderer.name.Contains("Camera") || renderer.name.Contains("LiDAR"))
            {
                renderer.material = CreateOrAssignMaterial(sensorMaterial, "Sensor");
            }
            else
            {
                // Apply default material or create based on naming convention
                renderer.material = CreateDefaultMaterial(renderer.name);
            }
        }
    }

    Material CreateOrAssignMaterial(Material preset, string name)
    {
        if (preset != null)
        {
            Material newMat = new Material(preset);
            newMat.name = name + "_Material";
            return newMat;
        }

        return CreateDefaultMaterial(name);
    }

    Material CreateDefaultMaterial(string name)
    {
        Material material = new Material(Shader.Find("Standard"));
        material.name = name + "_Material";

        // Set default robot-like colors
        if (name.ToLower().Contains("chassis") || name.ToLower().Contains("base"))
        {
            material.color = new Color(0.7f, 0.7f, 0.7f); // Light gray
        }
        else if (name.ToLower().Contains("wheel"))
        {
            material.color = new Color(0.2f, 0.2f, 0.2f); // Dark gray
        }
        else if (name.ToLower().Contains("sensor"))
        {
            material.color = new Color(0.1f, 0.1f, 0.8f); // Blue for sensors
            material.SetColor("_EmissionColor", new Color(0.2f, 0.2f, 1.0f));
            material.EnableKeyword("_EMISSION");
        }
        else
        {
            material.color = Color.white;
        }

        // Configure standard material properties
        material.SetFloat("_Metallic", 0.5f);
        material.SetFloat("_Smoothness", 0.5f);

        return material;
    }
}
```

## Animation and Joint Configuration

### Setting Up Robot Animations

For articulated robots, configure joint animations:

1. **Create Animation Controller**:
   - Right-click in Project window → Create → Animator Controller
   - Name it "RobotController"

2. **Configure Animation Parameters**:
   - Open Animator window (Window → Animation → Animator)
   - Add parameters for joint angles (floats)

3. **Create Animation Clips**:
   - For simple joint movements, create basic animation clips
   - For complex movements, use code-based animation

### Example Animation Script

```csharp
using UnityEngine;

public class RobotAnimationController : MonoBehaviour
{
    [Header("Animation Configuration")]
    public AnimationCurve wheelRotationCurve;
    public float animationSpeed = 1.0f;

    [Header("Joint Animation")]
    public Transform[] animatedJoints;
    public float[] jointSpeeds;
    public float[] jointOffsets;

    private float animationTime = 0f;

    void Start()
    {
        if (animatedJoints.Length != jointSpeeds.Length ||
            animatedJoints.Length != jointOffsets.Length)
        {
            Debug.LogError("Joint arrays must have the same length");
            enabled = false;
            return;
        }
    }

    void Update()
    {
        animationTime += Time.deltaTime * animationSpeed;
        AnimateJoints();
    }

    void AnimateJoints()
    {
        for (int i = 0; i < animatedJoints.Length; i++)
        {
            if (animatedJoints[i] != null)
            {
                float angle = Mathf.Sin(animationTime * jointSpeeds[i] + jointOffsets[i]) * 30f;
                animatedJoints[i].localRotation = Quaternion.Euler(0, 0, angle);
            }
        }
    }

    public void SetJointAnimationSpeed(int jointIndex, float speed)
    {
        if (jointIndex >= 0 && jointIndex < jointSpeeds.Length)
        {
            jointSpeeds[jointIndex] = speed;
        }
    }
}
```

## Performance Optimization

### Level of Detail (LOD) Setup

For complex robot models, implement LOD:

```csharp
using UnityEngine;

public class RobotLODSetup : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public float screenRelativeTransitionHeight = 0.2f;
        public Renderer[] renderers;
        public float renderersSize = 1.0f;
    }

    public LODLevel[] lodLevels;
    private LODGroup lodGroup;

    void Start()
    {
        SetupLOD();
    }

    void SetupLOD()
    {
        lodGroup = GetComponent<LODGroup>();
        if (lodGroup == null)
        {
            lodGroup = gameObject.AddComponent<LODGroup>();
        }

        LOD[] lods = new LOD[lodLevels.Length];

        for (int i = 0; i < lodLevels.Length; i++)
        {
            lods[i] = new LOD(lodLevels[i].screenRelativeTransitionHeight,
                             lodLevels[i].renderers);
        }

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }
}
```

### Occlusion Culling Setup

For large scenes with multiple robots:

1. **Mark Static Objects**: Select robot parts that don't move and check "Static" in Inspector
2. **Configure Occlusion Area**: Use Occlusion Area components for complex scenes
3. **Bake Occlusion Culling**: Window → Rendering → Occlusion Culling → Bake

## Integration with Robotics Data

### Real-time Robot State Update

```csharp
using UnityEngine;

public class RobotStateVisualizer : MonoBehaviour
{
    [Header("ROS Integration")]
    public string robotNamespace = "/my_robot";
    public float updateRate = 30.0f; // Hz

    [Header("Visual State")]
    public Transform robotTransform;
    public ArticulatedRobotSetup robotJoints;
    public Renderer[] statusIndicators;

    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        updateInterval = 1.0f / updateRate;
        robotTransform = transform;

        // Initialize status indicators
        InitializeStatusIndicators();
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            UpdateRobotVisualization();
            lastUpdateTime = Time.time;
        }
    }

    void UpdateRobotVisualization()
    {
        // This would typically receive data from ROS/other systems
        // For now, we'll simulate with example values

        // Update position and orientation
        UpdateRobotPosition();

        // Update joint angles
        UpdateJointAngles();

        // Update status indicators
        UpdateStatusIndicators();
    }

    void UpdateRobotPosition()
    {
        // Example: Move robot in a circle
        float time = Time.time;
        float radius = 2.0f;
        float x = Mathf.Cos(time * 0.5f) * radius;
        float z = Mathf.Sin(time * 0.5f) * radius;

        robotTransform.position = new Vector3(x, 0.5f, z);
        robotTransform.rotation = Quaternion.LookRotation(new Vector3(
            Mathf.Cos(time * 0.5f + Mathf.PI/2), 0, Mathf.Sin(time * 0.5f + Mathf.PI/2)));
    }

    void UpdateJointAngles()
    {
        if (robotJoints != null && robotJoints.joints != null)
        {
            // Example: Animate joints with time-based values
            for (int i = 0; i < robotJoints.joints.Length; i++)
            {
                float angle = Mathf.Sin(Time.time + i) * 30f;
                robotJoints.SetJointAngle(i, angle);
            }
        }
    }

    void UpdateStatusIndicators()
    {
        foreach (Renderer indicator in statusIndicators)
        {
            if (indicator != null)
            {
                // Change color based on status (green = active, red = error, etc.)
                float intensity = Mathf.PingPong(Time.time, 1.0f);
                indicator.material.color = Color.Lerp(Color.red, Color.green, intensity);
            }
        }
    }

    void InitializeStatusIndicators()
    {
        // Find and initialize status indicators
        statusIndicators = GetComponentsInChildren<Renderer>();
    }
}
```

## Troubleshooting Common Issues

### Issue: Model appears too large or small
**Solution**:
- Check the model's scale in 3D modeling software (should be in meters)
- Adjust the Scale Factor in Unity's import settings
- Use a reference object (1-meter cube) to verify scale

### Issue: Textures don't appear correctly
**Solutions**:
- Verify texture files are in the same folder as the model
- Check that textures are in supported formats (PNG, JPG, TGA)
- Ensure texture paths are correct in the material settings

### Issue: Joints don't animate properly
**Solutions**:
- Verify the hierarchy structure matches the intended kinematics
- Check that joint transforms are properly configured
- Ensure rotation axes are correct for each joint

### Issue: Performance problems with complex models
**Solutions**:
- Reduce polygon count in 3D modeling software
- Implement LOD system for distant robots
- Use occlusion culling for robots outside view
- Optimize material properties and shaders

## Best Practices

### 1. Modular Design
- Create separate prefabs for different robot components
- Use composition rather than monolithic models
- Maintain consistent naming conventions

### 2. Scalability
- Design models that can be easily modified for different robot types
- Use parameters and variables instead of hardcoded values
- Create reusable components and scripts

### 3. Documentation
- Include README files with model specifications
- Document material and texture requirements
- Provide usage examples and integration guides

## Next Steps

With your robot model properly imported and configured, you're ready to move on to setting up realistic materials and lighting for your robot. The next section will cover techniques for creating photorealistic rendering of robot models in Unity.
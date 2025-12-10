---
title: Exercise 2 - Unity Robot Visualization
sidebar_position: 6
---

# Exercise 2 - Unity Robot Visualization

## Learning Objective
Create a complete Unity scene with realistic robot visualization, including proper materials, lighting, and user interface controls that demonstrate high-fidelity rendering for digital twin applications.

## Estimated Time
90-120 minutes

## Prerequisites
- Unity installation completed
- Understanding of robot model import and setup
- Knowledge of materials and lighting principles
- Basic understanding of UI systems in Unity

## Materials Needed
- 3D robot model file (FBX/OBJ format)
- Textures for robot materials (optional)
- Unity Editor with robotics packages installed
- Basic C# scripting knowledge

---

### Part 1: Scene Setup and Robot Import
**Objective**: Create a Unity scene and import a robot model with proper configuration.

**Steps**:
1. Create a new Unity 3D project named "DigitalTwinVisualization"
2. Import your robot model into the project:
   - Create `Assets/Models/Robots/` folder
   - Place robot model file in this folder
   - Configure import settings:
     - Scale Factor: 1 (if model is in meters)
     - Mesh Compression: Medium
     - Read/Write enabled: Checked
     - Optimize Mesh: Checked

3. Create the robot prefab:
   - Drag the imported model into the scene
   - Create a new empty GameObject named "Robot"
   - Make the imported model a child of the Robot GameObject
   - Add a RobotModelSetup component to the Robot GameObject
   - Save as prefab in `Assets/Prefabs/Robots/`

4. Set up the basic scene:
   - Create a ground plane (10x10 units)
   - Add a basic environment with simple obstacles
   - Position the robot at coordinates (0, 0.5, 0)

**Verification**: The robot model should appear in the scene with proper scale and positioning.

---

### Part 2: Material and Lighting Configuration
**Objective**: Apply realistic materials to the robot and configure proper lighting for visualization.

**Steps**:
1. Create robot-specific materials:
   - Create `Assets/Materials/Robots/` folder
   - Create materials for different robot components:
     - Chassis material (metallic, moderate smoothness)
     - Wheel material (rubber-like, low smoothness)
     - Sensor material (slightly emissive)

2. Apply materials to robot components:
   - Select the robot in the scene
   - In the Inspector, assign appropriate materials to each mesh renderer
   - Adjust material properties:
     - Chassis: Metallic = 0.6, Smoothness = 0.4
     - Wheels: Metallic = 0.1, Smoothness = 0.2
     - Sensors: Emission enabled with blue color

3. Configure scene lighting:
   - Create a directional light for main illumination
   - Set rotation to (50, -30, 0)
   - Set intensity to 1.0
   - Add ambient lighting: Window → Rendering → Lighting Settings → Environment Lighting → Intensity = 0.5

4. Add additional lights for better visualization:
   - Create fill lights from different angles
   - Add spotlights to highlight specific robot features
   - Configure light layers to avoid performance issues

**Verification**: The robot should appear with realistic materials and proper lighting that highlights its features.

---

### Part 3: User Interface Implementation
**Objective**: Create a comprehensive UI system for robot monitoring and control.

**Steps**:
1. Create the UI canvas:
   - Create a new UI Canvas in the scene
   - Set Render Mode to "Screen Space - Overlay"
   - Add Canvas Scaler component with Scale With Screen Size

2. Create the status panel:
   - Create a Panel UI element
   - Add Text elements for:
     - Robot ID
     - Position coordinates
     - Battery level
     - Connection status
   - Create Image elements for status indicators

3. Create the control panel:
   - Create a separate Panel for controls
   - Add buttons for basic movement:
     - Move Forward
     - Move Backward
     - Turn Left
     - Turn Right
     - Stop
   - Add a slider for speed control
   - Add an emergency stop button

4. Create the sensor visualization panel:
   - Add text elements for sensor data
   - Create gauges for battery, CPU, and memory usage
   - Add placeholder for camera feed display

5. Write the UI manager script:
   ```csharp
   using UnityEngine;
   using UnityEngine.UI;

   public class ExerciseUIManager : MonoBehaviour
   {
       [Header("Status Panel")]
       public Text robotIdText;
       public Text positionText;
       public Text batteryText;
       public Image batteryIndicator;

       [Header("Control Panel")]
       public Button moveForwardButton;
       public Button moveBackwardButton;
       public Button turnLeftButton;
       public Button turnRightButton;
       public Button stopButton;
       public Button emergencyStopButton;
       public Slider speedSlider;

       [Header("Sensor Panel")]
       public Text sensorDataText;
       public Image cpuGauge;
       public Image memoryGauge;

       private Vector3 robotPosition = Vector3.zero;
       private float batteryLevel = 100f;
       private float speed = 0.5f;

       void Start()
       {
           SetupControlEvents();
           UpdateUIDisplay();
       }

       void SetupControlEvents()
       {
           if (moveForwardButton != null)
               moveForwardButton.onClick.AddListener(() => MoveRobot(Vector3.forward));

           if (moveBackwardButton != null)
               moveBackwardButton.onClick.AddListener(() => MoveRobot(Vector3.back));

           if (turnLeftButton != null)
               turnLeftButton.onClick.AddListener(() => TurnRobot(-1));

           if (turnRightButton != null)
               turnRightButton.onClick.AddListener(() => TurnRobot(1));

           if (stopButton != null)
               stopButton.onClick.AddListener(StopRobot);

           if (emergencyStopButton != null)
               emergencyStopButton.onClick.AddListener(EmergencyStop);

           if (speedSlider != null)
               speedSlider.onValueChanged.AddListener(value => speed = value);
       }

       void MoveRobot(Vector3 direction)
       {
           robotPosition += direction * speed * Time.deltaTime * 10f;
           UpdateUIDisplay();
       }

       void TurnRobot(int direction)
       {
           // In a real implementation, this would rotate the robot
           UpdateUIDisplay();
       }

       void StopRobot()
       {
           // Stop robot movement
       }

       void EmergencyStop()
       {
           batteryLevel -= 10f; // Simulate emergency drain
           UpdateUIDisplay();
       }

       void UpdateUIDisplay()
       {
           if (robotIdText != null)
               robotIdText.text = "Robot: DT-001";

           if (positionText != null)
               positionText.text = $"Position: ({robotPosition.x:F2}, {robotPosition.y:F2}, {robotPosition.z:F2})";

           if (batteryText != null)
               batteryText.text = $"Battery: {batteryLevel:F1}%";

           if (batteryIndicator != null)
               batteryIndicator.fillAmount = batteryLevel / 100f;

           if (sensorDataText != null)
               sensorDataText.text = $"Sensors: Active - Laser: 360pts, IMU: OK";

           if (cpuGauge != null)
               cpuGauge.fillAmount = Random.Range(0.2f, 0.6f);

           if (memoryGauge != null)
               memoryGauge.fillAmount = Random.Range(0.3f, 0.7f);
       }

       void Update()
       {
           // Simulate battery drain over time
           if (batteryLevel > 0)
               batteryLevel -= 0.01f * Time.deltaTime;

           UpdateUIDisplay();
       }
   }
   ```

6. Attach the script to the Canvas and assign UI elements in the Inspector.

**Verification**: The UI should display robot status, respond to control inputs, and update in real-time.

---

### Part 4: Animation and Interaction
**Objective**: Add basic animations and interaction capabilities to enhance visualization.

**Steps**:
1. Create a robot animation controller:
   - Create an Animator Controller asset
   - Create basic animation states for idle, moving, and turning
   - Configure transitions between states

2. Add wheel rotation animation:
   ```csharp
   using UnityEngine;

   public class WheelAnimation : MonoBehaviour
   {
       public Transform[] wheels;
       public float rotationSpeed = 10f;

       void Update()
       {
           foreach (Transform wheel in wheels)
           {
               if (wheel != null)
               {
                   wheel.Rotate(Vector3.right, rotationSpeed * Time.deltaTime * 100f);
               }
           }
       }
   }
   ```

3. Add sensor visualization:
   - Create a script to visualize sensor data
   - Add particle systems for LiDAR points
   - Create line renderers for sensor beams

4. Implement camera controls:
   - Add multiple camera views (top-down, side, first-person)
   - Create camera switching functionality
   - Add smooth camera transitions

5. Add interaction features:
   - Raycast-based selection of robot parts
   - Context menus for selected components
   - Detailed information panels

**Verification**: The robot should have animated wheels, visualized sensors, multiple camera views, and interactive elements.

---

### Part 5: Optimization and Polish
**Objective**: Optimize the scene for performance and add final polish.

**Steps**:
1. Implement Level of Detail (LOD):
   - Create simplified versions of the robot model
   - Configure LOD groups for distant viewing
   - Test performance at different distances

2. Optimize materials:
   - Use texture atlasing where possible
   - Implement shader variants to reduce overhead
   - Configure batching settings for multiple robots

3. Add post-processing effects:
   - Create a post-process volume
   - Add ambient occlusion for depth
   - Add bloom for light sources
   - Add color grading for visual appeal

4. Performance testing:
   - Monitor frame rate with Unity Profiler
   - Test on target hardware specifications
   - Optimize materials and lighting as needed

5. Add final touches:
   - Create a proper scene hierarchy
   - Add scene loading/unloading scripts
   - Implement save/load functionality for robot states
   - Add audio feedback for interactions

**Verification**: The scene should run smoothly (30+ FPS) with high visual quality and responsive interactions.

---

### Troubleshooting
**Common Issues**:
- **Issue**: Robot appears too large or small
  - **Solution**: Check model import scale settings and ensure model is in meters

- **Issue**: Materials don't appear correctly
  - **Solution**: Verify texture paths and material shader settings

- **Issue**: UI elements don't respond to input
  - **Solution**: Check for EventSystem in scene and proper Canvas configuration

- **Issue**: Poor performance
  - **Solution**: Reduce polygon count, optimize materials, implement LOD

**Helpful Commands**:
- Use Unity Profiler to identify performance bottlenecks
- Check Scene view to verify object positioning
- Use Console to debug script errors

---

### Solution and Discussion
**Expected Outcome**: A fully functional Unity scene with a robot that can be visually monitored and controlled through an intuitive interface, demonstrating high-fidelity visualization techniques for digital twin applications.

**Key Concepts Learned**:
- Robot model import and configuration in Unity
- Realistic material and lighting setup
- User interface design for robotics applications
- Performance optimization for real-time rendering
- Integration of visualization and control systems

**Extensions**:
- Add more sophisticated sensor visualization
- Implement multiple robots in the same scene
- Connect to real ROS systems for live data
- Add VR/AR support for immersive visualization

---

### Assessment Questions
1. How does the material configuration affect the realism of robot visualization?
2. What are the key performance considerations when rendering complex robot models in real-time?
3. How would you modify the UI to support multiple robots simultaneously?
4. What techniques would you use to visualize different types of sensor data effectively?
5. How can you ensure the UI remains usable across different screen sizes and resolutions?
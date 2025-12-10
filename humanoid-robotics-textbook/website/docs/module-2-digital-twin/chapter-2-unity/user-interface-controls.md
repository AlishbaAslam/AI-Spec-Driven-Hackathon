---
title: User Interface Controls for Robot Interaction
sidebar_position: 5
---

# User Interface Controls for Robot Interaction

This section provides comprehensive guidance on creating intuitive user interfaces for interacting with robot digital twins in Unity. Effective UI design is crucial for operators to monitor, control, and understand robot behavior in the digital twin environment.

## Overview

User interfaces in digital twin applications serve multiple purposes:
- **Monitoring**: Display robot status, sensor data, and operational metrics
- **Control**: Provide mechanisms for commanding robot actions
- **Visualization**: Present complex data in an understandable format
- **Interaction**: Enable direct manipulation of robot parameters

## UI Architecture for Robotics

### Core UI Components

The typical robot UI system includes:

1. **Status Panel**: Real-time robot state information
2. **Control Panel**: Command and control interface
3. **Sensor Visualization**: Display of sensor data and perception results
4. **Navigation Interface**: Path planning and movement controls
5. **Debug/Development Tools**: Advanced monitoring and debugging features

### UI Design Principles

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class RoboticsUIManager : MonoBehaviour
{
    [Header("UI Panels")]
    public GameObject statusPanel;
    public GameObject controlPanel;
    public GameObject sensorPanel;
    public GameObject navigationPanel;

    [Header("UI Elements")]
    public Text robotNameText;
    public Text statusText;
    public Text batteryText;
    public Slider speedSlider;
    public Button[] controlButtons;

    [Header("Color Schemes")]
    public Color activeColor = Color.green;
    public Color warningColor = Color.yellow;
    public Color errorColor = Color.red;
    public Color normalColor = Color.white;

    [Header("UI Configuration")]
    public bool showAdvancedControls = false;
    public float uiScale = 1.0f;

    private Dictionary<string, GameObject> uiPanels;
    private bool uiVisible = true;

    void Start()
    {
        InitializeUI();
        ConfigureUIColors();
    }

    void InitializeUI()
    {
        uiPanels = new Dictionary<string, GameObject>
        {
            {"Status", statusPanel},
            {"Control", controlPanel},
            {"Sensor", sensorPanel},
            {"Navigation", navigationPanel}
        };

        // Set initial visibility
        SetPanelVisibility("Status", true);
        SetPanelVisibility("Control", true);
        SetPanelVisibility("Sensor", false);
        SetPanelVisibility("Navigation", false);
    }

    void ConfigureUIColors()
    {
        // Apply consistent color scheme across UI elements
        if (robotNameText != null)
            robotNameText.color = activeColor;

        if (statusText != null)
            statusText.color = normalColor;

        if (batteryText != null)
            batteryText.color = normalColor;

        if (speedSlider != null)
        {
            speedSlider.colors.normalColor = normalColor;
            speedSlider.colors.highlightedColor = activeColor;
        }
    }

    public void SetPanelVisibility(string panelName, bool visible)
    {
        if (uiPanels.ContainsKey(panelName) && uiPanels[panelName] != null)
        {
            uiPanels[panelName].SetActive(visible);
        }
    }

    public void TogglePanel(string panelName)
    {
        if (uiPanels.ContainsKey(panelName) && uiPanels[panelName] != null)
        {
            uiPanels[panelName].SetActive(!uiPanels[panelName].activeSelf);
        }
    }

    public void SetUIVisibility(bool visible)
    {
        uiVisible = visible;
        foreach (GameObject panel in uiPanels.Values)
        {
            if (panel != null)
                panel.SetActive(visible);
        }
    }

    public void SetRobotStatus(string status, bool isError = false, bool isWarning = false)
    {
        if (statusText != null)
        {
            statusText.text = status;

            if (isError)
                statusText.color = errorColor;
            else if (isWarning)
                statusText.color = warningColor;
            else
                statusText.color = activeColor;
        }
    }

    public void SetBatteryLevel(float level)
    {
        if (batteryText != null)
        {
            batteryText.text = $"Battery: {level:F1}%";
            batteryText.color = GetBatteryColor(level);
        }
    }

    Color GetBatteryColor(float level)
    {
        if (level > 50) return activeColor;
        if (level > 20) return warningColor;
        return errorColor;
    }
}
```

## Status Panel Implementation

### Real-time Robot Status Display

```csharp
using UnityEngine;
using UnityEngine.UI;
using System;

public class RobotStatusPanel : MonoBehaviour
{
    [Header("Status Text Fields")]
    public Text robotIdText;
    public Text positionText;
    public Text orientationText;
    public Text velocityText;
    public Text batteryText;
    public Text cpuUsageText;
    public Text memoryUsageText;
    public Text connectionStatusText;

    [Header("Status Indicators")]
    public Image batteryIndicator;
    public Image cpuIndicator;
    public Image connectionIndicator;
    public Image operationalIndicator;

    [Header("Color Configuration")]
    public Color connectedColor = Color.green;
    public Color disconnectedColor = Color.red;
    public Color operationalColor = Color.green;
    public Color errorColor = Color.red;

    [Header("Update Configuration")]
    public float updateInterval = 0.5f;

    private float lastUpdateTime = 0f;
    private bool isOperational = true;
    private bool isConnected = true;

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            UpdateStatusDisplay();
            lastUpdateTime = Time.time;
        }
    }

    void UpdateStatusDisplay()
    {
        // Update robot identification
        if (robotIdText != null)
            robotIdText.text = $"Robot: {GetRobotId()}";

        // Update position and orientation
        if (positionText != null)
            positionText.text = $"Position: {GetRobotPosition():F2}";

        if (orientationText != null)
            orientationText.text = $"Orientation: {GetRobotOrientation():F2}";

        // Update velocity
        if (velocityText != null)
            velocityText.text = $"Velocity: {GetRobotVelocity():F2} m/s";

        // Update battery
        float batteryLevel = GetBatteryLevel();
        if (batteryText != null)
            batteryText.text = $"Battery: {batteryLevel:F1}%";

        // Update system resources
        if (cpuUsageText != null)
            cpuUsageText.text = $"CPU: {GetCpuUsage():F1}%";

        if (memoryUsageText != null)
            memoryUsageText.text = $"Memory: {GetMemoryUsage():F1}%";

        // Update connection status
        if (connectionStatusText != null)
            connectionStatusText.text = isConnected ? "Connected" : "Disconnected";

        // Update indicators
        UpdateIndicators(batteryLevel);
    }

    void UpdateIndicators(float batteryLevel)
    {
        // Battery indicator
        if (batteryIndicator != null)
        {
            batteryIndicator.fillAmount = batteryLevel / 100f;
            batteryIndicator.color = GetBatteryColor(batteryLevel);
        }

        // CPU indicator
        if (cpuIndicator != null)
        {
            float cpuUsage = GetCpuUsage();
            cpuIndicator.fillAmount = cpuUsage / 100f;
            cpuIndicator.color = GetResourceColor(cpuUsage);
        }

        // Connection indicator
        if (connectionIndicator != null)
        {
            connectionIndicator.color = isConnected ? connectedColor : disconnectedColor;
        }

        // Operational indicator
        if (operationalIndicator != null)
        {
            operationalIndicator.color = isOperational ? operationalColor : errorColor;
        }
    }

    Color GetBatteryColor(float level)
    {
        if (level > 50) return Color.green;
        if (level > 20) return Color.yellow;
        return Color.red;
    }

    Color GetResourceColor(float usage)
    {
        if (usage < 70) return Color.green;
        if (usage < 90) return Color.yellow;
        return Color.red;
    }

    // Placeholder methods - would connect to actual robot data
    string GetRobotId() { return "Robot_001"; }
    Vector3 GetRobotPosition() { return Vector3.zero; }
    float GetRobotOrientation() { return 0f; }
    float GetRobotVelocity() { return 0f; }
    float GetBatteryLevel() { return 85f; }
    float GetCpuUsage() { return 25f; }
    float GetMemoryUsage() { return 45f; }
}
```

## Control Panel Implementation

### Robot Command Interface

```csharp
using UnityEngine;
using UnityEngine.UI;

public class RobotControlPanel : MonoBehaviour
{
    [Header("Movement Controls")]
    public Button moveForwardButton;
    public Button moveBackwardButton;
    public Button turnLeftButton;
    public Button turnRightButton;
    public Button stopButton;
    public Slider speedSlider;
    public Text speedText;

    [Header("Navigation Controls")]
    public Button goToWaypointButton;
    public Button cancelNavigationButton;
    public InputField xPositionField;
    public InputField yPositionField;
    public InputField zPositionField;

    [Header("Action Controls")]
    public Button[] actionButtons;
    public Toggle[] featureToggles;

    [Header("Safety Controls")]
    public Button emergencyStopButton;
    public Toggle enableAutonomousToggle;

    private float currentSpeed = 0.5f;

    void Start()
    {
        SetupControlEvents();
        UpdateSpeedDisplay();
    }

    void SetupControlEvents()
    {
        // Movement controls
        if (moveForwardButton != null)
            moveForwardButton.onClick.AddListener(() => SendMoveCommand(Vector3.forward, currentSpeed));

        if (moveBackwardButton != null)
            moveBackwardButton.onClick.AddListener(() => SendMoveCommand(Vector3.back, currentSpeed));

        if (turnLeftButton != null)
            turnLeftButton.onClick.AddListener(() => SendTurnCommand(-1));

        if (turnRightButton != null)
            turnRightButton.onClick.AddListener(() => SendTurnCommand(1));

        if (stopButton != null)
            stopButton.onClick.AddListener(() => SendStopCommand());

        // Speed control
        if (speedSlider != null)
            speedSlider.onValueChanged.AddListener(OnSpeedChanged);

        // Navigation controls
        if (goToWaypointButton != null)
            goToWaypointButton.onClick.AddListener(SendNavigationCommand);

        if (cancelNavigationButton != null)
            cancelNavigationButton.onClick.AddListener(CancelNavigation);

        // Safety controls
        if (emergencyStopButton != null)
            emergencyStopButton.onClick.AddListener(TriggerEmergencyStop);

        if (enableAutonomousToggle != null)
            enableAutonomousToggle.onValueChanged.AddListener(OnAutonomousModeChanged);
    }

    void OnSpeedChanged(float value)
    {
        currentSpeed = value;
        UpdateSpeedDisplay();
    }

    void UpdateSpeedDisplay()
    {
        if (speedText != null)
            speedText.text = $"Speed: {currentSpeed:F2}x";

        if (speedSlider != null)
            speedSlider.value = currentSpeed;
    }

    public void SendMoveCommand(Vector3 direction, float speed)
    {
        // Send movement command to robot
        Debug.Log($"Moving {direction} at speed {speed}");
        // In real implementation, this would send command via ROS or other protocol
    }

    public void SendTurnCommand(int direction) // -1 for left, 1 for right
    {
        float turnSpeed = direction * currentSpeed;
        Debug.Log($"Turning with speed {turnSpeed}");
    }

    public void SendStopCommand()
    {
        Debug.Log("Stopping robot");
    }

    public void SendNavigationCommand()
    {
        if (float.TryParse(xPositionField.text, out float x) &&
            float.TryParse(yPositionField.text, out float y) &&
            float.TryParse(zPositionField.text, out float z))
        {
            Vector3 targetPosition = new Vector3(x, y, z);
            Debug.Log($"Navigating to position: {targetPosition}");
        }
    }

    public void CancelNavigation()
    {
        Debug.Log("Canceling navigation");
    }

    public void TriggerEmergencyStop()
    {
        Debug.Log("EMERGENCY STOP TRIGGERED!");
        // In real implementation, this would send immediate stop command
    }

    void OnAutonomousModeChanged(bool enabled)
    {
        Debug.Log($"Autonomous mode {(enabled ? "enabled" : "disabled")}");
    }

    public void ExecuteAction(int actionIndex)
    {
        if (actionIndex >= 0 && actionIndex < actionButtons.Length)
        {
            Debug.Log($"Executing action {actionIndex}");
        }
    }
}
```

## Sensor Data Visualization

### Real-time Sensor Display

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class SensorDataVisualization : MonoBehaviour
{
    [Header("Sensor Data Displays")]
    public Text laserScanText;
    public Text cameraFeedText;
    public Text imuDataText;
    public Text gpsDataText;
    public Text jointStatesText;

    [Header("Sensor Visualization")]
    public GameObject laserScanVisual;
    public RawImage cameraFeedImage;
    public Text[] sensorValueDisplays;

    [Header("Gauges and Meters")]
    public Image[] valueGauges;
    public Text[] gaugeLabels;

    [Header("Update Configuration")]
    public float laserScanUpdateRate = 10f; // Hz
    public float cameraUpdateRate = 5f;     // Hz
    public float sensorUpdateRate = 20f;    // Hz

    private Dictionary<string, float[]> sensorBuffers;
    private float lastLaserUpdate = 0f;
    private float lastCameraUpdate = 0f;
    private float lastSensorUpdate = 0f;

    void Start()
    {
        InitializeSensorBuffers();
        SetupSensorVisualization();
    }

    void InitializeSensorBuffers()
    {
        sensorBuffers = new Dictionary<string, float[]>
        {
            {"LaserScan", new float[360]}, // 360 degree scan
            {"IMU_Accel", new float[3]},   // X, Y, Z acceleration
            {"IMU_Gyro", new float[3]},    // X, Y, Z angular velocity
            {"Joint_Positions", new float[10]} // Up to 10 joints
        };
    }

    void SetupSensorVisualization()
    {
        // Initialize laser scan visualization
        if (laserScanVisual != null)
        {
            // Setup 3D visualization for laser scan points
            SetupLaserScanVisual();
        }

        // Initialize camera feed (would typically receive texture from ROS)
        if (cameraFeedImage != null)
        {
            // Camera feed would be updated from ROS sensor data
        }
    }

    void Update()
    {
        UpdateSensorDisplays();
    }

    void UpdateSensorDisplays()
    {
        float currentTime = Time.time;

        // Update laser scan data
        if (currentTime - lastLaserUpdate >= 1f / laserScanUpdateRate)
        {
            UpdateLaserScanDisplay();
            lastLaserUpdate = currentTime;
        }

        // Update camera feed
        if (cameraUpdateRate > 0 && currentTime - lastCameraUpdate >= 1f / cameraUpdateRate)
        {
            UpdateCameraDisplay();
            lastCameraUpdate = currentTime;
        }

        // Update other sensor data
        if (currentTime - lastSensorUpdate >= 1f / sensorUpdateRate)
        {
            UpdateSensorDataDisplay();
            UpdateGauges();
            lastSensorUpdate = currentTime;
        }
    }

    void UpdateLaserScanDisplay()
    {
        if (laserScanText != null)
        {
            // Display summary of laser scan data
            float[] scanData = sensorBuffers["LaserScan"];
            float minDistance = float.MaxValue;
            float maxDistance = 0f;

            for (int i = 0; i < scanData.Length; i++)
            {
                if (scanData[i] < minDistance && scanData[i] > 0) minDistance = scanData[i];
                if (scanData[i] > maxDistance) maxDistance = scanData[i];
            }

            laserScanText.text = $"Laser Scan - Min: {minDistance:F2}m, Max: {maxDistance:F2}m, Points: {scanData.Length}";
        }

        // Update 3D laser scan visualization
        UpdateLaserScanVisual();
    }

    void UpdateCameraDisplay()
    {
        if (cameraFeedText != null)
        {
            cameraFeedText.text = $"Camera Feed: Active ({Time.time:F2}s)";
        }
    }

    void UpdateSensorDataDisplay()
    {
        // Update IMU data
        if (imuDataText != null)
        {
            float[] accelData = sensorBuffers["IMU_Accel"];
            float[] gyroData = sensorBuffers["IMU_Gyro"];
            imuDataText.text = $"IMU - Accel: ({accelData[0]:F2}, {accelData[1]:F2}, {accelData[2]:F2}), " +
                              $"Gyro: ({gyroData[0]:F2}, {gyroData[1]:F2}, {gyroData[2]:F2})";
        }

        // Update GPS data
        if (gpsDataText != null)
        {
            gpsDataText.text = $"GPS: Lat: 0.0000, Lon: 0.0000, Alt: 0.00m"; // Placeholder
        }

        // Update joint states
        if (jointStatesText != null)
        {
            float[] jointData = sensorBuffers["Joint_Positions"];
            string jointStr = "Joints: ";
            for (int i = 0; i < jointData.Length && i < 5; i++) // Show first 5 joints
            {
                jointStr += $"J{i}:{jointData[i]:F2} ";
            }
            jointStatesText.text = jointStr;
        }
    }

    void UpdateGauges()
    {
        if (valueGauges != null && sensorValueDisplays != null)
        {
            // Update gauge fill amounts based on sensor values
            for (int i = 0; i < valueGauges.Length && i < sensorValueDisplays.Length; i++)
            {
                if (valueGauges[i] != null && sensorValueDisplays[i] != null)
                {
                    float value = GetSensorValueForGauge(i);
                    float normalizedValue = Mathf.Clamp01(value / 100f); // Assuming 0-100 range
                    valueGauges[i].fillAmount = normalizedValue;
                    sensorValueDisplays[i].text = $"{value:F1}";
                }
            }
        }
    }

    float GetSensorValueForGauge(int gaugeIndex)
    {
        // Return appropriate sensor value for each gauge
        // This would connect to actual sensor data in real implementation
        switch (gaugeIndex)
        {
            case 0: return Random.Range(0f, 100f); // Battery level
            case 1: return Random.Range(0f, 80f);  // CPU usage
            case 2: return Random.Range(0f, 90f);  // Memory usage
            case 3: return Random.Range(10f, 50f); // Temperature
            default: return 50f;
        }
    }

    void SetupLaserScanVisual()
    {
        // Create visualization for laser scan points
        // This would typically create line renderers or point clouds
    }

    void UpdateLaserScanVisual()
    {
        // Update the 3D visualization of laser scan data
        // In real implementation, this would process actual scan data
    }

    // Method to update sensor data from external source (e.g., ROS)
    public void UpdateLaserScanData(float[] scanData)
    {
        if (scanData != null && scanData.Length <= sensorBuffers["LaserScan"].Length)
        {
            System.Array.Copy(scanData, sensorBuffers["LaserScan"], scanData.Length);
        }
    }

    public void UpdateIMUData(float[] accelData, float[] gyroData)
    {
        if (accelData != null && accelData.Length == 3)
            System.Array.Copy(accelData, sensorBuffers["IMU_Accel"], 3);

        if (gyroData != null && gyroData.Length == 3)
            System.Array.Copy(gyroData, sensorBuffers["IMU_Gyro"], 3);
    }

    public void UpdateJointPositions(float[] jointPositions)
    {
        if (jointPositions != null)
        {
            int copyLength = Mathf.Min(jointPositions.Length, sensorBuffers["Joint_Positions"].Length);
            System.Array.Copy(jointPositions, sensorBuffers["Joint_Positions"], copyLength);
        }
    }
}
```

## Advanced UI Features

### Custom Inspector for UI Configuration

```csharp
#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(RoboticsUIManager))]
public class RoboticsUIManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        RoboticsUIManager manager = (RoboticsUIManager)target;

        // Standard inspector
        DrawDefaultInspector();

        // Custom UI
        GUILayout.Space(10);
        if (GUILayout.Button("Configure UI Panels"))
        {
            ConfigureUIPanels(manager);
        }

        if (GUILayout.Button("Test Status Updates"))
        {
            TestStatusUpdates(manager);
        }

        if (GUILayout.Button("Reset UI"))
        {
            ResetUI(manager);
        }
    }

    void ConfigureUIPanels(RoboticsUIManager manager)
    {
        // Custom configuration logic
        Debug.Log("Configuring UI panels...");
    }

    void TestStatusUpdates(RoboticsUIManager manager)
    {
        // Send test status updates
        manager.SetRobotStatus("Testing UI", false, false);
        manager.SetBatteryLevel(75f);
    }

    void ResetUI(RoboticsUIManager manager)
    {
        // Reset UI to default state
        manager.SetUIVisibility(true);
        manager.SetRobotStatus("Ready", false, false);
        manager.SetBatteryLevel(100f);
    }
}
#endif
```

### Responsive UI for Different Screen Sizes

```csharp
using UnityEngine;
using UnityEngine.UI;

public class ResponsiveRobotUI : MonoBehaviour
{
    [Header("UI Canvas")]
    public Canvas uiCanvas;
    public CanvasScaler canvasScaler;

    [Header("Screen Size Configuration")]
    public float desktopScale = 1.0f;
    public float tabletScale = 0.8f;
    public float mobileScale = 0.6f;

    [Header("UI Layouts")]
    public GameObject desktopLayout;
    public GameObject tabletLayout;
    public GameObject mobileLayout;

    private ScreenSize currentScreenSize = ScreenSize.Unknown;

    public enum ScreenSize
    {
        Unknown,
        Mobile,
        Tablet,
        Desktop
    }

    void Start()
    {
        ConfigureResponsiveUI();
    }

    void ConfigureResponsiveUI()
    {
        currentScreenSize = DetermineScreenSize();

        // Set canvas scaler for different screen sizes
        if (canvasScaler != null)
        {
            float scale = GetScaleForScreenSize(currentScreenSize);
            canvasScaler.scaleFactor = scale;
        }

        // Activate appropriate layout
        ActivateLayoutForScreenSize(currentScreenSize);
    }

    ScreenSize DetermineScreenSize()
    {
        int screenWidth = Screen.width;
        int screenHeight = Screen.height;
        float aspectRatio = (float)screenWidth / screenHeight;

        // Determine based on screen resolution and aspect ratio
        if (screenWidth <= 800 || screenHeight <= 600)
            return ScreenSize.Mobile;
        else if (screenWidth <= 1200 || screenHeight <= 800)
            return ScreenSize.Tablet;
        else
            return ScreenSize.Desktop;
    }

    float GetScaleForScreenSize(ScreenSize size)
    {
        switch (size)
        {
            case ScreenSize.Mobile: return mobileScale;
            case ScreenSize.Tablet: return tabletScale;
            case ScreenSize.Desktop: return desktopScale;
            default: return 1.0f;
        }
    }

    void ActivateLayoutForScreenSize(ScreenSize size)
    {
        if (desktopLayout != null) desktopLayout.SetActive(false);
        if (tabletLayout != null) tabletLayout.SetActive(false);
        if (mobileLayout != null) mobileLayout.SetActive(false);

        switch (size)
        {
            case ScreenSize.Mobile:
                if (mobileLayout != null) mobileLayout.SetActive(true);
                break;
            case ScreenSize.Tablet:
                if (tabletLayout != null) tabletLayout.SetActive(true);
                break;
            case ScreenSize.Desktop:
                if (desktopLayout != null) desktopLayout.SetActive(true);
                break;
        }
    }

    void Update()
    {
        // Check for screen size changes
        ScreenSize newScreenSize = DetermineScreenSize();
        if (newScreenSize != currentScreenSize)
        {
            currentScreenSize = newScreenSize;
            ConfigureResponsiveUI();
        }
    }
}
```

## Integration with Robotics Systems

### ROS Integration for UI Updates

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Nav;

public class ROSRobotUIConnector : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("ROS Topics")]
    public string laserScanTopic = "/robot/laser_scan";
    public string imuTopic = "/robot/imu";
    public string jointStatesTopic = "/robot/joint_states";
    public string batteryTopic = "/robot/battery";

    [Header("UI Components")]
    public SensorDataVisualization sensorDisplay;
    public RobotStatusPanel statusPanel;

    private ROSConnection ros;

    void Start()
    {
        ConnectToROS();
        SubscribeToTopics();
    }

    void ConnectToROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);
    }

    void SubscribeToTopics()
    {
        // Subscribe to laser scan
        ros.Subscribe<LaserScanMsg>(laserScanTopic, UpdateLaserScan);

        // Subscribe to IMU data
        ros.Subscribe<ImuMsg>(imuTopic, UpdateIMUData);

        // Subscribe to joint states
        ros.Subscribe<JointStateMsg>(jointStatesTopic, UpdateJointStates);

        // Subscribe to battery data
        ros.Subscribe<BatteryStateMsg>(batteryTopic, UpdateBatteryStatus);
    }

    void UpdateLaserScan(LaserScanMsg scanMsg)
    {
        if (sensorDisplay != null)
        {
            float[] ranges = new float[scanMsg.ranges.Length];
            for (int i = 0; i < scanMsg.ranges.Length; i++)
            {
                ranges[i] = scanMsg.ranges[i];
            }
            sensorDisplay.UpdateLaserScanData(ranges);
        }
    }

    void UpdateIMUData(ImuMsg imuMsg)
    {
        if (sensorDisplay != null)
        {
            float[] accel = { (float)imuMsg.linear_acceleration.x,
                             (float)imuMsg.linear_acceleration.y,
                             (float)imuMsg.linear_acceleration.z };
            float[] gyro = { (float)imuMsg.angular_velocity.x,
                            (float)imuMsg.angular_velocity.y,
                            (float)imuMsg.angular_velocity.z };
            sensorDisplay.UpdateIMUData(accel, gyro);
        }
    }

    void UpdateJointStates(JointStateMsg jointMsg)
    {
        if (sensorDisplay != null)
        {
            float[] positions = new float[jointMsg.position.Length];
            for (int i = 0; i < jointMsg.position.Length; i++)
            {
                positions[i] = (float)jointMsg.position[i];
            }
            sensorDisplay.UpdateJointPositions(positions);
        }
    }

    void UpdateBatteryStatus(BatteryStateMsg batteryMsg)
    {
        if (statusPanel != null)
        {
            statusPanel.SetBatteryLevel((float)batteryMsg.percentage * 100f);
        }
    }

    // Methods to send commands back to ROS
    public void SendVelocityCommand(float linearX, float angularZ)
    {
        var twistMsg = new TwistMsg();
        twistMsg.linear = new Vector3Msg(linearX, 0, 0);
        twistMsg.angular = new Vector3Msg(0, 0, angularZ);

        ros.Send("cmd_vel", twistMsg);
    }

    public void SendNavigationGoal(float x, float y, float theta)
    {
        // Send navigation goal to ROS
        // Implementation would depend on navigation stack
    }
}
```

## UI Best Practices and Guidelines

### 1. Information Hierarchy

- **Primary Information**: Critical data that requires immediate attention
- **Secondary Information**: Important but not urgent data
- **Tertiary Information**: Reference information for context

### 2. Color Usage

- **Red**: Critical alerts and emergency situations
- **Yellow/Orange**: Warnings and cautionary information
- **Green**: Normal operation and success states
- **Blue**: Information and neutral states
- **Gray**: Disabled or inactive elements

### 3. Layout Principles

- **Group related functions** together
- **Maintain consistent spacing** between elements
- **Use visual hierarchy** to guide attention
- **Ensure adequate touch targets** (minimum 44px for mobile)

### 4. Accessibility Considerations

- **High contrast** between text and background
- **Clear typography** with appropriate sizes
- **Keyboard navigation** support
- **Screen reader** compatibility

## Troubleshooting Common UI Issues

### Issue: UI elements not responding to input
**Solutions**:
- Check if Canvas is properly configured
- Verify EventSystem exists in scene
- Ensure UI elements are not blocked by other objects
- Check raycast settings on UI elements

### Issue: UI appears pixelated or blurry
**Solutions**:
- Check Canvas scaler settings
- Ensure UI textures are high resolution
- Verify reference resolution settings
- Check anti-aliasing settings

### Issue: Performance problems with UI updates
**Solutions**:
- Limit UI update frequency
- Use object pooling for frequently created elements
- Optimize UI rendering with appropriate canvas settings
- Use UI profiler to identify bottlenecks

### Issue: UI layout breaks on different screen sizes
**Solutions**:
- Use appropriate anchor and pivot settings
- Implement responsive design patterns
- Test on multiple screen resolutions
- Use Canvas Scaler with reference resolution

## Next Steps

With user interface controls properly implemented, you're ready to move on to developing hands-on exercises that combine all the concepts learned. The next section will provide a comprehensive exercise that integrates robot visualization, materials, lighting, and user interaction.
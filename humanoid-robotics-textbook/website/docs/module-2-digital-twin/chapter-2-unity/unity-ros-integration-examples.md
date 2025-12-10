---
title: Unity-ROS Integration Code Examples
sidebar_position: 8
---

# Unity-ROS Integration Code Examples

This section provides comprehensive code examples for integrating Unity with ROS (Robot Operating System) for digital twin applications. These examples demonstrate how to establish communication between Unity and ROS systems, enabling real-time data exchange for visualization and control.

## Overview

Unity-ROS integration enables:
- Real-time robot state visualization
- Command and control interfaces
- Sensor data visualization
- Synchronized simulation between Unity and Gazebo
- Digital twin synchronization

## Setting Up ROS Communication

### 1. Installing ROS TCP Connector

First, install the Unity ROS TCP Connector package:

1. In Unity, go to Window → Package Manager
2. Click the + button and select "Add package from git URL"
3. Enter: `com.unity.robotics.ros-tcp-connector`
4. Install the package

### 2. Basic ROS Connection Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public bool autoConnect = true;

    private ROSConnection ros;
    private bool isConnected = false;

    void Start()
    {
        if (autoConnect)
        {
            ConnectToROS();
        }
    }

    public void ConnectToROS()
    {
        try
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.Initialize(rosIPAddress, rosPort);
            isConnected = true;
            Debug.Log($"Connected to ROS at {rosIPAddress}:{rosPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to ROS: {e.Message}");
            isConnected = false;
        }
    }

    public bool IsConnected()
    {
        return isConnected && ros != null;
    }

    public void Disconnect()
    {
        if (ros != null)
        {
            ros.Close();
            isConnected = false;
            Debug.Log("Disconnected from ROS");
        }
    }

    void OnApplicationQuit()
    {
        Disconnect();
    }
}
```

## Message Type Examples

### 1. Twist Messages (Robot Movement)

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotMovementController : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string cmdVelTopic = "/cmd_vel";
    public float linearSpeed = 1.0f;
    public float angularSpeed = 1.0f;

    [Header("Input Configuration")]
    public string horizontalAxis = "Horizontal";
    public string verticalAxis = "Vertical";

    private ROSConnection ros;
    private float lastCommandTime = 0f;
    private float commandInterval = 0.1f; // Send command every 100ms

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
    }

    void Update()
    {
        // Get input from user
        float horizontal = Input.GetAxis(horizontalAxis);
        float vertical = Input.GetAxis(verticalAxis);

        // Send movement commands
        if (Time.time - lastCommandTime >= commandInterval)
        {
            SendVelocityCommand(vertical * linearSpeed, horizontal * angularSpeed);
            lastCommandTime = Time.time;
        }
    }

    void SendVelocityCommand(float linearX, float angularZ)
    {
        if (ros == null) return;

        var twist = new TwistMsg();
        twist.linear = new Vector3Msg(linearX, 0, 0);
        twist.angular = new Vector3Msg(0, 0, angularZ);

        ros.Send(cmdVelTopic, twist);
    }

    // Method to receive movement commands (for visualization)
    public void SetRobotVelocity(Vector3 linear, Vector3 angular)
    {
        // Apply velocity to robot for visualization
        transform.Translate(linear * Time.deltaTime, Space.World);
        transform.Rotate(angular * Time.deltaTime, Space.Self);
    }
}
```

### 2. Sensor Data Reception (Laser Scan)

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class LaserScanVisualizer : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string laserScanTopic = "/scan";
    public float maxRange = 10.0f;
    public int maxPoints = 360;

    [Header("Visualization")]
    public GameObject laserPointPrefab;
    public Color laserColor = Color.red;
    public float pointSize = 0.05f;

    private LineRenderer lineRenderer;
    private List<GameObject> laserPoints;
    private float[] ranges;
    private float[] angles;

    void Start()
    {
        InitializeVisualization();
        SubscribeToLaserScan();
    }

    void InitializeVisualization()
    {
        // Create line renderer for scan visualization
        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = laserColor;
        lineRenderer.endColor = laserColor;
        lineRenderer.startWidth = 0.02f;
        lineRenderer.endWidth = 0.02f;

        // Initialize point list
        laserPoints = new List<GameObject>();
        ranges = new float[maxPoints];
        angles = new float[maxPoints];

        // Pre-calculate angles
        for (int i = 0; i < maxPoints; i++)
        {
            angles[i] = Mathf.Deg2Rad * (i * 360f / maxPoints);
        }
    }

    void SubscribeToLaserScan()
    {
        var ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<LaserScanMsg>(laserScanTopic, OnLaserScanReceived);
    }

    void OnLaserScanReceived(LaserScanMsg scanMsg)
    {
        // Update ranges array
        int copyLength = Mathf.Min(scanMsg.ranges.Length, ranges.Length);
        for (int i = 0; i < copyLength; i++)
        {
            ranges[i] = (float)scanMsg.ranges[i];
        }

        // Update visualization
        UpdateLaserVisualization();
    }

    void UpdateLaserVisualization()
    {
        if (lineRenderer == null) return;

        List<Vector3> points = new List<Vector3>();

        for (int i = 0; i < ranges.Length; i++)
        {
            if (ranges[i] > 0 && ranges[i] < maxRange)
            {
                float x = ranges[i] * Mathf.Cos(angles[i]);
                float y = 0f;
                float z = ranges[i] * Mathf.Sin(angles[i]);

                Vector3 point = new Vector3(x, y, z);
                points.Add(point);
            }
        }

        if (points.Count > 0)
        {
            lineRenderer.positionCount = points.Count;
            lineRenderer.SetPositions(points.ToArray());
        }
    }

    // Alternative visualization using point objects
    public void CreatePointVisualization()
    {
        // Clear existing points
        foreach (GameObject point in laserPoints)
        {
            if (point != null)
                DestroyImmediate(point);
        }
        laserPoints.Clear();

        // Create new points
        for (int i = 0; i < ranges.Length; i++)
        {
            if (ranges[i] > 0 && ranges[i] < maxRange)
            {
                float x = ranges[i] * Mathf.Cos(angles[i]);
                float z = ranges[i] * Mathf.Sin(angles[i]);

                Vector3 position = new Vector3(x, 0.1f, z) + transform.position;

                GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                point.transform.position = position;
                point.transform.localScale = Vector3.one * pointSize;
                point.GetComponent<Renderer>().material.color = laserColor;
                point.GetComponent<Collider>().enabled = false; // Disable physics

                laserPoints.Add(point);
            }
        }
    }
}
```

### 3. IMU Data Processing

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class IMUDataProcessor : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string imuTopic = "/imu";
    public bool visualizeOrientation = true;

    [Header("IMU Data")]
    public Vector3 linearAcceleration;
    public Vector3 angularVelocity;
    public Vector3 orientation;

    [Header("Visualization")]
    public GameObject orientationIndicator;
    public float orientationScale = 0.1f;

    private bool hasData = false;

    void Start()
    {
        SubscribeToIMU();
    }

    void SubscribeToIMU()
    {
        var ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<ImuMsg>(imuTopic, OnIMUReceived);
    }

    void OnIMUReceived(ImuMsg imuMsg)
    {
        // Extract linear acceleration
        linearAcceleration = new Vector3(
            (float)imuMsg.linear_acceleration.x,
            (float)imuMsg.linear_acceleration.y,
            (float)imuMsg.linear_acceleration.z
        );

        // Extract angular velocity
        angularVelocity = new Vector3(
            (float)imuMsg.angular_velocity.x,
            (float)imuMsg.angular_velocity.y,
            (float)imuMsg.angular_velocity.z
        );

        // Extract orientation (convert from quaternion to Euler)
        orientation = QuaternionToEuler(
            (float)imuMsg.orientation.x,
            (float)imuMsg.orientation.y,
            (float)imuMsg.orientation.z,
            (float)imuMsg.orientation.w
        );

        hasData = true;

        // Update visualization
        if (visualizeOrientation && orientationIndicator != null)
        {
            orientationIndicator.transform.rotation = new Quaternion(
                (float)imuMsg.orientation.x,
                (float)imuMsg.orientation.y,
                (float)imuMsg.orientation.z,
                (float)imuMsg.orientation.w
            );
        }
    }

    Vector3 QuaternionToEuler(float x, float y, float z, float w)
    {
        Vector3 euler = new Vector3();

        // Convert quaternion to Euler angles
        float sinr_cosp = 2 * (w * x + y * z);
        float cosr_cosp = 1 - 2 * (x * x + y * y);
        euler.x = Mathf.Atan2(sinr_cosp, cosr_cosp);

        float sinp = 2 * (w * y - z * x);
        if (Mathf.Abs(sinp) >= 1)
            euler.y = Mathf.PI / 2 * Mathf.Sign(sinp);
        else
            euler.y = Mathf.Asin(sinp);

        float siny_cosp = 2 * (w * z + x * y);
        float cosy_cosp = 1 - 2 * (y * y + z * z);
        euler.z = Mathf.Atan2(siny_cosp, cosy_cosp);

        // Convert to degrees
        euler *= Mathf.Rad2Deg;

        return euler;
    }

    void Update()
    {
        if (hasData && visualizeOrientation && orientationIndicator != null)
        {
            // Smoothly update orientation
            orientationIndicator.transform.rotation = Quaternion.Slerp(
                orientationIndicator.transform.rotation,
                Quaternion.Euler(orientation),
                Time.deltaTime * 5f
            );
        }
    }

    // Method to get formatted IMU data for UI display
    public string GetIMUDataString()
    {
        if (!hasData) return "No IMU data";

        return $"Accel: ({linearAcceleration.x:F2}, {linearAcceleration.y:F2}, {linearAcceleration.z:F2})\n" +
               $"Gyro: ({angularVelocity.x:F2}, {angularVelocity.y:F2}, {angularVelocity.z:F2})\n" +
               $"Orient: ({orientation.x:F2}, {orientation.y:F2}, {orientation.z:F2})";
    }
}
```

## Advanced Integration Examples

### 1. Robot State Publisher

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Nav;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using System.Collections.Generic;

public class RobotStatePublisher : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string jointStatesTopic = "/joint_states";
    public string tfTopic = "/tf";
    public float publishRate = 30.0f; // Hz

    [Header("Robot Configuration")]
    public Transform robotBase;
    public List<JointInfo> joints;

    [System.Serializable]
    public class JointInfo
    {
        public string jointName;
        public Transform jointTransform;
        public JointType jointType;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    private float publishInterval;
    private float lastPublishTime;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        publishInterval = 1.0f / publishRate;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishRobotState();
            lastPublishTime = Time.time;
        }
    }

    void PublishRobotState()
    {
        // Publish joint states
        PublishJointStates();

        // Publish transform (TF)
        PublishTransforms();
    }

    void PublishJointStates()
    {
        var jointState = new JointStateMsg();
        jointState.header = new HeaderMsg();
        jointState.header.stamp = new TimeStamp(ROSTCPConnector.ClockType.ROS_TIME);
        jointState.header.frame_id = "base_link";

        List<string> names = new List<string>();
        List<double> positions = new List<double>();
        List<double> velocities = new List<double>();
        List<double> efforts = new List<double>();

        foreach (var joint in joints)
        {
            if (joint.jointTransform != null)
            {
                names.Add(joint.jointName);

                switch (joint.jointType)
                {
                    case JointType.Revolute:
                        // Extract rotation around the appropriate axis
                        float angle = joint.jointTransform.localEulerAngles.z;
                        positions.Add(angle * Mathf.Deg2Rad);
                        break;
                    case JointType.Prismatic:
                        // Extract position along the appropriate axis
                        float position = joint.jointTransform.localPosition.x;
                        positions.Add(position);
                        break;
                    case JointType.Fixed:
                        positions.Add(0.0);
                        break;
                }

                velocities.Add(0.0); // Calculate if needed
                efforts.Add(0.0);    // Calculate if needed
            }
        }

        jointState.name = names.ToArray();
        jointState.position = positions.ToArray();
        jointState.velocity = velocities.ToArray();
        jointState.effort = efforts.ToArray();

        ros.Send(jointStatesTopic, jointState);
    }

    void PublishTransforms()
    {
        var tf = new TFMessage();
        tf.transforms = new TransformStampedMsg[1];

        var transform = new TransformStampedMsg();
        transform.header = new HeaderMsg();
        transform.header.stamp = new TimeStamp(ROSTCPConnector.ClockType.ROS_TIME);
        transform.header.frame_id = "world";
        transform.child_frame_id = "base_link";

        // Set position
        transform.transform.translation = new Vector3Msg(
            robotBase.position.x,
            robotBase.position.y,
            robotBase.position.z
        );

        // Set rotation
        transform.transform.rotation = new QuaternionMsg(
            robotBase.rotation.x,
            robotBase.rotation.y,
            robotBase.rotation.z,
            robotBase.rotation.w
        );

        tf.transforms[0] = transform;

        ros.Send(tfTopic, tf);
    }
}
```

### 2. Service Client for Robot Control

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.StdSrvs;
using RosMessageTypes.Navigation;
using System;

public class RobotServiceClient : MonoBehaviour
{
    [Header("Service Names")]
    public string setBoolService = "/set_bool";
    public string clearCostmapsService = "/clear_costmaps";
    public string moveBaseService = "/move_base";

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
    }

    // Example: Call a simple boolean service
    public void CallSetBoolService(bool value, Action<bool, string> callback = null)
    {
        var request = new SetBoolRequest();
        request.data = value;

        ros.CallService<SetBoolRequest, SetBoolResponse>(setBoolService, request,
            (response) => {
                Debug.Log($"SetBool service response: {response.success}, {response.message}");
                callback?.Invoke(response.success, response.message);
            },
            (exception) => {
                Debug.LogError($"SetBool service call failed: {exception.Message}");
                callback?.Invoke(false, exception.Message);
            });
    }

    // Example: Clear navigation costmaps
    public void ClearNavigationCostmaps(Action<bool, string> callback = null)
    {
        var request = new EmptyRequest();

        ros.CallService<EmptyRequest, EmptyResponse>(clearCostmapsService,
            (response) => {
                Debug.Log("Costmaps cleared successfully");
                callback?.Invoke(true, "Costmaps cleared");
            },
            (exception) => {
                Debug.LogError($"Failed to clear costmaps: {exception.Message}");
                callback?.Invoke(false, exception.Message);
            });
    }

    // Example: Send navigation goal
    public void SendNavigationGoal(float x, float y, float theta, Action<bool, string> callback = null)
    {
        // Note: This is a simplified example
        // Actual move_base service requires more complex message types
        Debug.Log($"Sending navigation goal to ({x}, {y}, {theta})");

        // In a real implementation, you would use MoveBaseAction or NavigateToPose
        // This is a placeholder for the concept
        callback?.Invoke(true, "Navigation goal sent");
    }

    // Example: Emergency stop service
    public void TriggerEmergencyStop(Action<bool, string> callback = null)
    {
        CallSetBoolService(true, callback);
    }
}
```

### 3. Action Client for Navigation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Actionlib;
using RosMessageTypes.Navigation;
using System;

public class NavigationActionClient : MonoBehaviour
{
    [Header("Action Configuration")]
    public string moveBaseActionName = "/move_base";
    public float actionTimeout = 30.0f;

    private ROSConnection ros;
    private ActionClient<MoveBaseActionGoal, MoveBaseActionResult, MoveBaseActionFeedback> actionClient;

    [Header("Navigation Status")]
    public bool isNavigating = false;
    public ActionStatus currentStatus = ActionStatus.Unknown;

    public enum ActionStatus
    {
        Unknown,
        Pending,
        Active,
        Succeeded,
        Aborted,
        Preempted,
        Recalled,
        Preempting,
        Recalling,
        Lost
    }

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        InitializeActionClient();
    }

    void InitializeActionClient()
    {
        actionClient = new ActionClient<MoveBaseActionGoal, MoveBaseActionResult, MoveBaseActionFeedback>(
            ros, moveBaseActionName);

        actionClient.OnFeedback += OnActionFeedback;
        actionClient.OnResult += OnActionResult;
        actionClient.OnStatus += OnActionStatus;
    }

    public void SendNavigationGoal(float x, float y, float theta)
    {
        var goal = new MoveBaseGoal();
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = new TimeStamp(ROSTCPConnector.ClockType.ROS_TIME);

        // Set position
        goal.target_pose.pose.position.x = x;
        goal.target_pose.pose.position.y = y;
        goal.target_pose.pose.position.z = 0;

        // Convert theta to quaternion
        float halfTheta = theta * 0.5f;
        goal.target_pose.pose.orientation.x = 0;
        goal.target_pose.pose.orientation.y = 0;
        goal.target_pose.pose.orientation.z = Mathf.Sin(halfTheta);
        goal.target_pose.pose.orientation.w = Mathf.Cos(halfTheta);

        var actionGoal = new MoveBaseActionGoal();
        actionGoal.goal_id = new GoalID();
        actionGoal.goal_id.stamp = new TimeStamp(ROSTCPConnector.ClockType.ROS_TIME);
        actionGoal.goal_id.id = $"nav_goal_{Time.time}";
        actionGoal.goal = goal;

        actionClient.SendGoal(actionGoal);
        isNavigating = true;
        currentStatus = ActionStatus.Pending;

        Debug.Log($"Navigation goal sent: ({x}, {y}, {theta})");
    }

    void OnActionFeedback(MoveBaseActionFeedback feedback)
    {
        // Handle feedback from the action server
        Debug.Log("Received navigation feedback");
        // Update UI or visualization based on feedback
    }

    void OnActionResult(MoveBaseActionResult result)
    {
        isNavigating = false;
        Debug.Log($"Navigation result: {result.status.text}");
    }

    void OnActionStatus(GoalStatusArray statusArray)
    {
        if (statusArray.status_list.Length > 0)
        {
            var status = statusArray.status_list[0].status;
            currentStatus = (ActionStatus)status;

            Debug.Log($"Navigation status: {currentStatus}");
        }
    }

    public void CancelNavigation()
    {
        if (actionClient != null)
        {
            actionClient.CancelGoal();
            isNavigating = false;
            currentStatus = ActionStatus.Unknown;
            Debug.Log("Navigation cancelled");
        }
    }

    void OnApplicationQuit()
    {
        if (actionClient != null)
        {
            actionClient.CancelAllGoals();
        }
    }
}
```

## Error Handling and Best Practices

### 1. Robust Connection Management

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using System.Collections;

public class RobustROSConnection : MonoBehaviour
{
    [Header("Connection Configuration")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public int maxRetries = 5;
    public float retryDelay = 2.0f;

    [Header("Status")]
    public bool isConnected = false;
    public int retryCount = 0;

    private ROSConnection ros;
    private Coroutine connectionCoroutine;

    void Start()
    {
        AttemptConnection();
    }

    public void AttemptConnection()
    {
        if (connectionCoroutine != null)
        {
            StopCoroutine(connectionCoroutine);
        }

        connectionCoroutine = StartCoroutine(ConnectionRoutine());
    }

    IEnumerator ConnectionRoutine()
    {
        while (retryCount < maxRetries && !isConnected)
        {
            try
            {
                ros = ROSConnection.GetOrCreateInstance();
                ros.Initialize(rosIPAddress, rosPort);

                // Test the connection
                yield return new WaitForSeconds(0.5f);

                if (ros.IsConnected())
                {
                    isConnected = true;
                    retryCount = 0;
                    Debug.Log("Successfully connected to ROS");
                    OnConnectionEstablished();
                    yield break;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Connection attempt {retryCount + 1} failed: {e.Message}");
            }

            retryCount++;
            Debug.Log($"Retrying connection... ({retryCount}/{maxRetries})");
            yield return new WaitForSeconds(retryDelay);
        }

        if (!isConnected)
        {
            Debug.LogError("Failed to connect to ROS after maximum retries");
            OnConnectionFailed();
        }
    }

    void OnConnectionEstablished()
    {
        // Register message subscriptions here
        // Example: ros.Subscribe<SomeMessage>("/topic", OnMessageReceived);
    }

    void OnConnectionFailed()
    {
        // Handle connection failure - maybe disable certain features
        Debug.LogWarning("ROS connection failed - operating in offline mode");
    }

    public void SendWithRetry<T>(string topic, T message, int maxRetries = 3)
    {
        int attempts = 0;
        while (attempts < maxRetries && ros != null)
        {
            try
            {
                ros.Send(topic, message);
                return; // Success
            }
            catch (System.Exception e)
            {
                attempts++;
                Debug.LogWarning($"Send attempt {attempts} failed: {e.Message}");

                if (attempts >= maxRetries)
                {
                    Debug.LogError($"Failed to send message after {maxRetries} attempts");
                }
                else
                {
                    yield return new WaitForSeconds(0.1f);
                }
            }
        }
    }

    void OnApplicationQuit()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

## Performance Optimization

### 1. Efficient Message Handling

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using System.Collections.Generic;

public class OptimizedMessageHandler : MonoBehaviour
{
    [Header("Optimization Settings")]
    public float messageUpdateRate = 10.0f; // Hz
    public int maxMessageQueueSize = 100;

    private float messageInterval;
    private float lastMessageTime;
    private Queue<object> messageQueue;

    void Start()
    {
        messageInterval = 1.0f / messageUpdateRate;
        messageQueue = new Queue<object>();
    }

    void Update()
    {
        if (Time.time - lastMessageTime >= messageInterval)
        {
            ProcessQueuedMessages();
            lastMessageTime = Time.time;
        }
    }

    public void EnqueueMessage(object message)
    {
        if (messageQueue.Count >= maxMessageQueueSize)
        {
            messageQueue.Dequeue(); // Remove oldest message
        }

        messageQueue.Enqueue(message);
    }

    void ProcessQueuedMessages()
    {
        while (messageQueue.Count > 0)
        {
            var message = messageQueue.Dequeue();
            ProcessMessage(message);
        }
    }

    void ProcessMessage(object message)
    {
        // Process the message efficiently
        // Update only what's necessary for visualization
    }
}
```

## Troubleshooting Common Issues

### Issue: Connection timeout
**Symptoms**: Unity fails to connect to ROS system
**Solutions**:
- Verify ROS master is running: `rosnode list`
- Check IP address and port configuration
- Ensure firewall allows connections on the specified port
- Test connection with: `telnet <ip> <port>`

### Issue: Message deserialization errors
**Symptoms**: Exceptions when receiving ROS messages
**Solutions**:
- Verify message types match between Unity and ROS
- Check for version compatibility between ROS distributions
- Ensure all required message packages are installed

### Issue: Performance degradation
**Symptoms**: Low frame rate when processing ROS messages
**Solutions**:
- Reduce message update rates
- Use object pooling for frequently created objects
- Implement message batching
- Use coroutines for heavy processing

## Next Steps

With these Unity-ROS integration examples, you can create sophisticated digital twin systems that synchronize with real robot data. The next section will cover optimization techniques for high-fidelity rendering in these integrated systems.
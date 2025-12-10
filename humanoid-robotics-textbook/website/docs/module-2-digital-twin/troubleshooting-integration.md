---
sidebar_position: 9
title: "Troubleshooting Integration Issues"
---

# Troubleshooting Integration Issues: Gazebo-Unity Digital Twin

## Overview

This troubleshooting guide addresses common issues encountered when integrating Gazebo physics simulation with Unity visualization in digital twin systems. The guide covers problems related to communication, synchronization, performance, and system compatibility.

## Common Communication Issues

### 1. ROS Bridge Connection Failures

**Problem**: Unity cannot connect to the ROS bridge server.

**Symptoms**:
- Connection timeout errors
- WebSocket connection failures
- "Cannot reach ROS bridge" messages

**Solutions**:
1. **Verify ROS bridge is running**:
   ```bash
   # Check if rosbridge is running
   rostopic list

   # Start rosbridge if not running
   roslaunch rosbridge_server rosbridge_websocket.launch
   ```

2. **Check network configuration**:
   ```bash
   # Verify network connectivity
   netstat -an | grep 9090

   # Check firewall settings
   sudo ufw status
   ```

3. **Configure Unity connection parameters**:
   - Verify WebSocket URL matches ROS bridge address
   - Check port number (default: 9090)
   - Ensure correct protocol (ws:// for WebSocket)

### 2. Topic Subscription Issues

**Problem**: Unity cannot subscribe to ROS topics from Gazebo.

**Symptoms**:
- No data received in Unity
- "Topic not found" errors
- Empty message callbacks

**Solutions**:
1. **Verify topic availability**:
   ```bash
   # List available topics
   rostopic list

   # Check topic type
   rostopic type /joint_states

   # Verify topic messages
   rostopic echo /joint_states -n 1
   ```

2. **Check topic permissions**:
   ```bash
   # Ensure proper ROS environment
   echo $ROS_MASTER_URI
   echo $ROS_IP
   ```

3. **Implement robust subscription**:
   ```csharp
   // Retry connection with exponential backoff
   private IEnumerator SubscribeWithRetry(string topic, int maxRetries = 5)
   {
       int retryCount = 0;
       while (retryCount < maxRetries)
       {
           try
           {
               rosSocket.Subscribe<JointState>(topic, OnJointStateReceived);
               yield break; // Success, exit coroutine
           }
           catch (System.Exception e)
           {
               retryCount++;
               Debug.LogWarning($"Subscription failed, attempt {retryCount}: {e.Message}");
               yield return new WaitForSeconds(Mathf.Pow(2, retryCount)); // Exponential backoff
           }
       }
       Debug.LogError("Max retries reached for topic subscription");
   }
   ```

## Synchronization Problems

### 1. Delayed Synchronization

**Problem**: Gazebo and Unity states are not synchronized in real-time.

**Symptoms**:
- Lag between Gazebo and Unity movements
- Outdated state information
- Poor user experience due to delays

**Solutions**:
1. **Optimize update rates**:
   ```python
   # Balance update rate with network capabilities
   sync_rate = 60  # Hz - adjust based on performance
   ```

2. **Implement interpolation**:
   ```csharp
   // Smooth transitions between states
   private void UpdateRobotState(Vector3 targetPosition, Quaternion targetRotation, float deltaTime)
   {
       transform.position = Vector3.Lerp(transform.position, targetPosition, deltaTime * interpolationSpeed);
       transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, deltaTime * interpolationSpeed);
   }
   ```

3. **Monitor network latency**:
   ```python
   def measure_network_latency(self):
       start_time = time.time()
       # Send ping message
       self.ping_publisher.publish(Empty())
       # Wait for response
       # Calculate round-trip time
       latency = (time.time() - start_time) * 1000  # ms
       return latency
   ```

### 2. Coordinate System Mismatches

**Problem**: Objects appear in wrong positions or orientations between Gazebo and Unity.

**Symptoms**:
- Robot positioned incorrectly in Unity
- Rotations not matching between systems
- Sensor data in wrong coordinate frames

**Solutions**:
1. **Verify coordinate system transformations**:
   ```python
   def transform_gazebo_to_unity(self, gazebo_pose):
       """Convert from Gazebo (right-handed) to Unity (left-handed) coordinates"""
       unity_pose = Pose()
       unity_pose.position.x = gazebo_pose.position.x
       unity_pose.position.y = -gazebo_pose.position.z  # Swap Y and Z, negate Z
       unity_pose.position.z = gazebo_pose.position.y

       # Apply quaternion transformation for rotation
       unity_pose.orientation = self.transform_quaternion(gazebo_pose.orientation)
       return unity_pose
   ```

2. **Check TF tree configuration**:
   ```bash
   # Visualize TF tree
   rosrun tf view_frames

   # Check specific transforms
   rosrun tf tf_echo base_link lidar_link
   ```

3. **Validate URDF joint limits and types**:
   - Ensure joint types match between URDF and Unity models
   - Verify joint limits and ranges are consistent
   - Check that joint names match exactly

## Performance Issues

### 1. Low Frame Rates

**Problem**: Unity frame rate drops significantly during digital twin operation.

**Symptoms**:
- Choppiness in visualization
- Frame rates below 30 FPS
- Input lag and poor responsiveness

**Solutions**:
1. **Optimize rendering settings**:
   - Reduce shadow quality and resolution
   - Use Level of Detail (LOD) for complex models
   - Implement occlusion culling

2. **Throttle sensor data**:
   ```python
   # Reduce sensor update rates for better performance
   lidar_update_rate = 10  # Hz - reduce from default 30
   camera_update_rate = 15  # Hz - reduce from default 30
   ```

3. **Optimize Unity scripts**:
   ```csharp
   // Cache frequently accessed components
   private Transform cachedTransform;
   private Renderer cachedRenderer;

   void Start()
   {
       cachedTransform = transform;
       cachedRenderer = GetComponent<Renderer>();
   }
   ```

### 2. High CPU/GPU Usage

**Problem**: System resources are consumed excessively during digital twin operation.

**Symptoms**:
- High CPU usage (>80%)
- GPU memory exhaustion
- System thermal throttling

**Solutions**:
1. **Profile and optimize**:
   - Use Unity Profiler to identify bottlenecks
   - Optimize shader complexity
   - Reduce polygon count in 3D models

2. **Implement resource management**:
   ```csharp
   // Use object pooling for frequently created objects
   public class PointCloudPool
   {
       private Queue<GameObject> pool = new Queue<GameObject>();
       private GameObject prefab;

       public GameObject GetPoint()
       {
           if (pool.Count > 0)
           {
               GameObject point = pool.Dequeue();
               point.SetActive(true);
               return point;
           }
           else
           {
               return GameObject.Instantiate(prefab);
           }
       }

       public void ReturnPoint(GameObject point)
       {
           point.SetActive(false);
           pool.Enqueue(point);
       }
   }
   ```

## Sensor Simulation Issues

### 1. LiDAR Data Problems

**Problem**: LiDAR sensor data is incorrect or not visualized properly in Unity.

**Symptoms**:
- Empty or invalid LiDAR scans
- Incorrect point cloud visualization
- Range values outside expected bounds

**Solutions**:
1. **Verify Gazebo LiDAR configuration**:
   ```xml
   <sensor type="ray" name="lidar_sensor">
     <ray>
       <scan>
         <horizontal>
           <samples>720</samples>  <!-- Ensure appropriate sample count -->
           <resolution>1</resolution>
           <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
           <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
         </horizontal>
       </scan>
       <range>
         <min>0.1</min>    <!-- Minimum range -->
         <max>30.0</max>   <!-- Maximum range -->
         <resolution>0.01</resolution>
       </range>
     </ray>
   </sensor>
   ```

2. **Implement robust LiDAR visualization**:
   ```csharp
   public void UpdateLidarVisualization(LaserScan scan)
   {
       // Validate scan data before processing
       if (scan.ranges.Count == 0 || scan.angle_increment == 0)
       {
           Debug.LogWarning("Invalid LiDAR data received");
           return;
       }

       // Process valid ranges only
       for (int i = 0; i < scan.ranges.Count; i++)
       {
           double range = scan.ranges[i];
           if (range >= scan.range_min && range <= scan.range_max)
           {
               // Calculate position and update visualization
               float angle = (float)(scan.angle_min + i * scan.angle_increment);
               Vector3 point = CalculateLidarPoint(range, angle);
               UpdateLidarPoint(i, point);
           }
       }
   }
   ```

### 2. Camera Image Issues

**Problem**: Camera images are not displaying correctly or are distorted.

**Symptoms**:
- Black or corrupted images in Unity
- Incorrect image dimensions
- Color channel mismatches

**Solutions**:
1. **Verify camera configuration**:
   ```xml
   <sensor type="camera" name="camera_sensor">
     <camera>
       <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
       <image>
         <width>640</width>
         <height>480</height>
         <format>R8G8B8</format>  <!-- Ensure format matches Unity expectation -->
       </image>
       <clip>
         <near>0.1</near>
         <far>10</far>
       </clip>
     </camera>
   </sensor>
   ```

2. **Implement proper image processing**:
   ```csharp
   public void ProcessCameraImage(Image image)
   {
       // Validate image dimensions
       if (image.width != expectedWidth || image.height != expectedHeight)
       {
           Debug.LogError($"Camera image dimensions mismatch: {image.width}x{image.height}");
           return;
       }

       // Ensure correct data size
       int expectedSize = image.width * image.height * 3; // RGB
       if (image.data.Count != expectedSize)
       {
           Debug.LogError($"Camera image data size mismatch");
           return;
       }

       // Process image data
       UpdateCameraTexture(image);
   }
   ```

## Platform Compatibility Issues

### 1. Cross-Platform Problems

**Problem**: Digital twin system behaves differently across platforms.

**Symptoms**:
- Different performance on Windows vs Linux vs Mac
- Inconsistent behavior between platforms
- Platform-specific errors

**Solutions**:
1. **Use platform-agnostic configurations**:
   ```csharp
   // Conditional compilation for platform-specific code
   #if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
       string rosBridgeUrl = "ws://127.0.0.1:9090";
   #elif UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX
       string rosBridgeUrl = "ws://localhost:9090";
   #endif
   ```

2. **Test on target platforms**:
   - Regularly test on all target platforms
   - Use consistent ROS distributions across platforms
   - Validate network configurations

### 2. Version Compatibility

**Problem**: Different versions of Gazebo, Unity, or ROS cause compatibility issues.

**Symptoms**:
- API incompatibilities
- Plugin loading failures
- Message format mismatches

**Solutions**:
1. **Maintain version documentation**:
   - Document exact versions of all components
   - Test version combinations systematically
   - Use version control for all dependencies

2. **Implement version checking**:
   ```python
   def check_ros_version():
       import rospkg
       rospack = rospkg.RosPack()
       ros_version = rospack.get_path('roscpp')  # Get ROS version info
       return ros_version
   ```

## Advanced Troubleshooting Techniques

### 1. Logging and Monitoring

**Implement comprehensive logging**:
```python
import logging

class IntegrationLogger:
    def __init__(self):
        self.logger = logging.getLogger('digital_twin_integration')
        self.logger.setLevel(logging.DEBUG)

        # Create file handler
        fh = logging.FileHandler('integration.log')
        fh.setLevel(logging.DEBUG)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_sync_status(self, delay, quality):
        self.logger.info(f"Sync Status - Delay: {delay:.3f}s, Quality: {quality:.3f}")
```

### 2. Diagnostic Tools

**Create diagnostic messages**:
```python
from std_msgs.msg import Float32MultiArray

def publish_diagnostics():
    diagnostic_msg = Float32MultiArray()
    diagnostic_msg.data = [
        sync_delay,      # Synchronization delay
        network_latency, # Network latency
        frame_rate,      # Current frame rate
        cpu_usage,       # CPU usage percentage
        memory_usage     # Memory usage percentage
    ]
    diagnostic_publisher.publish(diagnostic_msg)
```

## Prevention Strategies

### 1. Robust System Design
- Implement graceful degradation when components fail
- Use modular architecture for easy debugging
- Include comprehensive error handling

### 2. Continuous Integration
- Set up automated testing for integration
- Monitor system performance continuously
- Implement alerting for critical failures

### 3. Documentation and Knowledge Sharing
- Maintain troubleshooting documentation
- Create runbooks for common issues
- Share knowledge across team members

## Quick Reference

### Common Commands
```bash
# Check ROS status
rostopic list
rostopic hz /joint_states
rostopic echo /joint_states -n 1

# Check network connectivity
netstat -an | grep 9090
ping localhost

# Restart ROS master
roscore
```

### Unity Debugging
- Use Unity Profiler for performance issues
- Check Console for error messages
- Verify Scene hierarchy and component configurations

This troubleshooting guide should be updated regularly as new issues are encountered and resolved in your digital twin implementations.
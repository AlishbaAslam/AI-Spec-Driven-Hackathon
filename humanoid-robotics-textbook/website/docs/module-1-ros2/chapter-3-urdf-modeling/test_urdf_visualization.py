#!/usr/bin/env python3
# Test for URDF visualization functionality
# This test verifies that URDF models can be properly visualized

import unittest
import xml.etree.ElementTree as ET
import tempfile
import os


class TestURDFVisualization(unittest.TestCase):
    """Test cases for URDF visualization functionality."""

    def setUp(self):
        """Set up test URDF models for visualization."""
        # Simple humanoid URDF for testing
        self.humanoid_urdf = '''<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Body -->
  <link name="torso">
    <inertial>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>'''

    def test_visual_elements_exist(self):
        """Test that visual elements exist in URDF for visualization."""
        root = ET.fromstring(self.humanoid_urdf)

        # Find all links with visual elements
        links_with_visual = root.findall('.//link[visual]')
        self.assertGreater(len(links_with_visual), 0, "URDF should have links with visual elements")

        for link in links_with_visual:
            visual = link.find('visual')
            self.assertIsNotNone(visual, f"Link {link.attrib['name']} should have visual element")

            geometry = visual.find('geometry')
            self.assertIsNotNone(geometry, f"Link {link.attrib['name']} visual should have geometry")

            # Check that a geometry type exists (box, cylinder, sphere, mesh)
            geometry_types = ['box', 'cylinder', 'sphere', 'mesh']
            geometry_found = any(visual.find(geom_type) is not None for geom_type in geometry_types)
            self.assertTrue(geometry_found, f"Link {link.attrib['name']} should have a valid geometry type")

    def test_material_definitions(self):
        """Test that materials are properly defined for visualization."""
        root = ET.fromstring(self.humanoid_urdf)

        # Find all materials
        materials = root.findall('.//material')
        self.assertGreater(len(materials), 0, "URDF should have material definitions")

        for material in materials:
            self.assertIn('name', material.attrib, "Material should have a name")

            # Check if color is defined
            color_elem = material.find('color')
            if color_elem is not None:
                self.assertIn('rgba', color_elem.attrib, "Color should have rgba attribute")

                # Parse RGBA values
                rgba_str = color_elem.attrib['rgba']
                rgba_values = list(map(float, rgba_str.split()))
                self.assertEqual(len(rgba_values), 4, "RGBA should have 4 values (r, g, b, a)")

                for val in rgba_values:
                    self.assertGreaterEqual(val, 0.0, "Color values should be >= 0")
                    self.assertLessEqual(val, 1.0, "Color values should be <= 1")

    def test_visual_origin_definitions(self):
        """Test that visual origins are properly defined."""
        root = ET.fromstring(self.humanoid_urdf)

        visuals = root.findall('.//visual')
        self.assertGreater(len(visuals), 0, "URDF should have visual elements")

        for visual in visuals:
            origin = visual.find('origin')
            self.assertIsNotNone(origin, "Visual element should have origin")

            # Check for xyz and rpy attributes
            if 'xyz' in origin.attrib:
                xyz_values = list(map(float, origin.attrib['xyz'].split()))
                self.assertEqual(len(xyz_values), 3, "XYZ should have 3 values (x, y, z)")

            if 'rpy' in origin.attrib:
                rpy_values = list(map(float, origin.attrib['rpy'].split()))
                self.assertEqual(len(rpy_values), 3, "RPY should have 3 values (roll, pitch, yaw)")

    def test_geometry_types_for_visualization(self):
        """Test that geometry types are appropriate for visualization."""
        root = ET.fromstring(self.humanoid_urdf)

        geometry_types_found = set()
        geometries = root.findall('.//geometry/*')

        for geom in geometries:
            geom_type = geom.tag
            self.assertIn(geom_type, ['box', 'cylinder', 'sphere', 'mesh'],
                         f"Invalid geometry type: {geom_type}")
            geometry_types_found.add(geom_type)

        # At least one geometry type should be present
        self.assertGreater(len(geometry_types_found), 0, "URDF should have at least one geometry type")

    def test_joint_visualization_elements(self):
        """Test that joints have appropriate elements for visualization."""
        root = ET.fromstring(self.humanoid_urdf)

        joints = root.findall('joint')
        self.assertGreater(len(joints), 0, "URDF should have joint elements")

        for joint in joints:
            # Check required elements for visualization
            parent = joint.find('parent')
            child = joint.find('child')
            origin = joint.find('origin')

            self.assertIsNotNone(parent, f"Joint {joint.attrib['name']} should have parent")
            self.assertIsNotNone(child, f"Joint {joint.attrib['name']} should have child")
            self.assertIsNotNone(origin, f"Joint {joint.attrib['name']} should have origin")

            # Verify parent and child link names exist in the robot
            parent_link_name = parent.attrib['link']
            child_link_name = child.attrib['link']

            parent_link = root.find(f".//link[@name='{parent_link_name}']")
            child_link = root.find(f".//link[@name='{child_link_name}']")

            self.assertIsNotNone(parent_link, f"Parent link '{parent_link_name}' should exist")
            self.assertIsNotNone(child_link, f"Child link '{child_link_name}' should exist")

    def test_urdf_file_creation_for_visualization(self):
        """Test creating a URDF file that's suitable for visualization."""
        # Create a temporary URDF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(self.humanoid_urdf)
            temp_filename = f.name

        try:
            # Verify the file can be read and parsed
            with open(temp_filename, 'r') as f:
                content = f.read()

            root = ET.fromstring(content)
            self.assertEqual(root.attrib['name'], 'simple_humanoid')

            # Verify it has the expected structure for visualization
            links = root.findall('link')
            joints = root.findall('joint')
            materials = root.findall('.//material')

            self.assertGreater(len(links), 0, "Should have links for visualization")
            self.assertGreater(len(joints), 0, "Should have joints for visualization")
            self.assertGreater(len(materials), 0, "Should have materials for visualization")

        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)

    def test_visualization_ready_urdf(self):
        """Test that URDF is ready for visualization tools like RViz."""
        root = ET.fromstring(self.humanoid_urdf)

        # For RViz visualization, we need:
        # 1. All links to have visual elements
        # 2. Proper material definitions
        # 3. Valid geometry types
        # 4. Proper joint definitions

        links = root.findall('link')
        self.assertGreater(len(links), 0, "Should have links")

        visualization_issues = []

        for link in links:
            link_name = link.attrib['name']

            # Check for visual element
            visual = link.find('visual')
            if visual is None:
                visualization_issues.append(f"Link {link_name} missing visual element")

            # Check for geometry within visual
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is None:
                    visualization_issues.append(f"Link {link_name} visual missing geometry")
                else:
                    # Check if at least one geometry type exists
                    geom_types = ['box', 'cylinder', 'sphere', 'mesh']
                    if not any(geometry.find(gt) is not None for gt in geom_types):
                        visualization_issues.append(f"Link {link_name} has invalid geometry type")

        if visualization_issues:
            print("Visualization issues found:")
            for issue in visualization_issues:
                print(f"  - {issue}")
            self.fail(f"URDF has {len(visualization_issues)} visualization issues")
        else:
            print("URDF is ready for visualization: PASSED")


def main():
    """Run the URDF visualization tests."""
    print("Running URDF visualization tests...")

    # For this test to follow TDD approach, we would normally expect it to fail initially
    # But since we're validating existing knowledge, we expect it to pass
    unittest.main()


if __name__ == '__main__':
    main()
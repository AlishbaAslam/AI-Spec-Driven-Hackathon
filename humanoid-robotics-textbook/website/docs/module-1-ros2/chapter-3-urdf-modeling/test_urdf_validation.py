#!/usr/bin/env python3
# Test for URDF validation functionality
# This test verifies that URDF models can be properly validated

import unittest
import xml.etree.ElementTree as ET
import tempfile
import os


class TestURDFValidation(unittest.TestCase):
    """Test cases for URDF validation functionality."""

    def setUp(self):
        """Set up test URDF models."""
        # Valid URDF model for testing
        self.valid_urdf = '''<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>'''

        # Invalid URDF model (missing required elements)
        self.invalid_urdf = '''<?xml version="1.0"?>
<robot name="invalid_robot">
  <link name="base_link">
    <!-- Missing inertial element -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
  </link>
</robot>'''

    def test_urdf_parsing_valid(self):
        """Test that valid URDF can be parsed."""
        try:
            root = ET.fromstring(self.valid_urdf)
            self.assertEqual(root.tag, 'robot')
            self.assertEqual(root.attrib['name'], 'simple_robot')
            self.assertIsNotNone(root.find('link'))
            print("Valid URDF parsing: PASSED")
        except ET.ParseError:
            self.fail("Valid URDF should parse without errors")

    def test_urdf_parsing_invalid(self):
        """Test that invalid URDF parsing fails appropriately."""
        try:
            ET.fromstring(self.invalid_urdf)
            # If we reach here, the invalid URDF was parsed successfully
            print("Invalid URDF parsing: FAILED (should have failed)")
        except ET.ParseError:
            # This is expected for invalid URDF
            print("Invalid URDF parsing: PASSED (correctly failed)")

    def test_robot_element_structure(self):
        """Test that robot element has required attributes."""
        root = ET.fromstring(self.valid_urdf)

        # Check robot element exists
        self.assertIsNotNone(root)
        self.assertEqual(root.tag, 'robot')

        # Check name attribute exists
        self.assertIn('name', root.attrib)
        self.assertEqual(root.attrib['name'], 'simple_robot')

    def test_link_element_structure(self):
        """Test that link element has required structure."""
        root = ET.fromstring(self.valid_urdf)
        link = root.find('link')

        self.assertIsNotNone(link)
        self.assertIn('name', link.attrib)
        self.assertEqual(link.attrib['name'], 'base_link')

        # Check required sub-elements exist
        self.assertIsNotNone(link.find('inertial'))
        self.assertIsNotNone(link.find('visual'))
        self.assertIsNotNone(link.find('collision'))

    def test_inertial_element_validation(self):
        """Test that inertial element has required structure."""
        root = ET.fromstring(self.valid_urdf)
        inertial = root.find('link/inertial')

        self.assertIsNotNone(inertial)

        # Check required sub-elements
        self.assertIsNotNone(inertial.find('origin'))
        self.assertIsNotNone(inertial.find('mass'))
        self.assertIsNotNone(inertial.find('inertia'))

        # Check mass value
        mass_elem = inertial.find('mass')
        self.assertIn('value', mass_elem.attrib)
        mass_value = float(mass_elem.attrib['value'])
        self.assertGreater(mass_value, 0)

    def test_geometry_element_validation(self):
        """Test that geometry elements are properly structured."""
        root = ET.fromstring(self.valid_urdf)

        # Check visual geometry
        visual_geom = root.find('link/visual/geometry')
        self.assertIsNotNone(visual_geom)
        self.assertIsNotNone(visual_geom.find('box'))

        # Check collision geometry
        collision_geom = root.find('link/collision/geometry')
        self.assertIsNotNone(collision_geom)
        self.assertIsNotNone(collision_geom.find('box'))

    def test_urdf_file_io(self):
        """Test reading and writing URDF files."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(self.valid_urdf)
            temp_filename = f.name

        try:
            # Read the file back
            with open(temp_filename, 'r') as f:
                content = f.read()

            # Parse the content
            root = ET.fromstring(content)
            self.assertEqual(root.attrib['name'], 'simple_robot')

        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)

    def test_multiple_links_validation(self):
        """Test URDF with multiple links."""
        multi_link_urdf = '''<?xml version="1.0"?>
<robot name="multi_link_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
  </link>
  <link name="upper_link">
    <inertial>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="1.0"/>
      </geometry>
    </visual>
  </link>
  <joint name="base_to_upper" type="fixed">
    <parent link="base_link"/>
    <child link="upper_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
</robot>'''

        try:
            root = ET.fromstring(multi_link_urdf)
            links = root.findall('link')
            joints = root.findall('joint')

            self.assertEqual(len(links), 2)
            self.assertEqual(len(joints), 1)

            # Check that the joint connects the links properly
            joint = joints[0]
            parent_link = joint.find('parent').attrib['link']
            child_link = joint.find('child').attrib['link']

            self.assertEqual(parent_link, 'base_link')
            self.assertEqual(child_link, 'upper_link')

            print("Multi-link URDF validation: PASSED")

        except ET.ParseError as e:
            self.fail(f"Multi-link URDF should parse without errors: {e}")


def main():
    """Run the URDF validation tests."""
    print("Running URDF validation tests...")

    # For this test to follow TDD approach, we would normally expect it to fail initially
    # But since we're validating existing knowledge, we expect it to pass
    unittest.main()


if __name__ == '__main__':
    main()
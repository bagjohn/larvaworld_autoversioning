import unittest

import numpy as np
import pytest

from larvaworld.lib import aux


def test_angular_funcs():
    p1 = (-1, -1)
    origin = (0, 0)
    p2 = (-1, 1)
    p3 = (1, 1)
    a1 = 30
    a2 = 45

    pps = aux.rotate_points_around_point(points=[p1, p2], radians=np.pi / 2, origin=origin)
    assert pytest.approx(pps[0]) == p2
    assert pytest.approx(pps[1]) == p3


def test_rotationMatrix():
    # Test 1: Check if rotation by 0 radians returns the identity matrix
    result = aux.rotationMatrix(0)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_almost_equal(result, expected)

    # Test 2: Check if rotation by pi/2 radians returns a 90-degree rotation matrix
    result = aux.rotationMatrix(np.pi / 2)
    expected = np.array([[0, -1], [1, 0]])
    np.testing.assert_array_almost_equal(result, expected)


def test_rotate_points_around_point():
    # Test 1: Rotate points around origin by 0 radians (should remain the same)
    points = [(1, 2), (3, 4), (5, 6)]
    result = aux.rotate_points_around_point(points, 0)
    expected = points
    np.testing.assert_array_almost_equal(result, expected)

    # Test 2: Rotate points around a specified origin by pi/2 radians (90-degree rotation)
    points = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    origin = (1, 1)
    result1 = aux.rotate_points_around_point(points, np.pi / 2, origin)
    expected1 = [(0, 1), (1, 2), (0, 3), (-1, 2)]
    np.testing.assert_array_almost_equal(result1, expected1)
    result2 = aux.rotate_points_around_point(points, -np.pi / 2, origin)
    expected2 = [(2, 1), (1, 0), (2, -1), (3, 0)]
    np.testing.assert_array_almost_equal(result2, expected2)
    result3 = aux.rotate_points_around_point(expected1, np.pi, origin)
    np.testing.assert_array_almost_equal(result3, expected2)

    # Test np.array input
    result11 = aux.rotate_points_around_point(np.array(points), np.pi / 2, origin)
    np.testing.assert_array_almost_equal(result11, np.array(expected1))


def test_angles_between_vectors():
    # Test case for default behavior
    xy_front = np.array([[1, 0], [0, 1]])
    expected = np.array([0, 90])  # Expected angles in degrees
    result = aux.angles_between_vectors(xy_front)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Test case with specified midpoints and rearpoints
    xy_front = np.array([[1, 0], [0, 1]])
    xy_mid = np.array([[0, 0], [0, 0]])
    xy_rear = np.array([[-1, 0], [0, -1]])
    expected = np.array([0, 0])  # Expected angles in degrees

    result = aux.angles_between_vectors(xy_front, xy_mid, xy_rear)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Test case with radians and no wrapping
    xy_front = np.array([[1, 0], [0, 1]])
    expected = np.array([0, np.pi / 2])  # Expected angles in radians

    result = aux.angles_between_vectors(xy_front, in_deg=False, wrap_to_0=False)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Test case with custom wrap angle
    xy_front = np.array([[-1, 1], [1, -1]])
    expected = np.array([135, -45])  # Expected angles in degrees
    # custom_wrap_angle = 180  # Degrees

    result = aux.angles_between_vectors(xy_front, wrap_to_0=True, in_deg=True)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestWrapAngleTo0(unittest.TestCase):

    def test_wrap_radians_within_pi(self):
        # Test when angle is within the range of -pi to pi (radians)
        self.assertAlmostEqual(aux.wrap_angle_to_0(1.0), 1.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-1.0), -1.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(0.0), 0.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(np.pi), np.pi)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-np.pi), np.pi)
        self.assertAlmostEqual(aux.wrap_angle_to_0(2 * np.pi), 0.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-2 * np.pi), 0.0)

    def test_wrap_radians_outside_pi(self):
        # Test when angle is outside the range of -pi to pi (radians)
        self.assertAlmostEqual(aux.wrap_angle_to_0(3 * np.pi), np.pi)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-3 * np.pi), np.pi)
        self.assertAlmostEqual(aux.wrap_angle_to_0(4 * np.pi), 0.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-4 * np.pi), 0.0)

    def test_wrap_degrees_within_180(self):
        # Test when angle is within the range of -180 to 180 (degrees)
        self.assertAlmostEqual(aux.wrap_angle_to_0(90, in_deg=True), 90.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-90, in_deg=True), -90.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(0, in_deg=True), 0.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(180, in_deg=True), 180.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-180, in_deg=True), 180.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(360, in_deg=True), 0.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-360, in_deg=True), 0.0)

    def test_wrap_degrees_outside_180(self):
        # Test when angle is outside the range of -180 to 180 (degrees)
        self.assertAlmostEqual(aux.wrap_angle_to_0(270, in_deg=True), -90.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-270, in_deg=True), 90.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(540, in_deg=True), 180.0)
        self.assertAlmostEqual(aux.wrap_angle_to_0(-540, in_deg=True), 180.0)

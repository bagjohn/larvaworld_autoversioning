import unittest

import numpy as np
import pytest

from larvaworld.lib import aux


class TestCompBearing(unittest.TestCase):

    def test_degrees_in_deg(self):
        xs = [1.0, 2.0, 3.0]
        ys = [1.0, 2.0, 0.0]
        ors = 90.0
        expected = np.array([-135., -135.,  -90.])
        result = aux.comp_bearing(xs, ys, ors, in_deg=True)
        np.testing.assert_almost_equal(result, expected)



    def test_degrees_in_rad(self):
        xs = [1.0, 2.0, 3.0]
        ys = [1.0, 2.0, 0.0]
        ors = 90.0
        expected = np.deg2rad(np.array([-135., -135.,  -90.]))
        result = aux.comp_bearing(xs, ys, ors, in_deg=False)
        np.testing.assert_almost_equal(result, expected)

    # def test_radians_in_rad(self):
    #     xs = [1.0, 2.0, 3.0]
    #     ys = [1.0, 2.0, 1.0]
    #     ors = np.deg2rad(90.0)
    #     expected_result = np.array([90.0, 45.0, 90.0])
    #     result = aux.comp_bearing(xs, ys, ors, in_deg=False)
    #     np.testing.assert_almost_equal(result, expected_result)

    def test_location_argument(self):
        xs = [1.0, 2.0, 3.0]
        ys = [1.0, 2.0, 1.0]
        ors = 90.0
        loc = (1.0, 1.0)
        expected = np.array([  90., -135.,  -90.])
        result = aux.comp_bearing(xs, ys, ors, loc=loc, in_deg=True)
        np.testing.assert_almost_equal(result, expected)

    def test_negative_orientations(self):
        xs = [1.0, 2.0, 3.0]
        ys = [1.0, 0.0, -3.0]
        ors = [-90.0, -180.0, -270.0]

        result = aux.comp_bearing(xs, ys, ors)
        expected = [ 45.,   0., -45.]
        np.testing.assert_almost_equal(result, expected)

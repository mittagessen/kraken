# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest
import os

import shapely.geometry as geom
import numpy as np

from kraken.lib.segmentation import is_in_region, reading_order, topsort

thisfile = os.path.abspath(os.path.dirname(__file__))
resources = os.path.abspath(os.path.join(thisfile, 'resources'))


class TestReadingOrder(unittest.TestCase):

    """
    Test the reading order algorithms.
    """
    def test_is_in_region(self):
        """
        A line should be in its rectangular bounding box.
        """
        line = geom.LineString([(0, 0), (1, 1)])
        polygon = geom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertTrue(is_in_region(line, polygon))
    
    def test_is_in_region2(self):
        """
        A real baseline should be in its polygonization.
        """
        line = geom.LineString([(268,656), (888,656)])
        polygon = geom.Polygon([(268,656), (265,613), (885,611), (888,656), (885,675), (265,672)])
        self.assertTrue(is_in_region(line, polygon))
    
    def test_is_in_region3(self):
        """
        A line that does not cross the box should not be in the region.
        """
        line = geom.LineString([(2, 2), (1, 1)])
        polygon = geom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertFalse(is_in_region(line, polygon))

    def test_order_simple_over_under(self):
        """
        Two lines (as their polygonal boundaries) are already in order.
        In this example, the boundaries are rectangles that align vertically,
        have horizontal base lines and do not overlap or touch::

            AAAA

            BBBB
        
        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        linestr0 = geom.LineString(polygon0)
        line0 = (slice(linestr0.bounds[1], linestr0.bounds[0]), slice(linestr0.bounds[3], linestr0.bounds[2]))
        polygon1 = [[10, 30], [10, 40], [100, 40], [100, 30], [10, 30]]
        linestr1 = geom.LineString(polygon1)
        line1 = (slice(linestr1.bounds[1], linestr1.bounds[0]), slice(linestr1.bounds[3], linestr1.bounds[2]))
        order = reading_order([line0, line1])
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    def test_order_simple_over_under_touching(self):
        """
        Two lines (as their polygonal boundaries) are already in order.
        In this example, the boundaries are rectangles that align vertically,
        have horizontal base lines and touch::

            AAAA
            BBBB
        
        """
        polygon0 = [[10, 10], [10, 30], [100, 30], [100, 10], [10, 10]]
        linestr0 = geom.LineString(polygon0)
        line0 = (slice(linestr0.bounds[1], linestr0.bounds[0]), slice(linestr0.bounds[3], linestr0.bounds[2]))
        polygon1 = [[10, 30], [10, 40], [100, 40], [100, 30], [10, 30]]
        linestr1 = geom.LineString(polygon1)
        line1 = (slice(linestr1.bounds[1], linestr1.bounds[0]), slice(linestr1.bounds[3], linestr1.bounds[2]))
        order = reading_order([line0, line1])
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    def test_order_simple_left_right(self):
        """
        Two lines (as their polygonal boundaries) are already in order.
        In this example, the boundaries are rectangles that align horizontally,
        have horizontal base lines and do not overlap or touch::

            AAAA  BBBB
        
        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        linestr0 = geom.LineString(polygon0)
        line0 = (slice(linestr0.bounds[1], linestr0.bounds[0]), slice(linestr0.bounds[3], linestr0.bounds[2]))
        polygon1 = [[150, 10], [150, 20], [250, 20], [250, 10], [150, 10]]
        linestr1 = geom.LineString(polygon1)
        line1 = (slice(linestr1.bounds[1], linestr1.bounds[0]), slice(linestr1.bounds[3], linestr1.bounds[2]))
        order = reading_order([line0, line1])
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    def test_order_simple_left_right_touching(self):
        """
        Two lines (as their polygonal boundaries) are already in order.
        In this example, the boundaries are rectangles that align horizontally,
        have horizontal base lines and touch::

            AAAA  BBBB
        
        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        linestr0 = geom.LineString(polygon0)
        line0 = (slice(linestr0.bounds[1], linestr0.bounds[0]), slice(linestr0.bounds[3], linestr0.bounds[2]))
        polygon1 = [[100, 10], [100, 20], [250, 20], [250, 10], [100, 10]]
        linestr1 = geom.LineString(polygon1)
        line1 = (slice(linestr1.bounds[1], linestr1.bounds[0]), slice(linestr1.bounds[3], linestr1.bounds[2]))
        order = reading_order([line0, line1])
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    def test_order_real_reverse(self):
        """
        Real example: lines are in reverse order.
        """
        polygon0 = [[474, 2712], [466, 2669], [1741, 2655], [1749, 2696], [1746, 2709], [474, 2725]]
        linestr0 = geom.LineString(polygon0)
        line0 = (slice(linestr0.bounds[1], linestr0.bounds[0]), slice(linestr0.bounds[3], linestr0.bounds[2]))
        polygon1 = [[493, 2409], [488, 2374], [1733, 2361], [1741, 2395], [1738, 2409], [493, 2422]]
        linestr1 = geom.LineString(polygon1)
        line1 = (slice(linestr1.bounds[1], linestr1.bounds[0]), slice(linestr1.bounds[3], linestr1.bounds[2]))
        order = reading_order([line0, line1])
        # line1 should come before line0, lines do not come before themselves
        expected = np.array([[0, 0], [1, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))
    
    def test_order_real_in_order(self):
        """
        Real (modified) example: lines are in order.
        """
        polygon0 = [[493, 2409], [488, 2374], [1733, 2361], [1741, 2395], [1738, 2409], [493, 2422]]
        linestr0 = geom.LineString(polygon0)
        line0 = (slice(linestr0.bounds[1], linestr0.bounds[0]), slice(linestr0.bounds[3], linestr0.bounds[2]))
        polygon1 = [[474, 2712], [466, 2669], [1741, 2655], [1749, 2696], [1746, 2709], [474, 2725]]
        linestr1 = geom.LineString(polygon1)
        line1 = (slice(linestr1.bounds[1], linestr1.bounds[0]), slice(linestr1.bounds[3], linestr1.bounds[2]))
        order = reading_order([line0, line1])
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    def test_topsort_ordered(self):
        """
        Return list for three lines that are already in order.
        """
        partial_sort = np.array([[1,1,1], [0,1,1], [0,0,1]])
        expected = [0,1,2]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

    def test_topsort_ordered_no_self(self):
        """
        Return list for three lines that are already in order.
        """
        partial_sort = np.array([[0,1,1], [0,0,1], [0,0,0]])
        expected = [0,1,2]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

    def test_topsort_unordered(self):
        """
        Return list for three lines that are partially in order.
        """
        partial_sort = np.array([[1,1,1], [0,1,0], [0,1,1]])
        expected = [0,2,1]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

    def test_topsort_unordered_no_self(self):
        """
        Return list for three lines that are partially in order.
        """
        partial_sort = np.array([[0,1,1], [0,0,0], [0,1,0]])
        expected = [0,2,1]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

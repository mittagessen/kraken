# -*- coding: utf-8 -*-
import pytest
import unittest
from typing import Sequence, Tuple

import shapely.geometry as geom
import numpy as np

from kraken.lib.segmentation import is_in_region, reading_order, topsort


def polygon_slices(polygon: Sequence[Tuple[int, int]]) -> Tuple[slice, slice]:
    """Convert polygons to slices for reading_order"""
    linestr = geom.LineString(polygon)
    slices = (slice(linestr.bounds[1], linestr.bounds[3]),
             slice(linestr.bounds[0], linestr.bounds[2]))
    return slices


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
        line = geom.LineString([(268, 656), (888, 656)])
        polygon = geom.Polygon([(268, 656), (265, 613), (885, 611), (888, 656), (885, 675), (265, 672)])
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

        The reading order should be the same for left-to-right and right-to-left.
        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        polygon1 = [[10, 30], [10, 40], [100, 40], [100, 30], [10, 30]]
        order_lr = reading_order([polygon_slices(line) for line in [polygon0, polygon1]])
        order_rl = reading_order([polygon_slices(line) for line in [polygon0, polygon1]], 'rl')
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order_lr, expected), "Reading order is not as expected: {}".format(order_lr))
        self.assertTrue(np.array_equal(order_rl, expected), "Reading order is not as expected: {}".format(order_rl))

    def test_order_simple_over_under_touching(self):
        """
        Two lines (as their polygonal boundaries) are already in order.
        In this example, the boundaries are rectangles that align vertically,
        have horizontal base lines and touch::

            AAAA
            BBBB

        The reading order should be the same for left-to-right and right-to-left.
        """
        polygon0 = [[10, 10], [10, 30], [100, 30], [100, 10], [10, 10]]
        polygon1 = [[10, 30], [10, 40], [100, 40], [100, 30], [10, 30]]
        order_lr = reading_order([polygon_slices(line) for line in [polygon0, polygon1]])
        order_rl = reading_order([polygon_slices(line) for line in [polygon0, polygon1]], 'rl')
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order_lr, expected), "Reading order is not as expected: {}".format(order_lr))
        self.assertTrue(np.array_equal(order_rl, expected), "Reading order is not as expected: {}".format(order_rl))

    def test_order_simple_left_right(self):
        """
        Two lines (as their polygonal boundaries) are already in order.
        In this example, the boundaries are rectangles that align horizontally,
        have horizontal base lines and do not overlap or touch::

            AAAA  BBBB

        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        polygon1 = [[150, 10], [150, 20], [250, 20], [250, 10], [150, 10]]
        order = reading_order([polygon_slices(line) for line in [polygon0, polygon1]])
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    @pytest.mark.xfail
    def test_order_simple_left_right_touching(self):
        """
        Two lines (as their polygonal boundaries) are already in order.
        In this example, the boundaries are rectangles that align horizontally,
        have horizontal base lines and touch::

            AAAABBBB

        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        polygon1 = [[100, 10], [100, 20], [250, 20], [250, 10], [100, 10]]
        order = reading_order([polygon_slices(line) for line in [polygon0, polygon1]])
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    def test_order_simple_right_left(self):
        """
        Two lines (as their polygonal boundaries) are in reverse RTL-order.
        In this example, the boundaries are rectangles that align horizontally,
        have horizontal base lines and do not overlap or touch::

            BBBB  AAAA

        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        polygon1 = [[150, 10], [150, 20], [250, 20], [250, 10], [150, 10]]
        order = reading_order([polygon_slices(line) for line in [polygon0, polygon1]], 'rl')
        # line1 should come before line0, lines do not come before themselves
        expected = np.array([[0, 0], [1, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    @pytest.mark.xfail
    def test_order_simple_right_left_touching(self):
        """
        Two lines (as their polygonal boundaries) are in reverse RTL-order.
        In this example, the boundaries are rectangles that align horizontally,
        have horizontal base lines and touch::

            BBBBAAAA

        """
        polygon0 = [[10, 10], [10, 20], [100, 20], [100, 10], [10, 10]]
        polygon1 = [[100, 10], [100, 20], [250, 20], [250, 10], [100, 10]]
        order = reading_order([polygon_slices(line) for line in [polygon0, polygon1]])
        # line1 should come before line0, lines do not come before themselves
        expected = np.array([[0, 0], [1, 0]])
        self.assertTrue(np.array_equal(order, expected), "Reading order is not as expected: {}".format(order))

    def test_order_real_reverse(self):
        """
        Real example: lines are in reverse order.
        The reading order should be the same for left-to-right and right-to-left.
        """
        polygon0 = [[474, 2712], [466, 2669], [1741, 2655], [1749, 2696], [1746, 2709], [474, 2725]]
        polygon1 = [[493, 2409], [488, 2374], [1733, 2361], [1741, 2395], [1738, 2409], [493, 2422]]
        order_lr = reading_order([polygon_slices(line) for line in [polygon0, polygon1]])
        order_rl = reading_order([polygon_slices(line) for line in [polygon0, polygon1]], 'rl')
        # line1 should come before line0, lines do not come before themselves
        expected = np.array([[0, 0], [1, 0]])
        self.assertTrue(np.array_equal(order_lr, expected), "Reading order is not as expected: {}".format(order_lr))
        self.assertTrue(np.array_equal(order_rl, expected), "Reading order is not as expected: {}".format(order_rl))

    def test_order_real_in_order(self):
        """
        Real (modified) example: lines are in order.
        The reading order should be the same for left-to-right and right-to-left.
        """
        polygon0 = [[493, 2409], [488, 2374], [1733, 2361], [1741, 2395], [1738, 2409], [493, 2422]]
        polygon1 = [[474, 2712], [466, 2669], [1741, 2655], [1749, 2696], [1746, 2709], [474, 2725]]
        order_lr = reading_order([polygon_slices(line) for line in [polygon0, polygon1]])
        order_rl = reading_order([polygon_slices(line) for line in [polygon0, polygon1]], 'rl')
        # line0 should come before line1, lines do not come before themselves
        expected = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(order_lr, expected), "Reading order is not as expected: {}".format(order_lr))
        self.assertTrue(np.array_equal(order_rl, expected), "Reading order is not as expected: {}".format(order_rl))

    def test_topsort_ordered(self):
        """
        Return list for three lines that are already in order.
        """
        partial_sort = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        expected = [0, 1, 2]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

    def test_topsort_ordered_no_self(self):
        """
        Return list for three lines that are already in order.
        """
        partial_sort = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        expected = [0, 1, 2]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

    def test_topsort_unordered(self):
        """
        Return list for three lines that are partially in order.
        """
        partial_sort = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]])
        expected = [0, 2, 1]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

    def test_topsort_unordered_no_self(self):
        """
        Return list for three lines that are partially in order.
        """
        partial_sort = np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]])
        expected = [0, 2, 1]
        self.assertTrue(np.array_equal(topsort(partial_sort), expected))

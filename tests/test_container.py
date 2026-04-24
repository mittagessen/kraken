# -*- coding: utf-8 -*-
"""
Tests for the bbox <-> baseline casting methods on kraken.containers
line and Segmentation records.
"""
import unittest

from kraken.containers import BaselineLine, BBoxLine, Region, Segmentation


SHARED_KWARGS = dict(
    id='line-1',
    text='hello',
    base_dir='L',
    imagename='page.png',
    tags={'type': [{'tag': 'default'}]},
    split='train',
    regions=['r1'],
    language=['eng'],
)

SHARED_FIELDS = ('id', 'text', 'base_dir', 'imagename', 'tags', 'split',
                 'regions', 'language')


class TestBaselineToBBox(unittest.TestCase):
    """
    Test ``BaselineLine.to_bbox``.
    """

    def test_boundary_extents(self):
        """
        Bbox is computed from the bounding polygon extents.
        """
        bl = BaselineLine(
            baseline=[(20, 40), (90, 42)],
            boundary=[(10, 30), (100, 30), (100, 55), (10, 55), (10, 30)],
            **SHARED_KWARGS,
        )
        out = bl.to_bbox()
        self.assertIsInstance(out, BBoxLine)
        self.assertEqual(out.bbox, (10, 30, 100, 55))

    def test_falls_back_to_baseline(self):
        """
        When ``boundary`` is None the baseline polyline extents are used.
        """
        bl = BaselineLine(baseline=[(5, 8), (60, 12), (90, 7)], boundary=None, **SHARED_KWARGS)
        out = bl.to_bbox()
        self.assertEqual(out.bbox, (5, 7, 90, 12))

    def test_falls_back_on_empty_boundary(self):
        """
        An empty ``boundary`` list also falls back to the baseline.
        """
        bl = BaselineLine(baseline=[(5, 8), (60, 12)], boundary=[], **SHARED_KWARGS)
        out = bl.to_bbox()
        self.assertEqual(out.bbox, (5, 8, 60, 12))

    def test_preserves_shared_fields(self):
        """
        Shared ocr_line fields are carried over to the new BBoxLine.
        """
        bl = BaselineLine(baseline=[(0, 0), (1, 1)], **SHARED_KWARGS)
        out = bl.to_bbox()
        for f in SHARED_FIELDS:
            self.assertEqual(getattr(bl, f), getattr(out, f))
        self.assertEqual(out.type, 'bbox')

    def test_text_direction_is_respected(self):
        """
        The ``text_direction`` argument is assigned to the returned BBoxLine.
        """
        for td in ('horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'):
            with self.subTest(text_direction=td):
                bl = BaselineLine(baseline=[(0, 0), (10, 10)], **SHARED_KWARGS)
                out = bl.to_bbox(text_direction=td)
                self.assertEqual(out.text_direction, td)

    def test_does_not_mutate_input(self):
        """
        The input instance's fields are unchanged after the call.
        """
        baseline = [(0, 0), (10, 10)]
        boundary = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        bl = BaselineLine(id='x', baseline=list(baseline), boundary=list(boundary))
        bl.to_bbox()
        self.assertEqual(bl.baseline, baseline)
        self.assertEqual(bl.boundary, boundary)


class TestBBoxToBaseline(unittest.TestCase):
    """
    Test ``BBoxLine.to_baseline``.
    """

    BBOX = (10, 20, 110, 70)
    EXPECTED_CLOSED_BOUNDARY = [(10, 20), (110, 20), (110, 70), (10, 70), (10, 20)]

    # (text_direction, topline) → expected baseline endpoints
    BASELINE_CASES = [
        # horizontal-lr
        ('horizontal-lr', False, [(10, 20 + (3 * 50) // 4), (110, 20 + (3 * 50) // 4)]),
        ('horizontal-lr', True,  [(10, 20 + 50 // 4), (110, 20 + 50 // 4)]),
        ('horizontal-lr', None,  [(10, 20 + 50 // 2), (110, 20 + 50 // 2)]),
        # horizontal-rl (endpoints reversed)
        ('horizontal-rl', False, [(110, 20 + (3 * 50) // 4), (10, 20 + (3 * 50) // 4)]),
        ('horizontal-rl', True,  [(110, 20 + 50 // 4), (10, 20 + 50 // 4)]),
        ('horizontal-rl', None,  [(110, 20 + 50 // 2), (10, 20 + 50 // 2)]),
        # vertical-lr: lower quadrant rotates to the left edge
        ('vertical-lr', False, [(10 + 100 // 4, 20), (10 + 100 // 4, 70)]),
        ('vertical-lr', True,  [(10 + (3 * 100) // 4, 20), (10 + (3 * 100) // 4, 70)]),
        ('vertical-lr', None,  [(10 + 100 // 2, 20), (10 + 100 // 2, 70)]),
        # vertical-rl: lower quadrant rotates to the right edge
        ('vertical-rl', False, [(10 + (3 * 100) // 4, 20), (10 + (3 * 100) // 4, 70)]),
        ('vertical-rl', True,  [(10 + 100 // 4, 20), (10 + 100 // 4, 70)]),
        ('vertical-rl', None,  [(10 + 100 // 2, 20), (10 + 100 // 2, 70)]),
    ]

    def test_placement(self):
        """
        Baseline endpoints are correct for each (text_direction, topline)
        combination, and the boundary is always the closed 5-point rectangle.
        """
        for td, topline, expected_baseline in self.BASELINE_CASES:
            with self.subTest(text_direction=td, topline=topline):
                bx = BBoxLine(bbox=self.BBOX, text_direction=td, **SHARED_KWARGS)
                out = bx.to_baseline(topline=topline)
                self.assertIsInstance(out, BaselineLine)
                self.assertEqual(out.baseline, expected_baseline)
                self.assertEqual(out.boundary, self.EXPECTED_CLOSED_BOUNDARY)
                self.assertEqual(out.type, 'baselines')

    def test_preserves_shared_fields(self):
        """
        Shared ocr_line fields are carried over to the new BaselineLine.
        """
        bx = BBoxLine(bbox=self.BBOX, text_direction='horizontal-lr', **SHARED_KWARGS)
        out = bx.to_baseline()
        for f in SHARED_FIELDS:
            self.assertEqual(getattr(bx, f), getattr(out, f))

    def test_degenerate_bbox(self):
        """
        A zero-extent bbox produces a zero-length baseline without raising.
        """
        bx = BBoxLine(id='x', bbox=(5, 5, 5, 5), text_direction='horizontal-lr')
        out = bx.to_baseline()
        self.assertEqual(out.baseline, [(5, 5), (5, 5)])
        self.assertEqual(out.boundary, [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)])

    def test_does_not_mutate_input(self):
        """
        The input instance's fields are unchanged after the call.
        """
        bx = BBoxLine(id='x', bbox=self.BBOX, text_direction='horizontal-lr')
        bx.to_baseline()
        self.assertEqual(bx.bbox, self.BBOX)
        self.assertEqual(bx.text_direction, 'horizontal-lr')


class TestSegmentationCasts(unittest.TestCase):
    """
    Test ``Segmentation.to_bbox`` and ``Segmentation.to_baselines``.
    """

    def _make_baselines_seg(self):
        return Segmentation(
            type='baselines',
            imagename='page.png',
            text_direction='horizontal-lr',
            script_detection=True,
            lines=[
                BaselineLine(
                    id='l1',
                    baseline=[(10, 50), (100, 50)],
                    boundary=[(10, 30), (100, 30), (100, 70), (10, 70), (10, 30)],
                    text='foo',
                ),
                BaselineLine(
                    id='l2',
                    baseline=[(10, 150), (100, 150)],
                    boundary=[(10, 130), (100, 130), (100, 170), (10, 170), (10, 130)],
                    text='bar',
                ),
            ],
            regions={'text': [Region(id='r1', boundary=[(0, 0), (200, 0), (200, 200), (0, 200), (0, 0)])]},
            line_orders=[[0, 1]],
            language=['eng'],
        )

    def _make_bbox_seg(self):
        return Segmentation(
            type='bbox',
            imagename='page.png',
            text_direction='horizontal-rl',
            script_detection=False,
            lines=[
                BBoxLine(id='l1', bbox=(10, 30, 100, 70), text_direction='horizontal-rl', text='foo'),
                BBoxLine(id='l2', bbox=(10, 130, 100, 170), text_direction='horizontal-rl', text='bar'),
            ],
            regions={'text': [Region(id='r1', boundary=[(0, 0), (200, 0), (200, 200), (0, 200), (0, 0)])]},
            line_orders=[[0, 1]],
            language=['eng'],
        )

    def test_baselines_to_bbox_round_trip(self):
        """
        A baselines Segmentation converts to bbox with inherited text_direction.
        """
        seg = self._make_baselines_seg()
        out = seg.to_bbox()
        self.assertEqual(out.type, 'bbox')
        self.assertEqual(len(out.lines), 2)
        for ln in out.lines:
            self.assertIsInstance(ln, BBoxLine)
            self.assertEqual(ln.text_direction, seg.text_direction)
        self.assertEqual(out.lines[0].bbox, (10, 30, 100, 70))
        self.assertEqual(out.lines[1].bbox, (10, 130, 100, 170))

    def test_bbox_to_baselines_round_trip(self):
        """
        A bbox Segmentation converts to baselines with closed boundaries.
        """
        seg = self._make_bbox_seg()
        out = seg.to_baselines()
        self.assertEqual(out.type, 'baselines')
        self.assertEqual(len(out.lines), 2)
        for ln in out.lines:
            self.assertIsInstance(ln, BaselineLine)
            self.assertEqual(ln.boundary[0], ln.boundary[-1])

    def test_to_baselines_topline_true(self):
        """
        ``topline=True`` places the baseline in the upper quadrant.
        """
        seg = self._make_bbox_seg()  # horizontal-rl
        out = seg.to_baselines(topline=True)
        # h = 40, h//4 = 10 → y = 30 + 10 = 40 on first line
        self.assertEqual(out.lines[0].baseline, [(100, 40), (10, 40)])

    def test_to_baselines_topline_none(self):
        """
        ``topline=None`` places the baseline at the center.
        """
        seg = self._make_bbox_seg()
        out = seg.to_baselines(topline=None)
        # h//2 = 20 → y = 50 on first line
        self.assertEqual(out.lines[0].baseline, [(100, 50), (10, 50)])

    def test_passthrough_fields(self):
        """
        Non-line fields pass through unchanged from the source Segmentation.
        """
        seg = self._make_baselines_seg()
        out = seg.to_bbox()
        self.assertEqual(out.imagename, seg.imagename)
        self.assertEqual(out.text_direction, seg.text_direction)
        self.assertEqual(out.script_detection, seg.script_detection)
        self.assertEqual(out.line_orders, seg.line_orders)
        self.assertEqual(out.language, seg.language)
        self.assertEqual(list(out.regions.keys()), list(seg.regions.keys()))
        self.assertEqual(out.regions['text'][0].id, 'r1')

    def test_identity_case_baselines_deep_copy(self):
        """
        Casting a baselines Segmentation to baselines deep-copies everything.
        """
        seg = self._make_baselines_seg()
        out = seg.to_baselines()
        self.assertIsNot(out, seg)
        self.assertEqual(out.type, 'baselines')
        self.assertIsNot(out.lines, seg.lines)
        self.assertEqual(len(out.lines), len(seg.lines))
        for orig, new in zip(seg.lines, out.lines):
            self.assertIsNot(orig, new)
            self.assertEqual(orig.baseline, new.baseline)
            self.assertEqual(orig.boundary, new.boundary)
        self.assertIsNot(out.regions, seg.regions)
        self.assertIsNot(out.regions['text'], seg.regions['text'])
        self.assertIsNot(out.regions['text'][0], seg.regions['text'][0])
        self.assertIsNot(out.line_orders, seg.line_orders)
        self.assertEqual(out.line_orders, seg.line_orders)

    def test_identity_case_bbox_deep_copy(self):
        """
        Casting a bbox Segmentation to bbox deep-copies everything.
        """
        seg = self._make_bbox_seg()
        out = seg.to_bbox()
        self.assertIsNot(out, seg)
        self.assertIsNot(out.lines, seg.lines)
        for orig, new in zip(seg.lines, out.lines):
            self.assertIsNot(orig, new)
            self.assertEqual(orig.bbox, new.bbox)

    def test_conversion_path_produces_independent_lines(self):
        """
        Cross-type conversion also returns an independent Segmentation.
        """
        seg = self._make_baselines_seg()
        out = seg.to_bbox()
        self.assertIsNot(out.lines, seg.lines)
        for orig, new in zip(seg.lines, out.lines):
            self.assertIsNot(orig, new)
        self.assertIsNot(out.regions, seg.regions)

    def test_empty_lines(self):
        """
        Segmentations with no lines convert without error.
        """
        seg = Segmentation(
            type='baselines',
            imagename='page.png',
            text_direction='horizontal-lr',
            script_detection=False,
            lines=[],
        )
        out = seg.to_bbox()
        self.assertEqual(out.type, 'bbox')
        self.assertEqual(out.lines, [])

    def test_bbox_to_baselines_vertical(self):
        """
        Vertical-rl bbox Segmentation produces baselines on the right edge.
        """
        seg = Segmentation(
            type='bbox',
            imagename='page.png',
            text_direction='vertical-rl',
            script_detection=False,
            lines=[BBoxLine(id='l1', bbox=(10, 20, 110, 70), text_direction='vertical-rl')],
        )
        out = seg.to_baselines(topline=False)
        # vertical-rl lower quadrant → right edge inset: x = 10 + (3*100)//4 = 85
        self.assertEqual(out.lines[0].baseline, [(85, 20), (85, 70)])

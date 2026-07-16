# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

from pytest import raises

from kraken.lib import xml
from kraken.containers import BaselineLine, BBoxLine

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestXMLParser(unittest.TestCase):
    """
    Tests XML (ALTO/PAGE) parsing
    """
    def setUp(self):
        self.alto_doc_root = resources / 'alto'
        self.page_doc_root = resources / 'page'
        self.alto_doc = self.alto_doc_root / 'bsb00084914_00007.xml'
        self.alto_zero_dims_doc = self.alto_doc_root / 'zero_dims.xml'
        self.page_doc = self.page_doc_root / 'cPAS-2000.xml'
        self.page_zero_dims_doc = self.page_doc_root / 'zero_dims.xml'
        self.reg_alto_doc = self.alto_doc_root / 'reg_test.xml'

        self.invalid_alto_docs = self.alto_doc_root / 'invalid'
        self.invalid_page_docs = self.page_doc_root / 'invalid'

    def test_page_parsing(self):
        """
        Test parsing of PAGE XML files with reading order.
        """
        doc = xml.XMLPage(self.page_doc, filetype='page')
        self.assertEqual(len(doc.get_sorted_lines()), 97)
        self.assertEqual(len([item for x in doc.regions.values() for item in x]), 4)

    def test_alto_parsing(self):
        """
        Test parsing of ALTO XML files with reading order.
        """
        doc = xml.XMLPage(self.alto_doc, filetype='alto')
        self.assertEqual(len(doc.get_sorted_lines()), 30)
        self.assertEqual(len([item for x in doc.regions.values() for item in x]), 5)

    def test_auto_parsing(self):
        """
        Test parsing of PAGE and ALTO XML files with auto-format determination.
        """
        doc = xml.XMLPage(self.page_doc, filetype='xml')
        self.assertEqual(doc.filetype, 'page')
        doc = xml.XMLPage(self.alto_doc, filetype='xml')
        self.assertEqual(doc.filetype, 'alto')

    def test_failure_invalid_documents(self):
        """
        Test that parsing invalid or format-mismatched documents raises ValueError.
        """
        cases = [
            ('alto_as_page', self.alto_doc, {'filetype': 'page'}),
            ('page_as_alto', self.page_doc, {'filetype': 'alto'}),
            ('alto_invalid_image', self.invalid_alto_docs / 'image.xml', {}),
            ('alto_invalid_measurement_unit', self.invalid_alto_docs / 'mu.xml', {}),
            ('alto_invalid_dims', self.invalid_alto_docs / 'dims.xml', {}),
            ('alto_zero_dims_missing_image', self.invalid_alto_docs / 'zero_dims_missing_image.xml', {'filetype': 'alto'}),
            ('page_invalid_image', self.invalid_page_docs / 'image.xml', {}),
            ('page_invalid_dims', self.invalid_page_docs / 'dims.xml', {}),
            ('page_zero_dims_missing_image', self.invalid_page_docs / 'zero_dims_missing_image.xml', {'filetype': 'page'}),
        ]
        for desc, path, kwargs in cases:
            with self.subTest(desc):
                with raises(ValueError):
                    xml.XMLPage(path, **kwargs)

    def test_zero_dims_fallback_to_image_size(self):
        """
        Test that zero page dimensions are resolved from the image file.
        """
        for fmt, path in (('alto', self.alto_zero_dims_doc),
                          ('page', self.page_zero_dims_doc)):
            with self.subTest(fmt=fmt):
                doc = xml.XMLPage(path, filetype=fmt)
                self.assertEqual(doc.image_size, (123, 45))

    def test_alto_basedirection(self):
        """
        Test proper handling of base direction attribute, including inheritance
        from regions.
        """
        seg = xml.XMLPage(self.alto_doc).to_container()
        base_dirs = [x.base_dir for x in seg.lines]
        self.assertEqual(base_dirs, ['L', 'L', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', None, None, None, None, 'R'])

    def test_alto_language_parsing(self):
        """
        Test proper handling of language attribute, including inheritance from
        page and region.
        """
        seg = xml.XMLPage(self.alto_doc).to_container()
        languages = [x.language for x in seg.lines]
        self.assertEqual(languages, [['iai'], ['deu', 'heb'], ['deu', 'heb'],
                                     ['deu', 'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                     ['deu', 'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                     ['deu', 'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                     ['deu', 'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                     ['deu', 'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                     ['deu', 'heb'], ['deu', 'heb'], ['deu', 'heb'],
                                     ['deu', 'heb'], ['deu', 'heb'], ['eng'],
                                     ['deu', 'heb'], ['hbo'], ['hbo'], ['hbo'],
                                     ['deu', 'eng'], ['hbo']])
        for k, langs in [('Main', [['deu', 'heb']]),
                         ('Paratext', [['hbo'], ['hbo'], ['hbo']]),
                         ('Margin', [['hbo']])]:
            self.assertEqual([x.language for x in seg.regions[k]],
                             langs)

    def test_alto_fallback_region_boundaries(self):
        """
        Test region boundary parsing hierarchy shape -> rect -> None.
        """
        doc = xml.XMLPage(self.reg_alto_doc)
        self.assertEqual(set(doc.regions.keys()), set(['text']))
        for reg, boundary in zip(doc.regions['text'], [[(812, 606), (2755, 648), (2723, 3192), (808, 3240)],
                                                       [(596, 2850), (596, 3008), (729, 3008), (729, 2850)],
                                                       None]):
            self.assertEqual(reg.boundary, boundary)

    def test_alto_tag_parsing(self):
        """
        Test correct parsing of tag references.
        """
        seg = xml.XMLPage(self.alto_doc).to_container()
        line_tags = [line.tags for line in seg.lines]
        self.assertEqual(line_tags, [None, None, {'type': [{'type': 'heading'}]},
                                     None, None, None, None, None, None, None, None,
                                     {'label_0': [{'type': 'foo'}], 'label_1': [{'type': 'bar'}]},
                                     {'label_1': [{'type': 'bar'}, {'type': 'baz'}]},
                                     None, None, None, None, None, None, None, None, None, None,
                                     {'language': [{'type': 'eng'}]}, None, None, None, None,
                                     {'language': [{'type': 'deu'}, {'type': 'eng'}]}, None])

    def test_alto_split_parsing(self):
        """
        Test correct parsing of splits: TAGREFS pointing at TYPE="split"
        OtherTags set line.split and are removed from line.tags.
        """
        seg = xml.XMLPage(self.alto_doc).to_container()
        line_splits = [line.split for line in seg.lines]
        expected = [None] * len(seg.lines)
        expected[2] = 'train'
        expected[12] = 'test'
        expected[13] = 'validation'
        expected[18] = 'train'
        self.assertEqual(line_splits, expected)
        for line in seg.lines:
            if line.tags is not None:
                self.assertNotIn('split', line.tags)

    def test_alto_baseline_linetype(self):
        """
        Test parsing with baseline line objects.
        """
        seg = xml.XMLPage(self.alto_doc, linetype='baselines').to_container()
        self.assertEqual(len(seg.lines), 30)
        for line in seg.lines:
            self.assertIsInstance(line, BaselineLine)

    def test_alto_bbox_linetype(self):
        """
        Test parsing with bbox line objects.
        """
        seg = xml.XMLPage(self.alto_doc, linetype='bbox').to_container()
        self.assertEqual(len(seg.lines), 31)
        for line in seg.lines:
            self.assertIsInstance(line, BBoxLine)

    def test_page_basedirection(self):
        """
        Test proper handling of base direction attribute, including inheritance
        from regions.
        """
        seg = xml.XMLPage(self.page_doc).to_container()
        base_dirs = [x.base_dir for x in seg.lines]
        self.assertEqual(base_dirs, ['R', 'L', 'L', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
                                     'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                     'L'])

    def test_page_split_parsing(self):
        """
        Test correct parsing of splits.
        """
        seg = xml.XMLPage(self.page_doc).to_container()
        base_dirs = [x.split for x in seg.lines]
        self.assertEqual(base_dirs, ['train', None, None, None, 'validation', None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, 'train',
                                     'invalid', None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None,
                                     None, None, None])

    def test_page_language_parsing(self):
        """
        Test proper handling of language attribute, custom string and
        inheritance from page and region.
        """
        seg = xml.XMLPage(self.page_doc).to_container()
        languages = [x.language for x in seg.lines]
        self.assertEqual(languages, [['hbo'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['deu'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'], ['heb', 'deu', 'eng'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu'],
                                     ['pol', 'deu'], ['pol', 'deu'], ['pol', 'deu']])
        reg_langs = [x.language for x in seg.regions['Machine\\u0020Printed\\u0020text']]
        self.assertEqual(reg_langs, [['hbo'], ['heb', 'deu', 'eng'], ['pol', 'deu']])

    def test_alto_ro_string_refs_flattened(self):
        """
        Test that ALTO reading orders with String-level refs are properly
        flattened to line-level IDs instead of being discarded.
        """
        doc = xml.XMLPage(self.alto_doc, filetype='alto')
        # og_0 was previously discarded because it contained String refs
        self.assertIn('og_0', doc.reading_orders)
        ro = doc.reading_orders['og_0']
        self.assertEqual(ro['level'], 'line')
        # All IDs in the flattened order should be valid line IDs
        for lid in ro['order']:
            self.assertIn(lid, doc.lines)
        # og_0 should have 30 lines (all lines in the document)
        self.assertEqual(len(ro['order']), 30)

    def test_alto_ro_region_level_flattened(self):
        """
        Test that region-level flattened orders are created for ALTO explicit ROs.
        """
        doc = xml.XMLPage(self.alto_doc, filetype='alto')
        self.assertIn('og_0:regions', doc.reading_orders)
        ro = doc.reading_orders['og_0:regions']
        self.assertEqual(ro['level'], 'region')
        # All IDs should be valid region IDs
        region_ids = {reg.id for regs in doc.regions.values() for reg in regs}
        for rid in ro['order']:
            self.assertIn(rid, region_ids)

    def test_alto_order_levels(self):
        """
        Test that reading orders have correct level annotations.
        """
        doc = xml.XMLPage(self.alto_doc, filetype='alto')
        self.assertEqual(doc.reading_orders['line_implicit']['level'], 'line')
        self.assertEqual(doc.reading_orders['region_implicit']['level'], 'region')

    def test_page_explicit_ro_parsing(self):
        """
        Test parsing of PageXML files with explicit ReadingOrder element.
        """
        doc = xml.XMLPage(self.page_doc_root / 'explicit_ro.xml', filetype='page')
        self.assertEqual(len(doc.lines), 5)
        self.assertEqual(len([item for x in doc.regions.values() for item in x]), 3)
        # The explicit RO should exist as a flattened line-level order
        self.assertIn('ro_1', doc.reading_orders)
        ro = doc.reading_orders['ro_1']
        self.assertEqual(ro['level'], 'line')
        # The order is r2, r1, r3 -> expanded to lines
        line_ids = ro['order']
        # r2 has l3, l4; r1 has l1, l2; r3 has l5
        self.assertEqual(line_ids, ['l3', 'l4', 'l1', 'l2', 'l5'])

    def test_page_explicit_ro_region_level(self):
        """
        Test that region-level flattened order is created for PageXML explicit RO.
        """
        doc = xml.XMLPage(self.page_doc_root / 'explicit_ro.xml', filetype='page')
        self.assertIn('ro_1:regions', doc.reading_orders)
        ro = doc.reading_orders['ro_1:regions']
        self.assertEqual(ro['level'], 'region')
        self.assertEqual(ro['order'], ['r2', 'r1', 'r3'])

    def test_page_explicit_ro_unordered_group(self):
        """
        Test PageXML with top-level UnorderedGroup producing multiple partial orders.
        """
        doc = xml.XMLPage(self.page_doc_root / 'explicit_ro_unordered.xml', filetype='page')
        self.assertIn('ro_main', doc.reading_orders)
        self.assertIn('ro_margin', doc.reading_orders)
        # Both are children of UnorderedGroup => partial
        self.assertFalse(doc.reading_orders['ro_main']['is_total'])
        self.assertFalse(doc.reading_orders['ro_margin']['is_total'])
        # Main: r1, r2 -> l1, l2
        self.assertEqual(doc.reading_orders['ro_main']['order'], ['l1', 'l2'])
        # Margin: r3 -> l3
        self.assertEqual(doc.reading_orders['ro_margin']['order'], ['l3'])

    def test_page_order_levels(self):
        """
        Test that PageXML reading orders have correct level annotations.
        """
        doc = xml.XMLPage(self.page_doc, filetype='page')
        self.assertEqual(doc.reading_orders['line_implicit']['level'], 'line')
        self.assertEqual(doc.reading_orders['region_implicit']['level'], 'region')
        self.assertEqual(doc.reading_orders['region_transkribus']['level'], 'region')

    def test_to_container_line_orders(self):
        """
        Test that to_container() produces correct list[list[int]] for line_orders.
        """
        doc = xml.XMLPage(self.page_doc_root / 'explicit_ro.xml', filetype='page')
        seg = doc.to_container()
        # line_orders should be a list of lists of integers
        self.assertIsInstance(seg.line_orders, list)
        for order in seg.line_orders:
            self.assertIsInstance(order, list)
            for idx in order:
                self.assertIsInstance(idx, int)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(seg.lines))

    def test_to_container_explicit_ro_indices(self):
        """
        Test that explicit reading order indices match expected line positions.
        """
        doc = xml.XMLPage(self.page_doc_root / 'explicit_ro.xml', filetype='page')
        seg = doc.to_container()
        # Find the explicit ro_1 order in line_orders
        # The implicit order (line_implicit) gives: l1, l2, l3, l4, l5 (indices 0-4)
        # The explicit ro_1 order is: l3, l4, l1, l2, l5 (indices 2, 3, 0, 1, 4)
        line_id_to_idx = {line.id: idx for idx, line in enumerate(seg.lines)}
        expected_ro1_indices = [line_id_to_idx['l3'], line_id_to_idx['l4'],
                                line_id_to_idx['l1'], line_id_to_idx['l2'],
                                line_id_to_idx['l5']]
        # The explicit order should be in line_orders
        self.assertIn(expected_ro1_indices, seg.line_orders)

    def test_get_sorted_lines_by_region(self):
        """
        Test that get_sorted_lines_by_region returns line objects (not tuples).
        """
        doc = xml.XMLPage(self.page_doc_root / 'explicit_ro.xml', filetype='page')
        region_lines = doc.get_sorted_lines_by_region('r1')
        self.assertEqual(len(region_lines), 2)
        for line in region_lines:
            self.assertIsInstance(line, BaselineLine)
        self.assertEqual(region_lines[0].id, 'l1')
        self.assertEqual(region_lines[1].id, 'l2')

    def test_get_sorted_regions(self):
        """
        Test that get_sorted_regions returns Region objects in correct order.
        """
        doc = xml.XMLPage(self.page_doc_root / 'explicit_ro.xml', filetype='page')
        from kraken.containers import Region
        regions = doc.get_sorted_regions('ro_1:regions')
        self.assertEqual(len(regions), 3)
        self.assertEqual([r.id for r in regions], ['r2', 'r1', 'r3'])
        for r in regions:
            self.assertIsInstance(r, Region)

    # ---- Tests for graceful degradation with missing region coordinates ----

    def _missing_coords_cases(self):
        return [('alto', dict(doc=self.alto_doc_root / 'missing_coords_ro.xml',
                              nocoords_line='tl_3',
                              nocoords_region='tb_nocoords',
                              regions=['tb_1', 'tb_2'],
                              other_lines=['tl_1', 'tl_2', 'tl_4'],
                              ro='og_test')),
                ('page', dict(doc=self.page_doc_root / 'missing_coords_ro.xml',
                              nocoords_line='l3',
                              nocoords_region='r_nocoords',
                              regions=['r1', 'r2'],
                              other_lines=['l1', 'l2', 'l4'],
                              ro='ro_test'))]

    def test_missing_region_coords_lines_parsed(self):
        """
        Test that lines inside a region without coordinates are still parsed
        and included in the line list.
        """
        for fmt, case in self._missing_coords_cases():
            with self.subTest(fmt=fmt):
                doc = xml.XMLPage(case['doc'], filetype=fmt)
                self.assertIn(case['nocoords_line'], doc.lines)
                # Lines from coordinate-less regions have empty regions list
                self.assertEqual(doc.lines[case['nocoords_line']].regions, [])

    def test_missing_region_coords_region_excluded(self):
        """
        Test that a region without coordinates is not included in the regions
        dict.
        """
        for fmt, case in self._missing_coords_cases():
            with self.subTest(fmt=fmt):
                doc = xml.XMLPage(case['doc'], filetype=fmt)
                all_region_ids = {r.id for regs in doc.regions.values() for r in regs}
                self.assertNotIn(case['nocoords_region'], all_region_ids)
                # Regions with coordinates are still present
                for region_id in case['regions']:
                    self.assertIn(region_id, all_region_ids)

    def test_missing_region_coords_implicit_orders(self):
        """
        Test that a region without coordinates is excluded from implicit
        region order but its lines are in the implicit line order.
        """
        for fmt, case in self._missing_coords_cases():
            with self.subTest(fmt=fmt):
                doc = xml.XMLPage(case['doc'], filetype=fmt)
                self.assertNotIn(case['nocoords_region'], doc.reading_orders['region_implicit']['order'])
                for region_id in case['regions']:
                    self.assertIn(region_id, doc.reading_orders['region_implicit']['order'])
                self.assertIn(case['nocoords_line'], doc.reading_orders['line_implicit']['order'])

    def test_missing_region_coords_explicit_ro_skips(self):
        """
        Test that an explicit reading order referencing a region without
        coordinates skips it gracefully.
        """
        for fmt, case in self._missing_coords_cases():
            with self.subTest(fmt=fmt):
                doc = xml.XMLPage(case['doc'], filetype=fmt)
                # the explicit RO references both regions with coordinates and
                # the one without; the latter is skipped at both levels, so its
                # contained line is never expanded
                ro_line = doc.reading_orders[case['ro']]
                self.assertEqual(ro_line['level'], 'line')
                for line_id in case['other_lines']:
                    self.assertIn(line_id, ro_line['order'])
                self.assertNotIn(case['nocoords_line'], ro_line['order'])

                ro_region = doc.reading_orders[f"{case['ro']}:regions"]
                self.assertEqual(ro_region['level'], 'region')
                self.assertNotIn(case['nocoords_region'], ro_region['order'])
                self.assertEqual(ro_region['order'], case['regions'])

    def test_missing_region_coords_to_container(self):
        """
        Test that to_container() works correctly when some regions lack
        coordinates.
        """
        for fmt, case in self._missing_coords_cases():
            with self.subTest(fmt=fmt):
                doc = xml.XMLPage(case['doc'], filetype=fmt)
                seg = doc.to_container()
                self.assertEqual(len(seg.lines), 4)
                line_ids = [line.id for line in seg.lines]
                self.assertIn(case['nocoords_line'], line_ids)
                # line_orders should contain valid indices
                for order in seg.line_orders:
                    for idx in order:
                        self.assertIsInstance(idx, int)
                        self.assertGreaterEqual(idx, 0)
                        self.assertLess(idx, len(seg.lines))

    def test_page_missing_region_coords_transkribus_ro(self):
        """
        Test that Transkribus-style custom reading order skips regions without
        coordinates.
        """
        doc = xml.XMLPage(self.page_doc_root / 'missing_coords_transkribus.xml', filetype='page')
        # r_nocoords should not be in transkribus region order
        tr_ro = doc.reading_orders['region_transkribus']
        self.assertNotIn('r_nocoords', tr_ro['order'])
        self.assertEqual(tr_ro['order'], ['r1', 'r2'])

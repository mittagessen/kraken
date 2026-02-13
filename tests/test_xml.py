# -*- coding: utf-8 -*-
import json
import unittest
from pathlib import Path

import numpy as np
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

    def test_auto_parsing(self):
        """
        Test parsing of PAGE and ALTO XML files with auto-format determination.
        """
        doc = xml.XMLPage(self.page_doc, filetype='xml')
        self.assertEqual(doc.filetype, 'page')
        doc = xml.XMLPage(self.alto_doc, filetype='xml')
        self.assertEqual(doc.filetype, 'alto')

    def test_failure_page_alto_parsing(self):
        """
        Test that parsing ALTO files with PAGE as format fails.
        """
        with raises(ValueError):
            xml.XMLPage(self.alto_doc, filetype='page')

    def test_failure_alto_page_parsing(self):
        """
        Test that parsing PAGE files with ALTO as format fails.
        """
        with raises(ValueError):
            xml.XMLPage(self.page_doc, filetype='alto')

    def test_failure_alto_invalid_image(self):
        """
        Test that parsing aborts if image file path is invalid.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_alto_docs / 'image.xml')

    def test_failure_alto_invalid_measurement_unit(self):
        """
        Test that parsing aborts if measurement unit isn't "pixel"
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_alto_docs / 'mu.xml')

    def test_failure_alto_invalid_dims(self):
        """
        Test that parsing aborts if page dimensions aren't parseable as ints.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_alto_docs / 'dims.xml')

    def test_alto_zero_dims_fallback_to_image_size(self):
        """
        Test that zero ALTO page dimensions are resolved from the image file.
        """
        doc = xml.XMLPage(self.alto_zero_dims_doc, filetype='alto')
        self.assertEqual(doc.image_size, (123, 45))

    def test_failure_alto_zero_dims_missing_image(self):
        """
        Test that parsing aborts if ALTO dimensions are zero and image is missing.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_alto_docs / 'zero_dims_missing_image.xml', filetype='alto')

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
        Test correct parsing of splits.
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

    def test_failure_page_invalid_image(self):
        """
        Test that parsing aborts if image file path is invalid.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_page_docs / 'image.xml')

    def test_failure_page_invalid_dims(self):
        """
        Test that parsing aborts if page dimensions aren't parseable as ints.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_page_docs / 'dims.xml')

    def test_page_zero_dims_fallback_to_image_size(self):
        """
        Test that zero PageXML dimensions are resolved from the image file.
        """
        doc = xml.XMLPage(self.page_zero_dims_doc, filetype='page')
        self.assertEqual(doc.image_size, (123, 45))

    def test_failure_page_zero_dims_missing_image(self):
        """
        Test that parsing aborts if PageXML dimensions are zero and image is missing.
        """
        with raises(ValueError):
            xml.XMLPage(self.invalid_page_docs / 'zero_dims_missing_image.xml', filetype='page')

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

    def test_alto_missing_region_coords_lines_parsed(self):
        """
        Test that lines inside an ALTO region without coordinates are still
        parsed and included in the line list.
        """
        doc = xml.XMLPage(self.alto_doc_root / 'missing_coords_ro.xml', filetype='alto')
        # tl_3 is inside tb_nocoords which has no coordinates
        self.assertIn('tl_3', doc.lines)
        # Lines from coordinate-less regions have empty regions list
        self.assertEqual(doc.lines['tl_3'].regions, [])

    def test_alto_missing_region_coords_region_excluded(self):
        """
        Test that an ALTO region without coordinates is not included in the
        regions dict.
        """
        doc = xml.XMLPage(self.alto_doc_root / 'missing_coords_ro.xml', filetype='alto')
        all_region_ids = {r.id for regs in doc.regions.values() for r in regs}
        self.assertNotIn('tb_nocoords', all_region_ids)
        # Regions with coordinates are still present
        self.assertIn('tb_1', all_region_ids)
        self.assertIn('tb_2', all_region_ids)

    def test_alto_missing_region_coords_implicit_orders(self):
        """
        Test that an ALTO region without coordinates is excluded from implicit
        region order but its lines are in the implicit line order.
        """
        doc = xml.XMLPage(self.alto_doc_root / 'missing_coords_ro.xml', filetype='alto')
        # Region implicit order should not contain tb_nocoords
        self.assertNotIn('tb_nocoords', doc.reading_orders['region_implicit']['order'])
        self.assertIn('tb_1', doc.reading_orders['region_implicit']['order'])
        self.assertIn('tb_2', doc.reading_orders['region_implicit']['order'])
        # Line implicit order should still contain tl_3
        self.assertIn('tl_3', doc.reading_orders['line_implicit']['order'])

    def test_alto_missing_region_coords_explicit_ro_skips(self):
        """
        Test that an ALTO explicit reading order referencing a region without
        coordinates skips it gracefully.
        """
        doc = xml.XMLPage(self.alto_doc_root / 'missing_coords_ro.xml', filetype='alto')
        # The explicit RO references tb_1, tb_nocoords, tb_2
        # tb_nocoords should be skipped in both line and region level
        ro_line = doc.reading_orders['og_test']
        self.assertEqual(ro_line['level'], 'line')
        # tl_3 is inside tb_nocoords - since the region is missing, it can't
        # be expanded, so only lines from tb_1 and tb_2 are present
        self.assertIn('tl_1', ro_line['order'])
        self.assertIn('tl_2', ro_line['order'])
        self.assertIn('tl_4', ro_line['order'])
        self.assertNotIn('tl_3', ro_line['order'])

        ro_region = doc.reading_orders['og_test:regions']
        self.assertEqual(ro_region['level'], 'region')
        self.assertNotIn('tb_nocoords', ro_region['order'])
        self.assertEqual(ro_region['order'], ['tb_1', 'tb_2'])

    def test_alto_missing_region_coords_to_container(self):
        """
        Test that to_container() works correctly when some ALTO regions lack
        coordinates.
        """
        doc = xml.XMLPage(self.alto_doc_root / 'missing_coords_ro.xml', filetype='alto')
        seg = doc.to_container()
        # All 4 lines should be present
        self.assertEqual(len(seg.lines), 4)
        line_ids = [l.id for l in seg.lines]
        self.assertIn('tl_3', line_ids)
        # line_orders should contain valid indices
        for order in seg.line_orders:
            for idx in order:
                self.assertIsInstance(idx, int)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(seg.lines))

    def test_page_missing_region_coords_lines_parsed(self):
        """
        Test that lines inside a PageXML region without coordinates are still
        parsed and included in the line list.
        """
        doc = xml.XMLPage(self.page_doc_root / 'missing_coords_ro.xml', filetype='page')
        # l3 is inside r_nocoords which has no valid coordinates
        self.assertIn('l3', doc.lines)
        # Lines from coordinate-less regions have empty regions list
        self.assertEqual(doc.lines['l3'].regions, [])

    def test_page_missing_region_coords_region_excluded(self):
        """
        Test that a PageXML region without coordinates is not included in the
        regions dict.
        """
        doc = xml.XMLPage(self.page_doc_root / 'missing_coords_ro.xml', filetype='page')
        all_region_ids = {r.id for regs in doc.regions.values() for r in regs}
        self.assertNotIn('r_nocoords', all_region_ids)
        self.assertIn('r1', all_region_ids)
        self.assertIn('r2', all_region_ids)

    def test_page_missing_region_coords_implicit_orders(self):
        """
        Test that a PageXML region without coordinates is excluded from
        implicit region order but its lines are in the implicit line order.
        """
        doc = xml.XMLPage(self.page_doc_root / 'missing_coords_ro.xml', filetype='page')
        self.assertNotIn('r_nocoords', doc.reading_orders['region_implicit']['order'])
        self.assertIn('r1', doc.reading_orders['region_implicit']['order'])
        self.assertIn('r2', doc.reading_orders['region_implicit']['order'])
        # l3 should still be in implicit line order
        self.assertIn('l3', doc.reading_orders['line_implicit']['order'])

    def test_page_missing_region_coords_explicit_ro_skips(self):
        """
        Test that a PageXML explicit reading order referencing a region without
        coordinates skips it gracefully.
        """
        doc = xml.XMLPage(self.page_doc_root / 'missing_coords_ro.xml', filetype='page')
        # The explicit RO references r1, r_nocoords, r2
        ro_line = doc.reading_orders['ro_test']
        self.assertEqual(ro_line['level'], 'line')
        # r_nocoords is skipped so l3 is not expanded from it
        self.assertIn('l1', ro_line['order'])
        self.assertIn('l2', ro_line['order'])
        self.assertIn('l4', ro_line['order'])
        self.assertNotIn('l3', ro_line['order'])

        ro_region = doc.reading_orders['ro_test:regions']
        self.assertEqual(ro_region['level'], 'region')
        self.assertNotIn('r_nocoords', ro_region['order'])
        self.assertEqual(ro_region['order'], ['r1', 'r2'])

    def test_page_missing_region_coords_to_container(self):
        """
        Test that to_container() works correctly when some PageXML regions
        lack coordinates.
        """
        doc = xml.XMLPage(self.page_doc_root / 'missing_coords_ro.xml', filetype='page')
        seg = doc.to_container()
        # All 4 lines should be present
        self.assertEqual(len(seg.lines), 4)
        line_ids = [l.id for l in seg.lines]
        self.assertIn('l3', line_ids)
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

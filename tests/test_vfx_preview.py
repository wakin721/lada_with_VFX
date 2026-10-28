# SPDX-License-Identifier: AGPL-3.0
import importlib.util
from pathlib import Path
from fractions import Fraction
import unittest

spec = importlib.util.spec_from_file_location('vfx_preview_test', Path(__file__).resolve().parents[1] / 'lada/utils/vfx_preview.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Backend:
    def prepare(self, frame, bgr):
        return frame

    def interpolate(self, previous, current, index):
        return (previous, current, index)

    def output(self, frame):
        return frame


class PreviewTests(unittest.TestCase):
    def processor(self, multiplier=2):
        processor = module.VfxPreviewProcessor.__new__(module.VfxPreviewProcessor)
        processor.backend = Backend()
        processor.multiplier = multiplier
        processor.previous = processor.previous_pts = None
        return processor

    def test_first_frame_has_no_reorder_window_latency(self):
        self.assertEqual(list(self.processor().frames('first', 0)), [('first', 0)])

    def test_vfr_intermediate_timestamps_are_exact(self):
        processor = self.processor(3)
        list(processor.frames('first', 100))
        result = list(processor.frames('second', 201))
        self.assertEqual([pts for frame, pts in result], [Fraction(401, 3), Fraction(502, 3), 201])

    def test_seek_resets_history_instead_of_interpolating_across_it(self):
        processor = self.processor()
        list(processor.frames('first', 100))
        self.assertEqual(list(processor.frames('seeked', 20)), [('seeked', 20)])

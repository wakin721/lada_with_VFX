# SPDX-License-Identifier: AGPL-3.0
import importlib.util
from pathlib import Path
import unittest
from fractions import Fraction

spec = importlib.util.spec_from_file_location('video_timing_test', Path(__file__).resolve().parents[1] / 'lada/utils/video_timing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimingTests(unittest.TestCase):
    def test_vfr_screen_recording_clock_is_not_frame_rate(self):
        self.assertEqual(module.select_frame_rate(dict(r_frame_rate='90000/1', avg_frame_rate='6210000/2595127'), 57.669478), Fraction(6210000, 2595127))

    def test_fractional_cfr_keeps_exact_rate(self):
        self.assertEqual(module.select_frame_rate(dict(r_frame_rate='30000/1001', avg_frame_rate='30000/1001')), Fraction(30000, 1001))

    def test_unknown_rates_use_frame_count_and_duration(self):
        self.assertEqual(module.select_frame_rate(dict(r_frame_rate='0/0', avg_frame_rate='0/0', nb_frames='120'), 5), 24)

    def test_vfr_nominal_rate_is_not_used_for_average(self):
        self.assertEqual(module.select_frame_rate(dict(r_frame_rate='60/1', avg_frame_rate='25/1')), 25)

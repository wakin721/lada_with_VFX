# SPDX-License-Identifier: AGPL-3.0
"""Timing/lifecycle tests run on ordinary CI runners without CUDA or GTK."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest

# Load the timing module without lada.utils.__init__ (which imports PyTorch).
spec = importlib.util.spec_from_file_location('lada_vfx_test', Path(__file__).resolve().parents[1] / 'lada/utils/vfx.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
VfxOptions, VfxVideoWriter = module.VfxOptions, module.VfxVideoWriter


class Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, frame, pts, bgr2rgb=False):
        self.frames.append((frame, pts))

    def release(self):
        self.closed = True


class Backend:
    def __init__(self):
        self.closed = False

    def prepare(self, frame, bgr):
        return frame[0]

    def interpolate(self, previous, current, index):
        return ('interpolated', previous, current, index)

    def output(self, frame):
        return frame

    def close(self):
        self.closed = True


class VfxTests(unittest.TestCase):
    def make_writer(self, multiplier=2, fps=25, time_base=Fraction(1, 1000)):
        writer, backend = Writer(), Backend()
        return VfxVideoWriter(writer, backend, VfxOptions(multiplier=multiplier), fps, time_base)

    def test_defaults_do_not_load_nvidia(self):
        self.assertFalse(VfxOptions().enabled)
        self.assertNotIn('nvvfx', sys.modules)

    def test_cfr_keeps_duration_and_frame_order(self):
        for multiplier in (2, 3, 8):
            with self.subTest(multiplier=multiplier):
                vfx = self.make_writer(multiplier)
                for i in range(5):
                    vfx.write([i], i * 40)
                vfx.release()
                frames = vfx.writer.frames
                self.assertEqual(len(frames), 5 * multiplier)
                self.assertEqual([pts for _, pts in frames], list(range(0, 200 * multiplier, 40)))
                self.assertEqual([frame for frame, _ in frames[::multiplier]], list(range(5)))
                duration = (frames[-1][1] + 40) * Fraction(1, 1000 * multiplier)
                self.assertEqual(duration, Fraction(5, 25))

    def test_vfr_interpolates_actual_pts_gaps(self):
        vfx = self.make_writer()
        for i, pts in enumerate([100, 145, 220]):
            vfx.write([i], pts)
        vfx.release()
        self.assertEqual([pts for _, pts in vfx.writer.frames], [200, 245, 290, 365, 440, 480])

    def test_fractional_fps_keeps_exact_ticks(self):
        vfx = self.make_writer(2, Fraction(30000, 1001), Fraction(1, 30000))
        vfx.write([0], 0)
        vfx.write([1], 1001)
        vfx.release()
        self.assertEqual([pts for _, pts in vfx.writer.frames], [0, 1001, 2002, 3003])

    def test_final_vfr_still_uses_actual_media_end_time(self):
        vfx = self.make_writer()
        vfx.end_pts = 2000
        vfx.write([0], 0)
        vfx.write([1], 40)
        vfx.release()
        self.assertEqual([pts for _, pts in vfx.writer.frames], [0, 40, 80, 2040])

    def test_reorder_pts_keeps_decoded_images_in_order(self):
        vfx = self.make_writer()
        for image, pts in zip([0, 1, 2], [80, 0, 40]):
            vfx.write([image], pts)
        vfx.release()
        self.assertEqual([frame for frame, _ in vfx.writer.frames[::2]], [0, 1, 2])
        self.assertEqual([pts for _, pts in vfx.writer.frames], [0, 40, 80, 120, 160, 200])

    def test_pause_resume_does_not_finalize_or_pad_midstream(self):
        vfx = self.make_writer()
        for i in range(20):
            vfx.write([i], i * 40)
        # A GUI pause keeps the same writer. Resume supplies only new source frames.
        for i in range(20, 30):
            vfx.write([i], i * 40)
        vfx.release()
        self.assertEqual(len(vfx.writer.frames), 60)
        self.assertEqual([frame for frame, _ in vfx.writer.frames[::2]], list(range(30)))

    def test_failure_still_closes_native_resources(self):
        vfx = self.make_writer()
        def fail(frame, bgr):
            raise RuntimeError('SDK failure')
        vfx.backend.prepare = fail
        vfx.write([0], 0)
        with self.assertRaisesRegex(RuntimeError, 'SDK failure'):
            vfx.release()
        self.assertTrue(vfx.writer.closed)
        self.assertTrue(vfx.backend.closed)
        vfx.release()  # idempotent after failure

    def test_cancel_discards_pending_frames(self):
        vfx = self.make_writer()
        vfx.write([0], 0)
        vfx.release(finalize=False)
        self.assertEqual(vfx.writer.frames, [])
        self.assertTrue(vfx.backend.closed)

    def test_invalid_options_are_rejected(self):
        for kwargs in [dict(scale=0), dict(scale=2.0), dict(multiplier=9), dict(denoise='bad')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VfxOptions(**kwargs)


if __name__ == '__main__':
    unittest.main()

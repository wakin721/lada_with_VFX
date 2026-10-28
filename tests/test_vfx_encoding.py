# SPDX-License-Identifier: AGPL-3.0
import importlib.util
from pathlib import Path
import sys
from threading import Event
import unittest

spec = importlib.util.spec_from_file_location('vfx_encoding_test', Path(__file__).resolve().parents[1] / 'lada/utils/vfx_encoding.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, frame, pts, bgr2rgb=False):
        self.frames.append((frame, pts))

    def release(self):
        self.closed = True


class EncodingTests(unittest.TestCase):
    def test_order_and_all_frames_survive_bounded_queue(self):
        writer = Writer()
        worker = module.EncodingWorker(writer, capacity=1)
        for i in range(20):
            worker.write(i, i * 40)
        worker.release()
        self.assertEqual(writer.frames, [(i, i * 40) for i in range(20)])
        self.assertTrue(writer.closed)

    def test_errors_reach_producer_and_release_still_closes(self):
        writer = Writer()
        failed = Event()
        proceed = Event()
        def write(*args, **kwargs):
            proceed.wait(2)
            failed.set()
            raise ValueError('codec failed')
        writer.write = write
        worker = module.EncodingWorker(writer, capacity=1)
        worker.write('frame', 0)
        proceed.set()
        self.assertTrue(failed.wait(2))
        with self.assertRaisesRegex(RuntimeError, 'codec failed'):
            worker.release()
        self.assertTrue(writer.closed)
        self.assertFalse(worker.thread.is_alive())

    def test_download_is_resolved_in_encoding_thread(self):
        calls = []
        class Ready:
            def synchronize(self):
                calls.append('ready')
        class Host:
            def numpy(self):
                calls.append('numpy')
                return 'pixels'
        frame = module.DownloadedFrame(Host(), Ready(), object())
        writer = Writer()
        worker = module.EncodingWorker(writer)
        worker.write(frame, 0)
        worker.release()
        self.assertEqual(calls, ['ready', 'numpy'])
        self.assertEqual(writer.frames, [('pixels', 0)])
        self.assertIsNone(frame.device_image)

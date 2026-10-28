# SPDX-License-Identifier: AGPL-3.0
"""Bounded export handoff: overlap CUDA download with CPU conversion/encoding."""
from dataclasses import dataclass
from queue import Queue, Full
from threading import Thread
import time


@dataclass
class DownloadedFrame:
    host: object
    ready: object
    device_image: object

    def resolve(self):
        self.ready.synchronize()
        # Retain the CUDA allocation until its asynchronous copy has finished.
        self.device_image = None
        return self.host.numpy()


class EncodingWorker:
    def __init__(self, writer, capacity=2):
        self.writer = writer
        self.queue = Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.frames = 0
        self.encode_seconds = 0.0
        self.wait_seconds = 0.0
        self.thread = Thread(target=self._run, name='VFX video encoder', daemon=True)
        self.thread.start()

    def _raise_error(self):
        if self.error is not None:
            raise RuntimeError(f'VFX encoding failed: {self.error}') from self.error

    def write(self, frame, pts=None, bgr2rgb=False):
        if self.closed:
            raise RuntimeError('VFX encoder is already closed')
        self._raise_error()
        started = time.perf_counter()
        while True:
            try:
                self.queue.put((frame, pts, bgr2rgb), timeout=0.1)
                break
            except Full:
                self._raise_error()
        self.wait_seconds += time.perf_counter() - started
        self._raise_error()

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        break
                    if self.error is not None:
                        continue
                    frame, pts, bgr = item
                    if isinstance(frame, DownloadedFrame):
                        frame = frame.resolve()
                    started = time.perf_counter()
                    self.writer.write(frame, pts, bgr2rgb=bgr)
                    self.encode_seconds += time.perf_counter() - started
                    self.frames += 1
                except BaseException as error:
                    self.error = self.error or error
                finally:
                    self.queue.task_done()
                    # Release image storage before blocking on the next get().
                    item = None
                    frame = None
        finally:
            try:
                self.writer.release()
            except BaseException as error:
                self.error = self.error or error

    def release(self):
        if self.closed:
            self._raise_error()
            return
        self.closed = True
        self.queue.put(None)
        self.thread.join()
        self._raise_error()

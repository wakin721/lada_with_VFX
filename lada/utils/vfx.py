# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Optional NVIDIA VFX export stage. Importing this module never loads CUDA."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import heapq


@dataclass(frozen=True)
class VfxOptions:
    scale: int = 1
    denoise: str = 'off'
    multiplier: int = 1
    quality: str = 'HIGH'
    frame_quality: str = 'MEDIUM'

    def __post_init__(self):
        if type(self.scale) is not int or self.scale not in (1, 2, 3, 4):
            raise ValueError('VFX scale must be 1, 2, 3 or 4')
        if self.denoise not in ('off', 'LOW', 'MEDIUM', 'HIGH', 'ULTRA'):
            raise ValueError('Unknown VFX denoise level')
        if type(self.multiplier) is not int or not 1 <= self.multiplier <= 8:
            raise ValueError('VFX frame multiplier must be between 1 and 8')
        if self.quality not in ('LOW', 'MEDIUM', 'HIGH', 'ULTRA'):
            raise ValueError('Unknown VFX super resolution quality')
        if self.frame_quality not in ('LOW', 'MEDIUM', 'HIGH'):
            raise ValueError('Unknown VFX frame generation quality')

    @property
    def enabled(self):
        return self.scale != 1 or self.denoise != 'off' or self.multiplier != 1

    @classmethod
    def from_config(cls, config):
        return cls(config.vfx_scale, config.vfx_denoise, config.vfx_multiplier,
                   config.vfx_quality, config.vfx_frame_quality)


class VfxVideoWriter:
    """Keep input timing, including VFR, and retain interpolation state on pause.

    A finer time base makes intermediate PTS integral. Like VideoWriter, reorder
    a small PTS window without reordering the decoded images. Only release()
    pads the last source-frame interval; pausing does not finalize the stream.
    """

    def __init__(self, writer, backend, options, fps, time_base):
        self.writer = writer
        self.backend = backend
        self.options = options
        self.time_base = Fraction(time_base)
        self.frame_interval = Fraction(1, 1) / Fraction(fps) / self.time_base
        self.previous = None
        self.previous_pts = None
        self.frames = deque()
        self.timestamps = []
        self.timestamp_set = set()
        self.closed = False
        self.failed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release(finalize=exc_type is None)

    def write(self, frame, frame_pts=None, bgr2rgb=False):
        if self.closed:
            raise RuntimeError('VFX writer is already closed')
        if frame_pts is None:
            raise ValueError('VFX export requires source timestamps')
        if frame_pts in self.timestamp_set:
            return
        # Restoration workers may reuse their buffers while the PTS window fills.
        owned_frame = frame.clone() if hasattr(frame, 'clone') else frame.copy()
        self.frames.append((owned_frame, bgr2rgb))
        heapq.heappush(self.timestamps, frame_pts)
        self.timestamp_set.add(frame_pts)
        if len(self.frames) > 15:
            self._drain_one()

    def _emit(self, frame, pts):
        result = self.backend.output(frame)
        self.writer.write(result, pts, bgr2rgb=False)

    def _drain_one(self):
        frame, bgr = self.frames.popleft()
        pts = heapq.heappop(self.timestamps)
        self.timestamp_set.remove(pts)
        if self.previous_pts is not None and pts <= self.previous_pts:
            self.failed = True
            raise ValueError('VFX received non-increasing video timestamps')
        try:
            current = self.backend.prepare(frame, bgr)
            multiplier = self.options.multiplier
            if self.previous is not None and multiplier > 1:
                for index in range(1, multiplier):
                    generated = self.backend.interpolate(self.previous, current, index)
                    self._emit(generated, self.previous_pts * multiplier + (pts - self.previous_pts) * index)
            self._emit(current, pts * multiplier)
            self.previous, self.previous_pts = current, pts
        except BaseException:
            self.failed = True
            raise

    def release(self, finalize=True):
        if self.closed:
            return
        self.closed = True
        try:
            if finalize and not self.failed:
                while self.frames:
                    self._drain_one()
                # Hold the final image for the nominal final source-frame duration.
                # This preserves CFR duration (N input frames -> N*M output frames).
                if self.previous is not None:
                    for index in range(1, self.options.multiplier):
                        pts = self.previous_pts * self.options.multiplier + round(self.frame_interval * index)
                        self._emit(self.previous, pts)
        finally:
            try:
                self.writer.release()
            finally:
                self.backend.close()
                self.frames.clear()
                self.previous = None


def create_video_writer(output_path, width, height, fps, encoder, encoder_options,
                        time_base=None, mp4_fast_start=False, *, vfx=None, device='cuda:0'):
    from lada.utils.video_utils import VideoWriter

    options = vfx or VfxOptions()
    if not options.enabled:
        return VideoWriter(output_path, width, height, fps, encoder, encoder_options,
                           time_base=time_base, mp4_fast_start=mp4_fast_start)
    if time_base is None or Fraction(time_base) <= 0 or Fraction(fps) <= 0:
        raise ValueError('VFX export requires a valid video time base and frame rate')
    from lada.utils.vfx_nvidia import NvidiaVfxBackend

    backend = NvidiaVfxBackend(options, width, height, device)
    try:
        writer = VideoWriter(output_path, width * options.scale, height * options.scale,
                             fps * options.multiplier, encoder, encoder_options,
                             time_base=Fraction(time_base) / options.multiplier,
                             mp4_fast_start=mp4_fast_start)
    except BaseException:
        backend.close()
        raise
    return VfxVideoWriter(writer, backend, options, fps, time_base)


def release_video_writer(writer, *, finalize=True):
    if writer is not None:
        if isinstance(writer, VfxVideoWriter):
            writer.release(finalize=finalize)
        else:
            writer.release()

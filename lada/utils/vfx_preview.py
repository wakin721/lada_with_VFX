# SPDX-License-Identifier: AGPL-3.0
"""Streaming VFX: source timestamps and GPU tensors stay intact until display."""
from fractions import Fraction


class VfxPreviewProcessor:
    def __init__(self, options, width, height, device):
        from lada.utils.vfx_nvidia import NvidiaVfxBackend
        self.backend = NvidiaVfxBackend(options, width, height, device, gpu_output=True)
        self.multiplier = options.multiplier
        self.previous = None
        self.previous_pts = None

    def frames(self, frame, pts):
        current = self.backend.prepare(frame, bgr=True)
        if self.previous is not None:
            if pts <= self.previous_pts:
                # Discontinuous timestamps reset interpolation history (e.g. seek).
                self.previous = None
            else:
                for index in range(1, self.multiplier):
                    generated = self.backend.interpolate(self.previous, current, index)
                    timestamp = self.previous_pts + Fraction((pts - self.previous_pts) * index, self.multiplier)
                    yield self.backend.output(generated), timestamp
        yield self.backend.output(current), pts
        self.previous, self.previous_pts = current, pts

    def close(self):
        self.previous = None
        self.backend.close()

# SPDX-License-Identifier: AGPL-3.0
"""Exercise VFX on synthetic frames; no restoration weights or user media."""
import argparse
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='vfx-smoke.mp4')
    args = parser.parse_args()
    path = Path(args.output).resolve()
    if path.exists():
        parser.error(f'Refusing to overwrite {path}')

    import av
    import numpy as np
    from lada.utils.vfx import VfxOptions, create_video_writer

    options = VfxOptions(scale=2, denoise='LOW', multiplier=2)
    width, height, count, fps = 640, 360, 12, 24
    with create_video_writer(str(path), width, height, Fraction(fps), 'libx264', '-crf 18',
                             time_base=Fraction(1, fps), vfx=options, device='cuda:0') as vfx:
        for index in range(count):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:, :, 2] = np.linspace(0, 180, width, dtype=np.uint8)
            x = 20 + 20 * index
            image[100:200, x:x + 80] = (240, 120, 40)
            vfx.write(image, index, bgr2rgb=False)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert len(frames) == count * options.multiplier, len(frames)
        assert (stream.width, stream.height) == (width * options.scale, height * options.scale)
        assert stream.average_rate == fps * options.multiplier, stream.average_rate
        assert abs(float(stream.duration * stream.time_base) - count / fps) < 1 / (fps * options.multiplier)
        assert all(a.pts < b.pts for a, b in zip(frames, frames[1:]))
    print(f'PASS: {len(frames)} frames, 1280x720, 48 FPS, 0.5 seconds: {path}')


if __name__ == '__main__':
    main()

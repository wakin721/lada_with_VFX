# SPDX-License-Identifier: AGPL-3.0
"""Encode sparse VFR timestamps and verify presentation duration without a GPU."""
from fractions import Fraction
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import av
    import numpy as np
    from lada.utils.vfx import create_video_writer

    timestamps = [0, 87315, 93028, 4074879]
    time_base = Fraction(1, 90000)
    duration = Fraction('57.669478')
    with tempfile.TemporaryDirectory(prefix='lada-vfr-check-') as directory:
        output = Path(directory) / 'sparse-vfr.mp4'
        with create_video_writer(str(output), 64, 64, Fraction(6210000, 2595127),
                                 'libx264', '-crf 28', time_base=time_base,
                                 duration=duration, variable_frame_rate=True) as writer:
            for index, timestamp in enumerate(timestamps):
                image = np.full((64, 64, 3), index * 50, dtype=np.uint8)
                writer.write(image, timestamp)
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            frames = list(container.decode(stream))
            assert len(frames) == len(timestamps)
            for frame, timestamp in zip(frames, timestamps):
                assert abs(frame.pts * frame.time_base - timestamp * time_base) <= time_base
            assert abs(stream.duration * stream.time_base - duration) < Fraction(1, 100)
    print('PASS: sparse VFR timestamps and 57.669478-second final presentation duration')


if __name__ == '__main__':
    main()

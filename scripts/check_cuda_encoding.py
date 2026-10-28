# SPDX-License-Identifier: AGPL-3.0
"""Hardware integration check for CUDA->NVENC, including stream ownership/VFR.

Run in the NVIDIA VFX environment: python scripts/check_cuda_encoding.py OUTPUT_DIR
"""
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(output_dir):
    import av
    import torch
    from lada.utils.cuda_video_writer import CudaVideoWriter, CudaFrame
    from lada.utils.vfx_encoding import EncodingWorker
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for codec in ('h264_nvenc', 'hevc_nvenc'):
        for vfr in (False, True):
            for asynchronous in (False, True):
                path = output_dir / f'{codec}-{vfr}-{asynchronous}.mp4'
                pts = [i * 3000 for i in range(24)]
                if vfr:
                    pts = [p + (60000 if i >= 12 else 0) for i, p in enumerate(pts)]
                end = Fraction(pts[-1] + (90000 if vfr else 3000), 90000)
                native = CudaVideoWriter(str(path), 320, 240, Fraction(30), codec,
                    '-preset p4 -qp 20 -bf 3', Fraction(1, 90000), variable_frame_rate=vfr,
                    mp4_fast_start=asynchronous)
                native.set_end_time(end)
                writer = EncodingWorker(native, capacity=1) if asynchronous else native
                producer = torch.cuda.Stream()
                # Alternating saturated colors expose stale/reused CUDA buffers.
                for i, timestamp in enumerate(pts):
                    with torch.cuda.stream(producer):
                        image = torch.zeros((240, 320, 3), device='cuda', dtype=torch.uint8)
                        image[..., i % 3] = 200
                        frame = CudaFrame.capture(image)
                    writer.write(frame, timestamp)
                writer.release()
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    frames = list(container.decode(stream))
                    assert len(frames) == 24, (codec, vfr, len(frames))
                    assert [f.pts * f.time_base for f in frames] == [p * Fraction(1, 90000) for p in pts]
                    assert abs(stream.duration * stream.time_base - end) <= Fraction(1, 90000), (stream.duration, stream.time_base, end)
                    for i, frame in enumerate(frames):
                        colors = frame.to_ndarray(format='rgb24').mean(axis=(0, 1))
                        assert abs(colors[i % 3] - 200) < 5 and sum(colors) < 210, (i, colors)
                    results.append(dict(codec=codec, vfr=vfr, asynchronous=asynchronous,
                        frames=len(frames), duration=float(end), b_frames=native.num_b_frames))
    return results


if __name__ == '__main__':
    print(check(sys.argv[1]))

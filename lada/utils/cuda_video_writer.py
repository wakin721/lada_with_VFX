# SPDX-License-Identifier: AGPL-3.0
"""CUDA-resident NVENC input; only compressed packets reach host memory.

PyNvVideoCodec 2.2.3 supplies packet timestamps and NV_ENC_PIC_PARAMS.
Earlier releases must not be used: their byte-only results cannot preserve VFR.
"""
from dataclasses import dataclass
from fractions import Fraction
import io
import shlex


def encoder_config(encoder, options, variable_frame_rate=False):
    """Translate the deliberately limited CUDA presets, never ignore options."""
    if encoder not in ('hevc_nvenc', 'h264_nvenc'):
        raise ValueError('CUDA input requires H.264 or HEVC NVENC')
    tokens = shlex.split(options)
    if len(tokens) % 2:
        raise ValueError('Encoder options require key/value pairs')
    settings = dict(zip((s.lstrip('-') for s in tokens[::2]), tokens[1::2]))
    allowed = {'cuda-input', 'preset', 'tune', 'rc', 'qp', 'bf', 'rc-lookahead', 'g'}
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f'CUDA encoder does not support these options: {sorted(unknown)}')
    tune = {'hq': 'high_quality', 'll': 'low_latency', 'ull': 'ultra_low_latency'}
    if settings.get('tune', 'hq') not in tune or settings.get('rc', 'constqp') != 'constqp':
        raise ValueError('CUDA presets currently support constant QP with hq/ll/ull tuning')
    preset = settings.get('preset', 'p4').upper()
    if preset not in {f'P{i}' for i in range(1, 8)}:
        raise ValueError('CUDA encoder preset must be P1 through P7')
    qp = int(settings.get('qp', '23'))
    if not 0 <= qp <= 51:
        raise ValueError('CUDA encoder QP must be between 0 and 51')
    result = dict(codec=encoder.removesuffix('_nvenc'), preset=preset,
                  tuning_info=tune[settings.get('tune', 'hq')], rc='constqp', constqp=qp,
                  bf=0 if variable_frame_rate else int(settings.get('bf', '0')),
                  lookahead=int(settings.get('rc-lookahead', '0')))
    if 'g' in settings:
        result['gop'] = int(settings['g'])
    return result


def uses_cuda_input(options):
    tokens = shlex.split(options)
    return any(tokens[i] == '-cuda-input' and tokens[i + 1] == '1'
               for i in range(0, len(tokens) - 1, 2))


@dataclass
class CudaFrame:
    image: object
    ready: object

    @classmethod
    def capture(cls, image):
        import torch
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(image.device))
        return cls(image, ready)


class _CudaArray:
    def __init__(self, image):
        self.image = image
        self.__cuda_array_interface__ = {
            'shape': tuple(image.shape), 'strides': tuple(image.stride()),
            'typestr': '|u1', 'data': (image.data_ptr(), False), 'version': 3,
        }

    def cuda(self):
        return self


class CudaPassthroughBackend:
    def __init__(self, device):
        self.device = device

    def prepare(self, frame, bgr):
        import torch
        image = torch.as_tensor(frame, device=self.device)
        return image[..., [2, 1, 0]].contiguous() if bgr else image

    def output(self, image):
        return CudaFrame.capture(image)

    def close(self):
        pass


class CudaVideoWriter:
    def __init__(self, output_path, width, height, fps, encoder, encoder_options,
                 time_base, device='cuda:0', mp4_fast_start=False, variable_frame_rate=False):
        import importlib.metadata
        import torch
        import PyNvVideoCodec as nvc
        if importlib.metadata.version('PyNvVideoCodec') != '2.2.3':
            raise RuntimeError('CUDA video input requires PyNvVideoCodec==2.2.3')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA video input requires an NVIDIA CUDA device')
        self.device = torch.device('cuda', self.device.index if self.device.index is not None else torch.cuda.current_device())
        self.torch, self.nvc = torch, nvc
        self.width, self.height = width, height
        self.fps, self.time_base = Fraction(fps), Fraction(time_base)
        self.output_path, self.fast_start = str(output_path), mp4_fast_start
        self.config = encoder_config(encoder, encoder_options, variable_frame_rate)
        self.codec = self.config['codec']
        self.closed = False
        self.encoder = self.output_container = self.pending = None
        self.timing = {}
        self.submitted = self.packets = 0
        self.first_pts = None
        self.end_time = None
        self.variable_frame_rate = variable_frame_rate
        with torch.cuda.device(self.device):
            caps = nvc.GetEncoderCaps(gpuid=self.device.index, codec=self.codec)
            if width % 2 or height % 2 or not caps['width_min'] <= width <= caps['width_max'] or not caps['height_min'] <= height <= caps['height_max']:
                raise ValueError(f'CUDA NVENC cannot encode {width}x{height}; maximum {caps["width_max"]}x{caps["height_max"]}. Reduce VFX scale.')
            # A non-default stream is essential: SDK stream=0 creates its own
            # stream, so queued PyTorch writes could otherwise race NVENC reads.
            self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                self.encoder = nvc.CreateEncoder(width, height, 'ABGR', False,
                    gpu_id=self.device.index, cudastream=self.stream.cuda_stream,
                    fps=max(1, round(self.fps)), **self.config)
                params = self.encoder.GetEncodeReconfigureParams()
                params.frameRateNum, params.frameRateDen = self.fps.numerator, self.fps.denominator
                self.encoder.Reconfigure(params)
                self.num_b_frames = self.encoder.GetNumBFrames()
                if variable_frame_rate and self.num_b_frames:
                    raise RuntimeError('VFR CUDA export requires B frames disabled')
                self.sequence = self.encoder.GetSequenceParams()

    def set_end_time(self, end_time):
        self.end_time = Fraction(end_time)

    def write(self, frame, frame_pts=None, bgr2rgb=False):
        if self.closed:
            raise RuntimeError('CUDA encoder is closed')
        if frame_pts is None:
            raise ValueError('CUDA export requires source timestamps')
        if not isinstance(frame, CudaFrame):
            if not getattr(frame, 'is_cuda', False):
                raise ValueError('CUDA encoder requires CUDA images')
            frame = CudaFrame.capture(frame)
        if self.pending is not None:
            previous, pts, bgr = self.pending
            if frame_pts <= pts:
                raise ValueError('CUDA export timestamps must increase')
            self._encode(previous, pts, frame_pts - pts, bgr)
        self.pending = (frame, frame_pts, bgr2rgb)

    def _encode(self, frame, pts, duration, bgr):
        torch = self.torch
        image = frame.image
        if tuple(image.shape) != (self.height, self.width, 3) or image.dtype != torch.uint8 or image.device != self.device:
            raise ValueError('CUDA encoder expects an 8-bit HWC RGB image on the selected device')
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(frame.ready)
            rgba = torch.empty((self.height, self.width, 4), dtype=torch.uint8, device=self.device)
            rgba[..., :3] = image[..., [2, 1, 0]] if bgr else image
            rgba[..., 3] = 255
            params = self.nvc.NV_ENC_PIC_PARAMS()
            params.inputTimeStamp = params.frameIdx = self.submitted
            params.inputDuration = max(1, round(duration))
            self.timing[self.submitted] = (pts, max(1, round(duration)))
            if self.first_pts is None:
                self.first_pts = pts
            self.submitted += 1
            packets = self.encoder.Encode(_CudaArray(rgba), params)
            # Encode queues a device-to-device copy. Keep both allocations alive
            # until it completes; synchronize this stream, never the whole GPU.
            self.stream.synchronize()
        self._mux(packets)

    def _mux(self, packets):
        import av
        for encoded in packets:
            pts, duration = self.timing.pop(encoded['timestamp'])
            if self.output_container is None:
                options = {'movflags': '+frag_keyframe+empty_moov+delay_moov', 'avoid_negative_ts': 'disabled'} if self.fast_start and self.output_path.lower().endswith(('.mp4', '.mov')) else {}
                self.output_container = av.open(self.output_path, 'w', options=options)
                # Parse headers together with the first picture so older PyAV
                # versions also obtain dimensions without opening an encoder.
                with av.open(io.BytesIO(self.sequence + encoded['data']), format=self.codec) as template:
                    self.video_stream = self.output_container.add_stream_from_template(template.streams.video[0], opaque=True)
                self.video_stream.time_base = self.time_base
            packet = av.Packet(encoded['data'])
            packet.stream, packet.time_base = self.video_stream, self.time_base
            packet.pts, packet.duration = pts, duration
            if self.num_b_frames:
                packet.dts = self.first_pts + round(Fraction(self.packets - self.num_b_frames, 1) / self.fps / self.time_base)
            else:
                packet.dts = pts
            packet.is_keyframe = encoded['picture_type'] in (2, 3)
            self.output_container.mux(packet)
            self.packets += 1

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.pending is not None:
                frame, pts, bgr = self.pending
                duration = (self.end_time / self.time_base - pts) if self.end_time is not None else Fraction(1, 1) / self.fps / self.time_base
                self._encode(frame, pts, max(1, round(duration)), bgr)
                self.pending = None
            with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.stream):
                self._mux(self.encoder.EndEncode())
                self.stream.synchronize()
            if self.timing:
                raise RuntimeError('NVENC did not return every submitted frame')
        finally:
            if self.output_container is not None:
                self.output_container.close()
            self.encoder = self.pending = None
            self.timing.clear()

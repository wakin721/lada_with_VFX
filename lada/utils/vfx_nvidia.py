# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""NVIDIA nvidia-vfx 0.2 backend; loaded only when an export enables VFX."""


class NvidiaVfxBackend:
    def __init__(self, options, width, height, device, *, gpu_output=False):
        import torch
        try:
            from nvvfx import VideoSuperRes, VideoFrameGeneration
        except (ImportError, OSError) as exc:
            raise RuntimeError('NVIDIA VFX is unavailable. Install nvidia-vfx==0.2.0.0 '
                               'or use the Windows VFX build. See docs/vfx.md.') from exc

        self.torch = torch
        self.gpu_output = gpu_output
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('NVIDIA VFX requires an NVIDIA CUDA device; select it in settings.')
        self.device = torch.device('cuda', self.device.index if self.device.index is not None else torch.cuda.current_device())
        if options.multiplier > 1 and torch.cuda.get_device_capability(self.device) not in ((8, 9), (9, 0), (10, 0), (12, 0), (12, 1)):
            raise RuntimeError('NVIDIA frame generation requires Ada/Blackwell on Windows, '
                               'or Ada/Blackwell/Hopper on Linux.')
        # Hopper is supported for VFG only on Linux.
        import sys
        if options.multiplier > 1 and sys.platform == 'win32' and torch.cuda.get_device_capability(self.device) == (9, 0):
            raise RuntimeError('NVIDIA frame generation on Windows requires Ada or Blackwell.')
        self.width, self.height = width, height
        self.denoiser = self.interpolator = self.upscaler = None
        try:
            with torch.cuda.device(self.device):
                if options.denoise != 'off':
                    self.denoiser = VideoSuperRes(device=self.device.index,
                        quality=VideoSuperRes.QualityLevel['DENOISE_' + options.denoise])
                    self._load_spatial(self.denoiser, width, height, width, height)
                if options.multiplier > 1:
                    self.interpolator = VideoFrameGeneration(device=self.device.index,
                        mode=VideoFrameGeneration.Mode[options.frame_quality],
                        automatic_shot_change_detection_enabled=True)
                    self.interpolator.input_width = width
                    self.interpolator.input_height = height
                    self.interpolator.frame_multiplier = options.multiplier
                    self.interpolator.load()
                if options.scale > 1:
                    self.upscaler = VideoSuperRes(device=self.device.index,
                        quality=VideoSuperRes.QualityLevel[options.quality])
                    self._load_spatial(self.upscaler, width, height, width * options.scale, height * options.scale)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _load_spatial(effect, width, height, output_width, output_height):
        effect.input_width, effect.input_height = width, height
        effect.output_width, effect.output_height = output_width, output_height
        effect.load()

    def _run(self, effect, *frames, **kwargs):
        with self.torch.cuda.device(self.device):
            stream = self.torch.cuda.current_stream(self.device)
            result = effect.run(*frames, stream_ptr=stream.cuda_stream, **kwargs)
            # Effects reuse their output buffers. Own the image before another run.
            return self.torch.from_dlpack(result.image).clone()

    def prepare(self, frame, bgr):
        torch = self.torch
        with torch.cuda.device(self.device):
            tensor = torch.as_tensor(frame, device=self.device)
            if tuple(tensor.shape) != (self.height, self.width, 3) or tensor.dtype != torch.uint8:
                raise ValueError('VFX export expects an 8-bit HWC image with three channels')
            if bgr:
                tensor = tensor[..., [2, 1, 0]]
            tensor = tensor.permute(2, 0, 1).to(torch.float32).div_(255).contiguous()
            return self._run(self.denoiser, tensor) if self.denoiser else tensor

    def interpolate(self, previous, current, index):
        return self._run(self.interpolator, previous, current, frame_index=index)

    def output(self, frame):
        torch = self.torch
        with torch.cuda.device(self.device):
            result = self._run(self.upscaler, frame) if self.upscaler else frame
            result = result.mul(255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            if self.gpu_output:
                # GStreamer's appsrc consumes BGR. Convert before the sole D2H copy.
                return result[..., [2, 1, 0]].contiguous()
            return result.contiguous().cpu().numpy()

    def close(self):
        # Attempt every close even if a runtime reports an error during cleanup.
        first_error = None
        for name in ('upscaler', 'interpolator', 'denoiser'):
            effect = getattr(self, name, None)
            if effect is not None:
                setattr(self, name, None)
                try:
                    effect.close()
                except Exception as exc:
                    first_error = first_error or exc
        if first_error is not None:
            raise first_error

# NVIDIA VFX export

This fork adds optional NVIDIA Video Super Resolution, same-resolution denoising,
and Video Frame Generation to the CLI and GUI **export** path. Playback previews
continue to use the normal pipeline. All VFX options are off by default.

The processing order is denoise -> frame generation -> super resolution -> encoder.
Frame generation runs at the source resolution. Original frames and generated
frames both pass through super resolution. Shot-change detection is enabled to
avoid interpolation across cuts.

## Requirements

- Windows x64 with a supported NVIDIA Tensor Core GPU and driver 570.65 or newer.
- Video frame generation additionally requires Ada or Blackwell (RTX 40/50 series).
- NVIDIA CUDA must be selected as the device. CPU, Intel and AMD cannot run VFX.
- The implementation targets `nvidia-vfx==0.2.0.0`, including its bundled VFX 1.3 runtime.
- Source installations: install the existing Lada NVIDIA dependencies first, then
  `uv pip install --index-strategy unsafe-best-match -r packaging/windows/requirements-vfx.txt`.
  The portable Windows VFX build already includes these libraries.

No NGC API key is required for the bundled Python wheel. NVIDIA license notices
are included in the package's distribution metadata. The VFX runtime remains
subject to NVIDIA's terms; Lada's source license is unchanged.

## GUI

Open the Export settings sidebar and expand **NVIDIA VFX**. Set super resolution
to 2x/3x/4x, choose a denoise level, and select a frame-rate multiplier (2x–8x).
Each feature can be enabled independently. Settings persist with the normal
configuration and are locked while export is running.

A practical starting point for an RTX 5070 Ti is 2x super resolution, LOW denoise,
2x frame rate, HIGH super-resolution quality and MEDIUM frame-generation quality.
This is a starting preset, not a performance guarantee. Try a short ordinary video
before exporting long files. If GPU memory is exhausted, reduce clip length,
scale, or quality. VFX and restoration share VRAM.

## CLI

```powershell
lada-cli --input input.mp4 --output output.mp4 --encoding-preset hevc-nvidia-gpu-hq --vfx-scale 2 --vfx-denoise LOW --vfx-frame-multiplier 2
```

Additional switches:

| Switch | Values | Default |
|---|---|---|
| `--vfx-scale` | 1, 2, 3, 4 | 1 (off) |
| `--vfx-denoise` | off, LOW, MEDIUM, HIGH, ULTRA | off |
| `--vfx-frame-multiplier` | 1–8 | 1 (off) |
| `--vfx-quality` | LOW, MEDIUM, HIGH, ULTRA | HIGH |
| `--vfx-frame-quality` | LOW, MEDIUM, HIGH | MEDIUM |

Frame rates are multiplied exactly: 29.97 becomes 59.94 at 2x, rather than 60.
Input PTS gaps are used for variable-frame-rate interpolation. The last image is
held over one nominal source-frame interval to preserve duration for CFR input;
the last duration of VFR input is estimated from the nominal rate. Audio goes
through the existing muxing path. Pause/resume retains interpolation state.

The pipeline currently uses the existing 8-bit RGB export path; it does not add
HDR or 10-bit export. Enabling VFX adds processing work, and does not accelerate
the existing restoration models. Real-time playback performance is not promised.

## GitHub Actions

The **Windows NVIDIA VFX** workflow runs timing/lifecycle tests, builds Windows
CLI and GTK GUI executables, bundles VFX DLLs and license notices, and uploads a
portable `.7z` artifact with checksums. Open Actions, select this workflow, and
choose **Run workflow** to rebuild. Uncheck `build_gui` for a CLI-only package.
Pushes to `main` or `vfx/**` also build automatically.

Hosted GitHub runners do not have a supported RTX GPU. CI verifies Python
compilation, timing logic and packaging, not GPU inference or visual quality.
The first GTK build can take a long time; later builds reuse the GTK cache.

For a local GPU smoke test without restoration model weights:

```powershell
python scripts/vfx_smoke.py --output vfx-smoke.mp4
```

The test creates synthetic moving shapes, runs all three VFX effects and writes
a short file. It does not read any user videos.

## References

- [NVIDIA VFX Python samples](https://github.com/NVIDIA-Maxine/nvidia-vfx-python-samples)
- [NVIDIA VFX Python package](https://pypi.org/project/nvidia-vfx/)
- [VSR requirements and modes](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html)
- [VFG requirements and parameters](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoFrameGeneration.html)

# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import threading
import time

import torch
from gi.repository import Gst, GstApp, GObject

from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerProvider
from lada.utils import video_utils, VideoMetadata, threading_utils
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, StopMarker, EofMarker, ErrorMarker

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class FrameRestorerAppSrc(GstApp.AppSrc):
    GST_PLUGIN_NAME = 'framerestorerappsrc'

    __gstmetadata__ = ('FrameRestorerAppSrc','Src', 'FrameRestorer AppSrc element', 'Lada Authors')

    __gsttemplates__ = (
        Gst.PadTemplate.new("src",
                            Gst.PadDirection.SRC,
                            Gst.PadPresence.ALWAYS,
                            Gst.Caps.new_any()),
    )

    __gproperties__ = {
        "frame-restorer-provider": (GObject.TYPE_PYOBJECT,
                          "FrameRestorerProvider",
                          "Frame restorer provider object to get a FrameRestorer instance",
                          GObject.ParamFlags.READWRITE
                          ),
        "video-metadata": (GObject.TYPE_PYOBJECT,
                          "VideoMetadata",
                          "Metadata of the video file that should be restored by FrameRestorer",
                          GObject.ParamFlags.READWRITE
                          )
    }

    def __init__(self):
        super().__init__()

        self.video_metadata: VideoMetadata | None = None
        self.cpu_frame: torch.Tensor | None = None
        self.vfx_processor = None
        self.pending_display_buffer = None

        self.frame_restorer: FrameRestorer | None = None
        self.frame_restorer_provider: FrameRestorerProvider | None = None
        self.frame_restorer_lock: threading.Lock = threading.Lock()


        self.appsource_thread: threading.Thread | None = None
        self.appsource_thread_should_be_running: bool = False # Variable controlling state of thread. False if stop or shutdown requested or EOF
        self.appsource_thread_stop_requested = False # Variable controlling state of thread. Set based on enough-data / need-data states to trigger start/stop.
        self.appsource_thread_shutdown_requested = False # Variable controlling state of thread. Forced shutdown which overwrites appsource_thread_stop_requested. True if element state was set to NULL.
        self.appsource_thread_eof = False # Variable controlling state of thread. Set if FrameRestorer gave out last frame / EOF. Unset if getting seek request.

        self.appsrc_lock: threading.Lock = threading.Lock()

        self.frame_duration_ns: float = 0
        self.current_timestamp_ns = 0

        self.set_property('is-live', False)
        self.set_property('emit-signals', True)
        self.set_property('stream-type', GstApp.AppStreamType.SEEKABLE)
        self.set_property('format', Gst.Format.TIME)
        self.set_property('max-buffers', 5) # doesn't need to be much as we're using this AppSrc with a Queue
        self.set_property('max-bytes', 0)
        self.set_property('block', False)

        self.connect('need-data', self._on_need_data)
        self.connect('enough-data', self._on_enough_data)
        self.connect('seek-data', self._on_seek_data)

    @GObject.Signal(name='processing-error', arg_types=(str,))
    def processing_error(self, message):
        pass

    def _report_error(self, error):
        logger.error('Preview processing failed: %s', error, exc_info=True)
        self.appsource_thread_should_be_running = False
        self.emit('processing-error', str(error))

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'video-metadata':
            return self.video_metadata
        elif prop.name == 'frame-restorer-provider':
            return self.frame_restorer_provider
        else:
            return super().do_get_property(prop)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'video-metadata':
            self.appsource_thread_eof = False
            if self.video_metadata is None:
                self._set_video_metadata(value)
            else:
                with self.appsrc_lock:
                    should_start = self.appsource_thread is not None and not self.appsource_thread_stop_requested
                    self._stop_appsource_worker()
                    self.current_timestamp_ns = 0
                    self._set_video_metadata(value)
                    if should_start:
                        self._start_appsource_worker()
        elif prop.name == 'frame-restorer-provider':
            self.frame_restorer_provider = value
        else:
            super().do_set_property(prop, value)

    def do_state_changed(self, oldstate: Gst.State, newstate: Gst.State, pending: Gst.State) -> None:
        logger.debug(f"appsource state change: {oldstate.name} -> {newstate.name} (pending: {pending.name})")
        if oldstate == Gst.State.READY and newstate == Gst.State.NULL:
            self._stop_appsource_worker(shutdown=True)
        elif oldstate == Gst.State.NULL and newstate == Gst.State.READY:
            self.appsource_thread_shutdown_requested = False

    def _set_video_metadata(self, video_metadata: VideoMetadata):
        self.video_metadata = video_metadata
        options = self.frame_restorer_provider.options.vfx if self.frame_restorer_provider else None
        scale = options.scale if options and options.enabled else 1
        multiplier = options.multiplier if options and options.enabled else 1
        fps = self.video_metadata.video_fps_exact * multiplier
        self.frame_duration_ns = (1 / float(fps)) * Gst.SECOND
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={GstPaddingHelpers.get_padded_width(self.video_metadata.video_width * scale)},height={self.video_metadata.video_height * scale},framerate={fps.numerator}/{fps.denominator}")
        self.set_property('caps', caps)
        self.set_property('duration', round(self.video_metadata.duration * Gst.SECOND))
        logger.debug(f"appsource set video metadata: {video_metadata.video_file}")

    def _on_need_data(self, src, length):
        logger.debug("appsource need-data")
        try:
            with self.appsrc_lock:
                self._start_appsource_worker()
        except Exception as error:
            self._report_error(error)
        return True

    def _on_enough_data(self, src):
        logger.debug("appsource enough-data")
        # This callback can run synchronously inside push-buffer. Taking the
        # locks that stop()/join() hold would deadlock a seek or file change.
        self.appsource_thread_stop_requested = True
        self.appsource_thread_should_be_running = False
        return True

    def _on_seek_data(self, appsrc, offset_ns):
        logger.debug(f"appsource seek: offset (sec): {offset_ns / Gst.SECOND}, current position (sec): {self.current_timestamp_ns / Gst.SECOND}")
        with self.appsrc_lock:
            if offset_ns == self.current_timestamp_ns:
                # nothing to do, we're already at the desired position in the file or already received this seek request
                logger.debug("appsource seek: skipped seek as we're already at the seek position")
                return True
            if self.appsource_thread_shutdown_requested:
                logger.debug("appsource seek: skipped seek as shutdown was requested.")
                return True
            self.appsource_thread_eof = False
            self._stop_appsource_worker()
            self._start_appsource_worker(seek_position=offset_ns)
        return True

    def _start_appsource_worker(self, seek_position=None):
        with self.frame_restorer_lock:
            if self.appsource_thread_shutdown_requested:
                logger.debug(f"appsource worker: requested to start but shutdown was requested. Will not start")
                return
            if self.appsource_thread_eof:
                logger.debug(f"appsource worker: requested to start but EOF. Will not start")
                return
            self.appsource_thread_stop_requested = False
            self.appsource_thread_should_be_running = True

            if self.appsource_thread and self.appsource_thread.is_alive():
                logger.debug(f"appsource worker: requested to start but already started")
                return

            if seek_position:
                logger.debug(f"appsource worker: applying pending seek timestamp")
                assert self.appsource_thread is None, "starting appsource worker with pending timestamp but worker is still running -> you need to stop the worker before setting a pending timestamp"
                assert self.frame_restorer is None, "starting appsource worker with pending timestamp but frame restorer is still running -> you need to stop the frame restorer before setting a pending timestamp"

            if not self.frame_restorer:
                logger.debug(f"appsource worker: setting up frame restorer")
                self.frame_restorer = self.frame_restorer_provider.get()
                options = self.frame_restorer_provider.options.vfx
                if options and options.enabled:
                    from lada.utils.vfx_preview import VfxPreviewProcessor
                    self.vfx_processor = VfxPreviewProcessor(options, self.video_metadata.video_width,
                                                             self.video_metadata.video_height,
                                                             self.frame_restorer_provider.options.device)
                if seek_position is not None:
                    self.frame_restorer.start(start_ns=int(seek_position))
                    self.current_timestamp_ns = seek_position
                else:
                    self.frame_restorer.start(start_ns=int(self.current_timestamp_ns))

            self.appsource_thread = threading.Thread(target=self._appsource_worker)
            self.appsource_thread.start()

    def _request_stop_appsource_worker(self):
        with self.frame_restorer_lock:
            self.appsource_thread_stop_requested = True
            self.appsource_thread_should_be_running = False

    def _stop_appsource_worker(self, shutdown=False):
        with self.frame_restorer_lock:
            start = time.time()
            if shutdown:
                logger.debug(f"appsource worker: shutdown requested")
                self.appsource_thread_shutdown_requested =True
            self.appsource_thread_stop_requested = True
            self.appsource_thread_should_be_running = False

            frame_restorer_thread_queue = None
            if self.frame_restorer:
                logger.debug(f"appsource worker: stopping frame restorer")
                self.frame_restorer.stop()
                frame_restorer_thread_queue = self.frame_restorer.get_frame_restoration_queue()
                # unblock consumer
                threading_utils.put_queue_stop_marker(frame_restorer_thread_queue)

            if self.appsource_thread:
                self.appsource_thread.join()
                logger.debug(f"appsource worker: joined appsource_thread")
                self.appsource_thread = None

            if self.frame_restorer:
                # garbage collection
                threading_utils.empty_out_queue(frame_restorer_thread_queue)
                self.frame_restorer = None
            self.pending_display_buffer = None
            if self.vfx_processor is not None:
                self.vfx_processor.close()
                self.vfx_processor = None

            logger.debug(f"appsource worker: stopped, took {time.time() - start}")

    def _appsource_worker(self):
        logger.debug("appsource worker: started")
        queue_marker = None
        while self.appsource_thread_should_be_running:
            try:
                queue_marker = self._get_next_frame_and_push_buffer()
            except Exception as error:
                self._report_error(error)
                return
            if queue_marker is EOF_MARKER:
                self.appsource_thread_should_be_running = False
                self.appsource_thread_eof = True
                self._flush_display_buffer()
                self.emit("end-of-stream")
            elif queue_marker is STOP_MARKER:
                self.appsource_thread_should_be_running = False
                if not self.appsource_thread_stop_requested:
                    logger.warning("appsource worker: Invalid state. Received queue stop marker but not requested to shutdown")
        if queue_marker is EOF_MARKER:
            logger.debug("appsource worker: stopped itself, EOF")
        elif queue_marker is STOP_MARKER:
            logger.debug("appsource worker: stopped by request")
        else:
            pass

    def _get_next_frame_and_push_buffer(self) -> StopMarker | EofMarker | None:
        result = self.frame_restorer.get_frame_restoration_queue().get()
        if isinstance(result, ErrorMarker):
            raise result
        if self.appsource_thread_stop_requested or result is STOP_MARKER or result is EOF_MARKER:
            logger.debug("appsource worker: frame_restoration_queue consumer unblocked")
            queue_marker = result
            return queue_marker
        else:
            frame, frame_pts = result

        if self.vfx_processor is not None:
            for enhanced, timestamp in self.vfx_processor.frames(frame, frame_pts):
                self._push_frame(enhanced, timestamp)
        else:
            self._push_frame(frame, frame_pts)
        return None

    def _push_frame(self, frame, frame_pts):
        frame_timestamp_ns = int((frame_pts * self.video_metadata.time_base) * Gst.SECOND)
        frame = GstPaddingHelpers.pad_frame(frame)
        device_type = frame.device.type
        if device_type in ('cuda', 'xpu', 'mps'):
            use_async_copy = device_type in ('cuda', 'xpu')
            if self.cpu_frame is None or frame.shape != self.cpu_frame.shape:
                self.cpu_frame = torch.empty((frame.shape[0], frame.shape[1], frame.shape[2]), dtype=frame.dtype, device='cpu', pin_memory=use_async_copy)
            self.cpu_frame.copy_(frame, non_blocking=use_async_copy)
            if use_async_copy:
                if device_type == 'cuda':
                    # Wait only for this stream's display transfer, not every
                    # CUDA stream used by detection/restoration in other workers.
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(frame.device))
                    event.synchronize()
                else:
                    torch.xpu.synchronize()
            data = self.cpu_frame.numpy().tobytes()
        else:
            data = frame.numpy().tobytes()

        # bytes owns the data after the reusable pinned host buffer is released.
        buf = Gst.Buffer.new_wrapped(data)
        buf.duration = round(self.frame_duration_ns)
        buf.pts = frame_timestamp_ns
        buf.offset = video_utils.offset_ns_to_frame_num(frame_timestamp_ns, self.video_metadata.video_fps_exact)
        if self.pending_display_buffer is not None:
            previous = self.pending_display_buffer
            previous.duration = max(1, buf.pts - previous.pts)
            self._emit_display_buffer(previous)
        self.pending_display_buffer = buf
        self.current_timestamp_ns = frame_timestamp_ns

    def _flush_display_buffer(self):
        if self.pending_display_buffer is None:
            return
        buf, self.pending_display_buffer = self.pending_display_buffer, None
        start = (self.video_metadata.start_pts or 0) * self.video_metadata.time_base
        end_ns = round((float(start) + self.video_metadata.duration) * Gst.SECOND)
        buf.duration = max(1, end_ns - buf.pts)
        self._emit_display_buffer(buf)

    def _emit_display_buffer(self, buf):
        result = self.emit('push-buffer', buf)
        if result not in (Gst.FlowReturn.OK, Gst.FlowReturn.FLUSHING):
            raise RuntimeError(f'Preview sink rejected a frame: {result}')

        return None


class GstPaddingHelpers:
    # TODO: As we're using BGR format GStreamer expects to receive a 'buffer size = rstride (image) * height' where 'rstride = RU4 (width * 3)'
    # RU4 here means that it will round up to nearest number divisible by 4. (https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-video-raw.html)
    # Most common video widths like 1920, 1280, 640 are divisible by 4 so no problem we can allocate a buffer according to numpy shape H*W*C
    # But if we receive a video with a width which isn't divisible by 4 (I saw this on a file with dimensions 854 x 480) then the pipeline would break as H*W*C is less then expected buffer size given above calculation.
    # For now let's just add some zero padding. Maybe there are ways to explicitly set the stride size or tell GStreamer about our zero padding but couldn't find anything...

    @staticmethod
    def pad_frame(frame: torch.Tensor):
        width = frame.shape[1]
        # TODO: see reasoning for this zero padding in TODO where we specify appsrc Caps
        if width % 4 != 0:
            pad_w = (-width) % 4
            pad_tensor = torch.zeros((frame.shape[0], pad_w, frame.shape[2]), dtype=frame.dtype, device=frame.device)
            return torch.cat((frame, pad_tensor), dim=1)
        return frame

    @staticmethod
    def get_padded_width(width):
        return width + (-width) % 4

GObject.type_register(FrameRestorerAppSrc)
__gstelementfactory__ = (FrameRestorerAppSrc.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, FrameRestorerAppSrc)

"""Persistent, single-worker video processing preserving every frame's presentation time."""

from __future__ import annotations

import logging
import math
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import av
import imageio_ffmpeg
from PIL import Image
from pydantic import BaseModel

from backend.captures import Capture, CapturePolicy, CaptureStore
from backend.detector import MAX_PIXELS, MODEL_NAMES, Detector, WeaponModelId, draw_annotations

LOGGER = logging.getLogger(__name__)
MAX_VIDEO_BYTES = 100 * 1024 * 1024
DEFAULT_VIDEO_DIR = Path(__file__).resolve().parent.parent / "video_jobs"
TERMINAL = {"completed", "failed", "cancelled"}


class VideoJob(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    frames_processed: int = 0
    total_frames: int | None = None
    fps: float | None = None
    progress: float | None = None
    video_url: str | None = None
    error: str | None = None
    capture: Capture | None = None
    capture_count: int = 0
    capture_error: str | None = None
    false_detection_count: int = 0
    model_id: WeaponModelId | None = None
    model_name: str | None = None
    rifle_model: str | None = None


class VideoSettings(BaseModel):
    classes: list[str]
    object_conf: float
    person_conf: float
    model_id: WeaponModelId
    auto_capture: bool
    policy: CapturePolicy


class VideoBusy(Exception):
    pass


class VideoCancelled(Exception):
    pass


class VideoJobs:
    def __init__(
        self, detector: Detector, captures: CaptureStore, directory: Path = DEFAULT_VIDEO_DIR,
    ) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.detector = detector
        self.captures = captures
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video")
        self.jobs: dict[str, VideoJob] = {}
        self.events: dict[str, threading.Event] = {}
        self.processes: dict[str, list[subprocess.Popen]] = {}
        self.active: str | None = None
        self.closed = False
        for path in sorted(self.directory.glob("*.json"), key=lambda item: item.stat().st_mtime):
            try:
                job = VideoJob.model_validate_json(path.read_text())
                if UUID(job.id).hex != path.stem:
                    continue
                self.jobs[job.id] = job
                if job.status not in TERMINAL:
                    job.status = "failed"
                    job.error = "Video processing was interrupted by a backend restart."
                if job.status == "completed" and not self.output(job.id).is_file():
                    job.status = "failed"
                    job.error = "The completed video is missing."
                    job.video_url = None
                self._cleanup(job.id, keep_output=job.status == "completed")
                self._persist(job)
            except (ValueError, OSError):
                LOGGER.exception("Could not restore video metadata %s", path.name)

    def output(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.mp4"

    def input(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.input"

    def _persist(self, job: VideoJob) -> None:
        staged = self.directory / f"{job.id}.json.part"
        staged.write_text(job.model_dump_json())
        staged.replace(self.directory / f"{job.id}.json")

    def _cleanup(self, job_id: str, keep_output: bool = False) -> None:
        for suffix in ("input", "encoded.mp4", "mux.mp4", "decode.log", "encode.log", "mux.log"):
            (self.directory / f"{job_id}.{suffix}").unlink(missing_ok=True)
        if not keep_output:
            self.output(job_id).unlink(missing_ok=True)

    def reserve(self, filename: str, settings: VideoSettings | None = None) -> VideoJob:
        with self.lock:
            if self.closed or self.active is not None:
                raise VideoBusy
            job = VideoJob(id=uuid4().hex, filename=filename)
            if settings is not None:
                job.model_id = settings.model_id
                job.model_name = MODEL_NAMES[settings.model_id]
                job.rifle_model = self.detector.rifle_filename if "rifle" in settings.classes else None
            self._persist(job)
            self.jobs[job.id] = job
            self.events[job.id] = threading.Event()
            self.active = job.id
            return job.model_copy(deep=True)

    def get(self, job_id: str) -> VideoJob | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def list(self) -> list[VideoJob]:
        with self.lock:
            recent = list(reversed(self.jobs.values()))
            recent.sort(key=lambda job: job.id != self.active)
            return [job.model_copy(deep=True) for job in recent[:20]]

    def update(self, job_id: str, **values) -> None:
        with self.lock:
            job = self.jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)
            self._persist(job)

    def finish(self, job_id: str, status: str, error: str | None = None) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job.status in TERMINAL:
                return
            if self.events[job_id].is_set():
                status, error = "cancelled", None
            self._cleanup(job_id, keep_output=status == "completed")
            self.update(
                job_id, status=status, error=error,
                video_url=f"/api/videos/{job_id}/result" if status == "completed" else None,
                progress=100.0 if status == "completed" else job.progress,
            )
            self.events.pop(job_id, None)
            if self.active == job_id:
                self.active = None

    def cancel(self, job_id: str) -> VideoJob | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.status not in TERMINAL:
                self.events[job_id].set()
                for process in self.processes.get(job_id, []):
                    if process.poll() is None:
                        process.terminate()
            return job.model_copy(deep=True)

    def submit(self, job_id: str, settings: VideoSettings) -> None:
        with self.lock:
            if self.closed:
                self.events[job_id].set()
                self.finish(job_id, "cancelled")
                return
            self.executor.submit(self._run, job_id, settings)

    def shutdown(self) -> None:
        with self.lock:
            self.closed = True
            if self.active:
                self.cancel(self.active)
        self.executor.shutdown(wait=True)
        with self.lock:
            if self.active:
                self.finish(self.active, "cancelled")

    def _check(self, job_id: str) -> None:
        if self.events[job_id].is_set():
            raise VideoCancelled

    def _spawn(self, job_id: str, arguments: list[str], log, **kwargs) -> subprocess.Popen:
        with self.lock:
            self._check(job_id)
            process = subprocess.Popen(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", *arguments],
                stderr=log, **kwargs,
            )
            self.processes.setdefault(job_id, []).append(process)
            return process

    @staticmethod
    def _wait(process: subprocess.Popen, log_path: Path, operation: str) -> None:
        if process.wait() != 0:
            detail = log_path.read_text(errors="replace")[-2000:].strip()
            raise RuntimeError(f"{operation} failed: {detail or 'FFmpeg exited unsuccessfully.'}")

    def _encode(self, job_id: str, settings: VideoSettings, source: Path, encoded: Path) -> int:
        with av.open(str(source)) as decoder:
            if not decoder.streams.video:
                raise ValueError("Video contains no video stream.")
            video = decoder.streams.video[0]
            width, height = video.width, video.height
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                raise ValueError("Video cannot be decoded or exceeds the 20-megapixel frame limit.")
            rate = video.average_rate or video.guessed_rate
            if rate is None or not math.isfinite(float(rate)) or rate <= 0:
                raise ValueError("Video does not have a valid frame rate.")
            time_base = video.time_base
            if time_base is None or time_base <= 0:
                raise ValueError("Video has no usable timestamp time base.")
            total = video.frames or None
            self.update(job_id, fps=float(rate), total_frames=total, progress=0.0 if total else None)
            video.codec_context.options = {"err_detect": "explode"}
            processed = 0
            durations: dict[int, int] = {}
            with av.open(str(encoded), "w", format="mp4") as encoder:
                output = encoder.add_stream("libx264", rate=rate)
                output.width, output.height = width + width % 2, height + height % 2
                output.pix_fmt = "yuv420p"
                output.time_base = time_base
                output.codec_context.time_base = time_base
                output.options = {
                    "preset": "veryfast", "crf": "20", "bf": "0", "tune": "zerolatency",
                }

                def mux_packets(packets) -> None:
                    for packet in packets:
                        if packet.time_base != time_base or packet.pts not in durations:
                            raise ValueError("H.264 encoder could not preserve source frame timestamps.")
                        packet.duration = durations.pop(packet.pts)
                        encoder.mux(packet)

                frames = iter(decoder.decode(video))
                frame = next(frames, None)
                while frame is not None:
                    self._check(job_id)
                    following = next(frames, None)
                    if frame.pts is None or frame.time_base != time_base:
                        raise ValueError("Video frame timestamps are missing or unsupported.")
                    if following is not None:
                        if following.pts is None or following.pts <= frame.pts:
                            raise ValueError("Video frame timestamps must be strictly increasing.")
                        duration = following.pts - frame.pts
                    else:
                        duration = frame.duration
                        if not duration and video.duration is not None:
                            duration = (video.start_time or 0) + video.duration - frame.pts
                    if not duration or duration <= 0:
                        raise ValueError("Video frame duration is missing or unsupported.")
                    if frame.width != width or frame.height != height:
                        raise ValueError("Videos that change resolution mid-stream are unsupported.")
                    image = frame.to_image()
                    result = self.detector.detect(
                        image, settings.classes, settings.object_conf, settings.person_conf,
                        settings.model_id, encode_annotation=False,
                    )
                    self._check(job_id)
                    annotated = draw_annotations(image, result.detections)
                    padded = annotated
                    if annotated.size != (output.width, output.height):
                        padded = Image.new("RGB", (output.width, output.height))
                        padded.paste(annotated)
                    converted = av.VideoFrame.from_image(padded)
                    converted.pts, converted.time_base, converted.duration = frame.pts, time_base, duration
                    durations[frame.pts] = duration
                    mux_packets(output.encode(converted))
                    processed += 1
                    values = {
                        "frames_processed": processed,
                        "progress": min(99.0, processed / total * 100) if total else None,
                        "false_detection_count": self.get(job_id).false_detection_count + result.false_detection_count,
                    }
                    if settings.auto_capture:
                        try:
                            capture = self.captures.save_if_eligible(
                                result, "video", settings.policy, annotated_frame=annotated,
                            )
                            if capture:
                                values.update(
                                    capture=capture, capture_count=self.get(job_id).capture_count + 1,
                                )
                        except Exception:
                            LOGGER.exception("Video automatic capture failed")
                            values["capture_error"] = "Capture could not be saved. Check disk space and folder permissions."
                    self.update(job_id, **values)
                    frame = following
                mux_packets(output.encode(None))
                if durations:
                    raise ValueError("H.264 encoding did not output every source frame.")
            if not processed:
                raise ValueError("Video contains no decodable frames.")
            if total is not None and processed < total:
                raise ValueError(f"Video ended early: decoded {processed} of {total} frames.")
            return processed

    def _run(self, job_id: str, settings: VideoSettings) -> None:
        status, error = "failed", None
        try:
            self._check(job_id)
            self.update(job_id, status="running")
            source = self.input(job_id)
            encoded = self.directory / f"{job_id}.encoded.mp4"
            processed = self._encode(job_id, settings, source, encoded)
            self._check(job_id)
            muxed = self.directory / f"{job_id}.mux.mp4"
            mux_log = self.directory / f"{job_id}.mux.log"
            with mux_log.open("wb") as log:
                mux = self._spawn(job_id, [
                    "-xerror", "-copyts", "-i", str(encoded), "-i", str(source),
                    "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-avoid_negative_ts", "disabled",
                    "-movflags", "+faststart", str(muxed),
                ], log, stdout=subprocess.DEVNULL)
                self._wait(mux, mux_log, "Audio muxing")
            self._check(job_id)
            muxed.replace(self.output(job_id))
            self.update(job_id, total_frames=processed)
            status = "completed"
        except VideoCancelled:
            status = "cancelled"
        except Exception as exc:
            LOGGER.exception("Video job %s failed", job_id)
            error = str(exc) or "Video processing failed."
        finally:
            with self.lock:
                processes = self.processes.pop(job_id, [])
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream:
                        try:
                            stream.close()
                        except OSError:
                            pass
            self.finish(job_id, status, error)

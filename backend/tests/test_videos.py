import subprocess
import threading
import time
from uuid import uuid4

import imageio_ffmpeg
import av
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import videos
from backend.captures import CapturePolicy, CaptureStore
from backend.detector import Detector, OBJECT_CLASSES
from backend.main import create_app
from backend.tests.test_api import model
from backend.videos import VideoJobs, VideoSettings


@pytest.fixture
def detector():
    objects = model(dict(enumerate(OBJECT_CLASSES)), [[1, 2, 25, 30, 0.9, 0]])
    people = model({0: "person"}, [[0, 0, 40, 40, 0.92, 0]])
    rifle = model({0: "rifle"}, [[2, 3, 30, 40, 0.9, 0]])
    return Detector(objects, people, "cpu", objects, rifle)


@pytest.fixture
def app(detector, tmp_path):
    return create_app(detector, CaptureStore(tmp_path / "captures"), tmp_path / "videos")


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def ffmpeg(*arguments):
    return subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", *arguments],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def source_video(directory, *, audio=False, odd=False):
    path = directory / "source.mkv"
    size = "65x49" if odd else "66x50"
    arguments = ["-f", "lavfi", "-i", f"testsrc=size={size}:rate=6:duration=1"]
    if audio:
        arguments += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "pcm_s16le"]
    ffmpeg(*arguments, "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path))
    return path


def upload(client, path, **settings):
    return client.post(
        "/api/videos", files={"file": ("../original.mkv", path.read_bytes(), "video/x-matroska")},
        data=settings,
    )


def wait_for(client, job_id, statuses=("completed", "failed", "cancelled")):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(f"/api/videos/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    pytest.fail(f"Video job did not reach {statuses}: {job}")


def assert_clean(store, job_id, completed=False):
    assert not store.input(job_id).exists()
    assert store.output(job_id).exists() == completed
    assert not list(store.directory.glob(f"{job_id}.*.mp4"))
    assert not list(store.directory.glob(f"{job_id}.*.log"))


def test_video_preserves_annotated_frames_without_alert_banner(client, app, tmp_path, monkeypatch):
    monkeypatch.setattr(videos, "draw_annotations",
                        lambda image, detections: Image.new("RGB", image.size, "white"))
    response = upload(client, source_video(tmp_path), classes='["pistol"]', auto_capture="false")
    assert response.status_code == 202
    job = wait_for(client, response.json()["id"])
    assert job["status"] == "completed", job
    frames = imageio_ffmpeg.read_frames(str(app.state.video_jobs.output(job["id"])), pix_fmt="rgb24")
    next(frames)
    assert all(value > 230 for value in next(frames)[:3])


def test_video_rejects_oversized_weapons_before_annotation_and_capture(client, app, detector, tmp_path, monkeypatch):
    detector.people = model({0: "person"}, [[0, 0, 5, 5, 0.92, 0]])
    annotated_labels = []

    def record_annotations(image, detections):
        annotated_labels.append([item.label for item in detections])
        return Image.new("RGB", image.size, "white")

    monkeypatch.setattr(videos, "draw_annotations", record_annotations)
    response = upload(client, source_video(tmp_path), classes='["person","pistol","rifle"]',
                      auto_capture="true", capture_classes='["pistol","rifle"]')
    assert response.status_code == 202
    job = wait_for(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert job["false_detection_count"] == 2 * job["frames_processed"]
    assert job["capture_count"] == 0
    assert job["capture"] is None
    assert annotated_labels == [["person"]] * job["frames_processed"]
    assert app.state.video_jobs.get(job["id"]).false_detection_count == job["false_detection_count"]
    restored = VideoJobs(detector, CaptureStore(tmp_path / "captures"), app.state.video_jobs.directory)
    try:
        assert restored.get(job["id"]).false_detection_count == job["false_detection_count"]
    finally:
        restored.shutdown()


@pytest.mark.parametrize("audio,odd", [(True, False), (False, True)])
def test_full_frames_h264_audio_capture_and_range(client, app, detector, tmp_path, monkeypatch, audio, odd):
    monkeypatch.setattr("backend.detector.annotate", lambda *args: pytest.fail("Per-frame JPEG encoding"))
    response = upload(
        client, source_video(tmp_path, audio=audio, odd=odd), model_id="v1",
        classes='["person","pistol","rifle"]', object_conf="0.5", person_conf="0.6",
        auto_capture="true", capture_object_conf="0.85", capture_person_conf="0.9",
        capture_classes='["rifle"]', warning_classes='["rifle"]', capture_cooldown="60",
    )
    assert response.status_code == 202
    job = wait_for(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert job["frames_processed"] == job["total_frames"] == 6
    assert job["fps"] == 6
    assert job["progress"] == 100
    assert job["filename"] == "../original.mkv"
    assert job["model_id"] == "v1"
    assert job["model_name"] == "Weapon detection model v1"
    assert job["rifle_model"] == "rifle_detector.pt"
    assert job["error"] is None
    assert job["capture_count"] == 1
    assert job["capture_error"] is None
    assert job["capture"]["model_id"] == "v1"
    assert job["capture"]["source"] == "video"
    assert job["capture"]["is_warning"] is True
    assert client.get(job["capture"]["image_url"]).status_code == 200
    assert detector.objects.predict.call_count == 6
    assert detector.rifle.predict.call_count == 6
    assert detector.people.predict.call_count == 6
    assert detector.objects.predict.call_args.kwargs["conf"] == 0.5
    assert detector.people.predict.call_args.kwargs["conf"] == 0.6
    output = app.state.video_jobs.output(job["id"])
    frames = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    metadata = next(frames)
    assert metadata["codec"] == "h264"
    assert metadata["size"] == (66, 50)
    assert len(list(frames)) == 6
    if audio:
        decoded_audio = ffmpeg("-i", str(output), "-map", "0:a:0", "-f", "s16le", "pipe:1").stdout
        assert len(decoded_audio) >= 44100 * 2
        assert any(decoded_audio)
        info = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(output)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stderr
        assert b"Audio: aac" in info
    result = client.get(job["video_url"])
    assert result.status_code == 200
    assert result.headers["content-type"] == "video/mp4"
    assert result.headers["content-disposition"].startswith("inline")
    ranged = client.get(job["video_url"], headers={"Range": "bytes=0-31"})
    assert ranged.status_code == 206
    assert ranged.content == result.content[:32]
    assert client.get(job["video_url"] + "?download=true").headers["content-disposition"].startswith("attachment")
    assert client.post(f"/api/videos/{job['id']}/cancel").json()["status"] == "completed"
    assert client.get("/api/videos").json()[0]["id"] == job["id"]
    assert_clean(app.state.video_jobs, job["id"], completed=True)


def block_detector(detector):
    entered, release = threading.Event(), threading.Event()
    result = detector.objects.predict.return_value
    calls = 0

    def predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(10)
        return result

    detector.objects.predict.side_effect = predict
    return entered, release


def test_progress_busy_cancellation_and_cleanup(client, app, detector, tmp_path):
    entered, release = block_detector(detector)
    try:
        path = source_video(tmp_path)
        mp4 = tmp_path / "progress.mp4"
        ffmpeg("-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4))
        path = mp4
        job = upload(client, path).json()
        assert entered.wait(10)
        running = client.get(f"/api/videos/{job['id']}").json()
        assert running["status"] == "running"
        assert running["frames_processed"] == 1
        assert 0 < running["progress"] < 100
        assert client.get("/api/videos").json()[0]["id"] == job["id"]
        assert upload(client, path).status_code == 409
        assert client.get(f"/api/videos/{job['id']}/result").status_code == 409
        client.post(f"/api/videos/{job['id']}/cancel")
    finally:
        release.set()
    cancelled = wait_for(client, job["id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["video_url"] is None
    assert cancelled["error"] is None
    assert_clean(app.state.video_jobs, job["id"])
    assert upload(client, path).status_code == 202


@pytest.mark.parametrize("content,status", [(b"", 400), (b"x" * 21, 413)])
def test_upload_limit_and_empty(client, app, monkeypatch, content, status):
    monkeypatch.setattr(videos, "MAX_VIDEO_BYTES", 20)
    response = client.post("/api/videos", files={"file": ("bad.mp4", content)})
    assert response.status_code == status
    job = client.get("/api/videos").json()[0]
    assert job["status"] == "failed"
    assert job["error"]
    assert_clean(app.state.video_jobs, job["id"])
    assert app.state.video_jobs.active is None


@pytest.mark.parametrize("settings", [
    {"classes": "[]"}, {"classes": '["invalid"]'}, {"object_conf": "nan"},
    {"person_conf": "0"}, {"model_id": ""}, {"model_id": "v2"},
    {"auto_capture": "true", "capture_classes": "[]"}, {"warning_classes": '["person"]'},
])
def test_settings_validation(client, settings):
    assert client.post("/api/videos", files={"file": ("video.mp4", b"x")}, data=settings).status_code == 422
    assert client.get("/api/videos").json() == []


@pytest.mark.parametrize("failure", ["invalid", "inference", "mux", "capture"])
def test_explicit_errors_and_capture_error(client, app, detector, tmp_path, monkeypatch, failure):
    path = source_video(tmp_path)
    if failure == "invalid":
        path.write_bytes(b"not a video")
    elif failure == "inference":
        detector.objects.predict.side_effect = RuntimeError("Inference exploded")
    elif failure == "mux":
        original_wait = VideoJobs._wait

        def fail_mux(process, log, operation):
            original_wait(process, log, operation)
            if operation == "Audio muxing":
                raise RuntimeError("Mux exploded")

        monkeypatch.setattr(VideoJobs, "_wait", staticmethod(fail_mux))
    elif failure == "capture":
        monkeypatch.setattr(app.state.capture_store, "save_if_eligible", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    job = upload(client, path, auto_capture="true").json()
    job = wait_for(client, job["id"])
    if failure == "capture":
        assert job["status"] == "completed"
        assert job["capture_error"]
        assert job["capture_count"] == 0
    else:
        assert job["status"] == "failed", job
        assert job["error"]
        assert job["video_url"] is None
    assert_clean(app.state.video_jobs, job["id"], completed=failure == "capture")


def test_restart_recovers_completed_and_interrupts_queued(detector, tmp_path):
    captures = CaptureStore(tmp_path / "captures")
    directory = tmp_path / "videos"
    store = VideoJobs(detector, captures, directory)
    interrupted = store.reserve("interrupted.mp4")
    store.input(interrupted.id).write_bytes(b"incomplete")
    (directory / f"{interrupted.id}.encoded.mp4").write_bytes(b"partial")
    store.executor.shutdown()
    restored = VideoJobs(detector, captures, directory)
    try:
        assert restored.get(interrupted.id).status == "failed"
        assert "restart" in restored.get(interrupted.id).error
        assert_clean(restored, interrupted.id)
        completed = restored.reserve("finished.mp4", VideoSettings(
            classes=["pistol", "rifle"], model_id="v1", object_conf=0.25, person_conf=0.25,
            auto_capture=False, policy=CapturePolicy(),
        ))
        restored.output(completed.id).write_bytes(b"finished")
        restored.finish(completed.id, "completed")
    finally:
        restored.shutdown()
    again = VideoJobs(detector, captures, directory)
    try:
        assert again.get(completed.id).status == "completed"
        assert again.get(completed.id).model_id == "v1"
        assert again.get(completed.id).model_name == "Weapon detection model v1"
        assert again.get(completed.id).rifle_model == "rifle_detector.pt"
        assert again.get(interrupted.id).model_id is None
        assert again.output(completed.id).read_bytes() == b"finished"
        assert again.cancel(completed.id).status == "completed"
    finally:
        again.shutdown()


def test_shutdown_cancels_active_work(app, detector, tmp_path):
    entered, release = block_detector(detector)
    with TestClient(app) as client:
        job = upload(client, source_video(tmp_path)).json()
        assert entered.wait(10)
        timer = threading.Timer(0.1, release.set)
        timer.start()
    timer.join()
    assert app.state.video_jobs.get(job["id"]).status == "cancelled"
    assert app.state.video_jobs.active is None
    assert_clean(app.state.video_jobs, job["id"])


def test_routes_uuid_and_unknown(client):
    assert client.get("/api/videos").status_code == 200
    assert client.get("/api/videos/123").status_code == 422
    for suffix in ("", "/result"):
        assert client.get(f"/api/videos/{uuid4()}{suffix}").status_code == 404
    assert client.post(f"/api/videos/{uuid4()}/cancel").status_code == 404


def media_timing(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        timestamps = [float(frame.pts * frame.time_base) for frame in frames]
        # With B-frames, decoder packet durations need not match presentation intervals.
        video = [
            (timestamp, timestamps[index + 1] - timestamp if index + 1 < len(frames)
             else float(frames[index].duration * frames[index].time_base))
            for index, timestamp in enumerate(timestamps)
        ]
        video_duration = video[-1][0] + video[-1][1] - video[0][0]
    with av.open(str(path)) as container:
        audio = container.streams.audio[0]
        samples = list(container.decode(audio))
        audio_start = float(samples[0].pts * samples[0].time_base)
        audio_end = float(samples[-1].pts * samples[-1].time_base) + samples[-1].samples / samples[-1].sample_rate
    return video, video_duration, audio_start, audio_end


@pytest.mark.parametrize("variable", [True, False])
def test_timestamp_and_audio_fidelity(client, tmp_path, app, variable):
    source = tmp_path / ("vfr.mov" if variable else "cfr.mp4")
    arguments = [
        "-f", "lavfi", "-i", "testsrc=size=66x50:rate=25:duration=1.24",
        "-itsoffset", "0.16", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.08",
    ]
    if variable:
        arguments += ["-vf", "select=eq(n\\,0)+eq(n\\,2)+eq(n\\,7)+eq(n\\,8)+eq(n\\,17)+eq(n\\,30)"]
    ffmpeg(
        *arguments, "-fps_mode", "vfr", "-c:v", "libx264", "-bf", "2",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    )
    original = media_timing(source)
    job = wait_for(client, upload(client, source).json()["id"])
    assert job["status"] == "completed", job
    assert job["model_id"] == "v0"
    assert job["model_name"] == "Weapon detection model v0"
    assert job["rifle_model"] is None
    converted = media_timing(app.state.video_jobs.output(job["id"]))
    expected_count = 6 if variable else 31
    assert len(original[0]) == len(converted[0]) == job["frames_processed"] == expected_count
    assert [item[0] for item in converted[0]] == pytest.approx([item[0] for item in original[0]], abs=0.001)
    assert [item[1] for item in converted[0]] == pytest.approx([item[1] for item in original[0]], abs=0.001)
    assert converted[1] == pytest.approx(original[1], abs=0.001)
    assert converted[2] == pytest.approx(original[2], abs=0.03)
    assert converted[3] == pytest.approx(original[3], abs=0.03)
    assert converted[3] - converted[1] == pytest.approx(original[3] - original[1], abs=0.03)

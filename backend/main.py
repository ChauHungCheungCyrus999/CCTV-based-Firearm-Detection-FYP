import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.detector import ALL_OBJECT_CLASSES, CLASSES, COLORS, WEAPON_CLASSES, DetectionResult, Detector, InvalidImage, ModelInfo, WeaponModelId, decode_image
from backend.captures import Capture, CapturePage, CapturePolicy, CaptureSource, CaptureStore
from backend import videos
from backend.videos import DEFAULT_VIDEO_DIR, VideoBusy, VideoJob, VideoJobs, VideoSettings

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class ClassInfo(BaseModel):
    id: str
    label: str
    color: str
    category: str


class DetectionResponse(DetectionResult):
    capture: Capture | None = None
    capture_error: str | None = None


def parse_classes(value: str, allowed: tuple[str, ...], allow_empty: bool = False) -> list[str]:
    try:
        selected = json.loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(422, "Classes must be a JSON array of class names.") from exc
    if (not isinstance(selected, list) or (not selected and not allow_empty)
            or len(selected) > len(allowed)
            or any(not isinstance(name, str) or name not in allowed for name in selected)):
        raise HTTPException(422, f"Choose valid classes from: {', '.join(allowed)}.")
    return list(dict.fromkeys(selected))


def create_app(
    detector: Detector | None = None, capture_store: CaptureStore | None = None,
    video_directory: Path = DEFAULT_VIDEO_DIR,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = detector if detector is not None else await run_in_threadpool(Detector.load)
        app.state.capture_store = capture_store if capture_store is not None else await run_in_threadpool(CaptureStore)
        app.state.video_jobs = await run_in_threadpool(
            VideoJobs, app.state.detector, app.state.capture_store, video_directory,
        )
        try:
            yield
        finally:
            await run_in_threadpool(app.state.video_jobs.shutdown)

    application = FastAPI(title="Sentinel Detection Dashboard API", version="1.0.0", lifespan=lifespan)
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    application.add_middleware(
        CORSMiddleware, allow_origins=[origin.strip() for origin in origins.split(",")],
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"],
    )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.error("Request failed: %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(status_code=500, content={
            "detail": "Detection failed on the server. Check the backend logs and try again."
        })

    @application.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "object_model": "best.pt", "person_model": "yolo11s.pt",
                "rifle_model": request.app.state.detector.rifle_filename if request.app.state.detector.rifle is not None else "not loaded",
                "device": request.app.state.detector.device}

    @application.get("/api/config", response_model=list[ClassInfo])
    def config() -> list[ClassInfo]:
        return [ClassInfo(
            id=name, label=name.capitalize(), color=COLORS[name],
            category="weapon" if name in WEAPON_CLASSES else "person" if name == "person" else "object",
        ) for name in CLASSES]

    @application.get("/api/models", response_model=list[ModelInfo])
    def models(request: Request) -> list[ModelInfo]:
        return request.app.state.detector.available_models()

    @application.get("/api/captures", response_model=CapturePage)
    def captures(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=100)] = 12,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> CapturePage:
        return request.app.state.capture_store.list(limit, offset)

    @application.get("/api/captures/{capture_id}/image")
    def capture_image(request: Request, capture_id: UUID, download: bool = False) -> FileResponse:
        store = request.app.state.capture_store
        capture = store.get(capture_id.hex)
        if capture is None:
            raise HTTPException(404, "Capture not found.")
        path = store.directory / capture.filename
        if not path.is_file():
            raise HTTPException(404, "The saved capture image is missing from the captures folder.")
        return FileResponse(
            path, media_type="image/jpeg", filename=capture.filename,
            content_disposition_type="attachment" if download else "inline",
        )

    @application.delete("/api/captures/{capture_id}")
    def delete_capture(request: Request, capture_id: UUID) -> dict[str, str]:
        try:
            deleted = request.app.state.capture_store.delete(capture_id.hex)
        except (OSError, sqlite3.Error) as exc:
            LOGGER.exception("Could not delete capture %s", capture_id.hex)
            raise HTTPException(500, "Could not delete this capture. Check folder permissions and backend logs, then retry.") from exc
        if not deleted:
            raise HTTPException(404, "Capture not found.")
        return {"deleted_id": capture_id.hex}

    @application.post("/api/detect", response_model=DetectionResponse)
    async def detect(
        request: Request,
        file: Annotated[UploadFile, File()],
        classes: Annotated[str, Form()] = '["person","pistol","knife"]',
        object_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.3,
        person_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.55,
        auto_capture: Annotated[bool, Form()] = False,
        source: Annotated[CaptureSource, Form()] = "image",
        capture_object_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.8,
        capture_person_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.8,
        capture_cooldown: Annotated[int, Form(ge=1, le=60)] = 10,
        capture_classes: Annotated[str, Form()] = json.dumps(list(ALL_OBJECT_CLASSES)),
        warning_classes: Annotated[str, Form()] = '["pistol","knife","rifle"]',
        model_id: Annotated[WeaponModelId, Form()] = "v0",
    ) -> DetectionResponse:
        # FastAPI normalizes empty optional form fields to their defaults.
        if (await request.form()).get("model_id") == "":
            raise HTTPException(422, "Choose a weapon detection model: v0 or v1.")
        if model_id not in request.app.state.detector.object_models:
            raise HTTPException(422, f"Weapon detection model {model_id} is not loaded. Choose an available model.")
        selected = parse_classes(classes, CLASSES)
        if "rifle" in selected and request.app.state.detector.rifle is None:
            raise HTTPException(422, "Rifle detection model is not loaded. Check RIFLE_MODEL_PATH and restart the backend.")
        policy = CapturePolicy(
            object_conf=capture_object_conf, person_conf=capture_person_conf,
            cooldown_seconds=capture_cooldown,
            classes=parse_classes(capture_classes, ALL_OBJECT_CLASSES, allow_empty=not auto_capture),
            warning_classes=parse_classes(warning_classes, ALL_OBJECT_CLASSES, allow_empty=True),
        )
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not content:
            raise HTTPException(400, "The uploaded image is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 10 MB upload limit.")
        try:
            image = await run_in_threadpool(decode_image, content)
        except InvalidImage as exc:
            raise HTTPException(415, str(exc)) from exc
        result = await run_in_threadpool(
            request.app.state.detector.detect, image, selected, object_conf, person_conf, model_id
        )
        response = DetectionResponse(**result.model_dump())
        if auto_capture:
            try:
                response.capture = await run_in_threadpool(
                    request.app.state.capture_store.save_if_eligible, result, source, policy
                )
            except (OSError, sqlite3.Error):
                LOGGER.exception("Automatic capture could not be saved")
                response.capture_error = "Detection succeeded, but the capture could not be saved. Check folder permissions, disk space, and backend logs."
        return response

    @application.get("/api/videos", response_model=list[VideoJob])
    def video_list(request: Request) -> list[VideoJob]:
        return request.app.state.video_jobs.list()

    @application.post("/api/videos", response_model=VideoJob, status_code=202)
    async def upload_video(
        request: Request,
        file: Annotated[UploadFile, File()],
        classes: Annotated[str, Form()] = '["person","pistol","knife"]',
        object_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.25,
        person_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.25,
        auto_capture: Annotated[bool, Form()] = False,
        capture_object_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.8,
        capture_person_conf: Annotated[float, Form(ge=0.01, le=1, allow_inf_nan=False)] = 0.8,
        capture_cooldown: Annotated[int, Form(ge=1, le=60)] = 10,
        capture_classes: Annotated[str, Form()] = json.dumps(list(ALL_OBJECT_CLASSES)),
        warning_classes: Annotated[str, Form()] = '["pistol","knife","rifle"]',
        model_id: Annotated[WeaponModelId, Form()] = "v0",
    ) -> VideoJob:
        if (await request.form()).get("model_id") == "":
            raise HTTPException(422, "Choose a weapon detection model: v0 or v1.")
        loaded = request.app.state.detector
        if model_id not in loaded.object_models:
            raise HTTPException(422, f"Weapon detection model {model_id} is not loaded. Choose an available model.")
        selected = parse_classes(classes, CLASSES)
        if "rifle" in selected and loaded.rifle is None:
            raise HTTPException(422, "Rifle detection model is not loaded. Check RIFLE_MODEL_PATH and restart the backend.")
        settings = VideoSettings(
            classes=selected, object_conf=object_conf, person_conf=person_conf,
            model_id=model_id, auto_capture=auto_capture,
            policy=CapturePolicy(
                object_conf=capture_object_conf, person_conf=capture_person_conf,
                cooldown_seconds=capture_cooldown,
                classes=parse_classes(capture_classes, ALL_OBJECT_CLASSES, allow_empty=not auto_capture),
                warning_classes=parse_classes(warning_classes, ALL_OBJECT_CLASSES, allow_empty=True),
            ),
        )
        store = request.app.state.video_jobs
        try:
            job = store.reserve(file.filename or "uploaded-video", settings)
        except VideoBusy as exc:
            await file.close()
            raise HTTPException(409, "Another video is uploading or processing. Wait or cancel it first.") from exc
        try:
            size = 0
            with store.input(job.id).open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    if store.events[job.id].is_set():
                        raise HTTPException(409, "Video upload was cancelled.")
                    size += len(chunk)
                    if size > videos.MAX_VIDEO_BYTES:
                        raise HTTPException(413, "Video exceeds the 100 MiB upload limit.")
                    await run_in_threadpool(output.write, chunk)
            if not size:
                raise HTTPException(400, "The uploaded video is empty.")
            store.submit(job.id, settings)
        except BaseException as exc:
            store.finish(
                job.id, "failed", str(exc.detail) if isinstance(exc, HTTPException) else "Video upload failed.",
            )
            raise
        finally:
            await file.close()
        return store.get(job.id)

    @application.get("/api/videos/{job_id}", response_model=VideoJob)
    def video_job(request: Request, job_id: UUID) -> VideoJob:
        job = request.app.state.video_jobs.get(job_id.hex)
        if job is None:
            raise HTTPException(404, "Video job not found.")
        return job

    @application.post("/api/videos/{job_id}/cancel", response_model=VideoJob)
    def cancel_video(request: Request, job_id: UUID) -> VideoJob:
        job = request.app.state.video_jobs.cancel(job_id.hex)
        if job is None:
            raise HTTPException(404, "Video job not found.")
        return job

    @application.get("/api/videos/{job_id}/result")
    def video_result(request: Request, job_id: UUID, download: bool = False) -> FileResponse:
        store = request.app.state.video_jobs
        job = store.get(job_id.hex)
        if job is None:
            raise HTTPException(404, "Video job not found.")
        if job.status != "completed":
            raise HTTPException(409, "Video is not completed.")
        path = store.output(job.id)
        if not path.is_file():
            raise HTTPException(404, "The annotated video is missing.")
        return FileResponse(
            path, media_type="video/mp4", filename=f"annotated_{job.id}.mp4",
            content_disposition_type="attachment" if download else "inline",
        )

    return application


app = create_app()

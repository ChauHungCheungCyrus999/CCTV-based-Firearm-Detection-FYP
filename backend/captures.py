import base64
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field
from PIL import Image

from backend.detector import ALL_OBJECT_CLASSES, MODEL_NAMES, OBJECT_CLASSES, Detection, DetectionResult, WeaponModelId

CAPTURE_THRESHOLD = 0.8
CAPTURE_COOLDOWN_SECONDS = 10
DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parent.parent / "captures"
CaptureSource = Literal["image", "camera", "video"]


class CapturePolicy(BaseModel):
    object_conf: float = Field(default=CAPTURE_THRESHOLD, ge=0.01, le=1, allow_inf_nan=False)
    person_conf: float = Field(default=CAPTURE_THRESHOLD, ge=0.01, le=1, allow_inf_nan=False)
    cooldown_seconds: int = Field(default=CAPTURE_COOLDOWN_SECONDS, ge=1, le=60)
    classes: list[str] = Field(default_factory=lambda: list(ALL_OBJECT_CLASSES))
    warning_classes: list[str] = Field(default_factory=lambda: ["pistol", "knife", "rifle"])


class Capture(BaseModel):
    id: str
    filename: str
    created_at: datetime
    image_url: str
    source: CaptureSource
    width: int
    height: int
    object_confidence: float
    person_confidence: float
    detections: list[Detection]
    # Captures without persisted policies predate the rifle detector.
    policy: CapturePolicy = Field(default_factory=lambda: CapturePolicy(
        classes=list(OBJECT_CLASSES), warning_classes=["pistol", "knife"],
    ))
    model_id: WeaponModelId = "v0"
    model_name: str = MODEL_NAMES["v0"]
    rifle_model: str | None = None

    @computed_field
    @property
    def is_warning(self) -> bool:
        return any(
            item.source == "object" and item.label in self.policy.warning_classes
            and item.confidence >= self.policy.object_conf
            for item in self.detections
        )


class CapturePage(BaseModel):
    items: list[Capture]
    total: int


class CaptureStore:
    def __init__(self, directory: Path = DEFAULT_CAPTURE_DIR) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "captures.sqlite3"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS captures_created_at ON captures(created_at DESC)")
            connection.execute("CREATE TABLE IF NOT EXISTS capture_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("""
                INSERT OR IGNORE INTO capture_state (key, value)
                SELECT 'last_capture_at', created_at FROM captures ORDER BY created_at DESC LIMIT 1
            """)

    def save_if_eligible(
        self, result: DetectionResult, source: CaptureSource, policy: CapturePolicy | None = None,
        *, annotated_frame: Image.Image | None = None,
    ) -> Capture | None:
        policy = policy if policy is not None else CapturePolicy()
        object_confidence = max(
            (item.confidence for item in result.detections
             if item.source == "object" and item.label in policy.classes), default=0,
        )
        person_confidence = max(
            (item.confidence for item in result.detections if item.source == "person"), default=0,
        )
        if object_confidence < policy.object_conf or person_confidence < policy.person_conf:
            return None

        with closing(sqlite3.connect(self.database, timeout=30)) as connection, connection:
            # Serialize the cooldown check and save, including across backend workers.
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            latest = connection.execute(
                "SELECT value FROM capture_state WHERE key = 'last_capture_at'"
            ).fetchone()
            if latest and (now - datetime.fromisoformat(latest[0])).total_seconds() < policy.cooldown_seconds:
                return None
            capture_id = uuid4().hex
            filename = f"capture_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{capture_id}.jpg"
            capture = Capture(
                id=capture_id, filename=filename, created_at=now,
                image_url=f"/api/captures/{capture_id}/image", source=source,
                width=result.width, height=result.height, object_confidence=object_confidence,
                person_confidence=person_confidence, detections=result.detections,
                policy=policy,
                model_id=result.model_id, model_name=result.model_name,
                rifle_model=result.rifle_model,
            )
            image_path = self.directory / filename
            try:
                with image_path.open("xb") as file:
                    if annotated_frame is not None:
                        annotated_frame.save(file, format="JPEG", quality=90)
                    else:
                        file.write(base64.b64decode(result.annotated_image.split(",", 1)[1], validate=True))
                connection.execute(
                    "INSERT INTO captures (id, created_at, metadata) VALUES (?, ?, ?)",
                    (capture.id, now.isoformat(), capture.model_dump_json()),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO capture_state (key, value) VALUES ('last_capture_at', ?)",
                    (now.isoformat(),),
                )
                connection.commit()
            except (OSError, sqlite3.Error):
                image_path.unlink(missing_ok=True)
                raise
        return capture

    def list(self, limit: int, offset: int) -> CapturePage:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("BEGIN")
            total = connection.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
            rows = connection.execute(
                "SELECT metadata FROM captures ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return CapturePage(items=[Capture.model_validate_json(row[0]) for row in rows], total=total)

    def get(self, capture_id: str) -> Capture | None:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                "SELECT metadata FROM captures WHERE id = ?", (capture_id,)
            ).fetchone()
        return Capture.model_validate_json(row[0]) if row else None

    def delete(self, capture_id: str) -> bool:
        staged = self.directory / f".deleting-{capture_id}.jpg"
        with closing(sqlite3.connect(self.database, timeout=30)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT metadata FROM captures WHERE id = ?", (capture_id,),
            ).fetchone()
            if row is None:
                if staged.is_file():
                    staged.unlink(missing_ok=True)
                    return True
                return False
            capture = Capture.model_validate_json(row[0])
            path = self.directory / capture.filename
            try:
                if path.is_file():
                    path.replace(staged)
                connection.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
                connection.commit()
            except (OSError, sqlite3.Error):
                if staged.is_file():
                    staged.replace(path)
                raise
        # If removing the staged file fails, a retry can finish deletion by this ID.
        staged.unlink(missing_ok=True)
        return True

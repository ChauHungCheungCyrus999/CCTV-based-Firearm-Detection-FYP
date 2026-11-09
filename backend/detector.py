import base64
import io
import logging
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / ".ultralytics"))
from ultralytics import YOLO, settings  # noqa: E402

settings.update({
    "sync": False, "wandb": False, "mlflow": False, "clearml": False,
    "comet": False, "neptune": False, "tensorboard": False,
})

LOGGER = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent / "models"
OBJECT_CLASSES = ("pistol", "smartphone", "knife", "purse", "banknote", "card")
ALL_OBJECT_CLASSES = (*OBJECT_CLASSES, "rifle")
WEAPON_CLASSES = {"pistol", "knife", "rifle"}
CLASSES = ("person", *ALL_OBJECT_CLASSES)
RIFLE_MODEL_FILENAME = "rifle_detector.pt"
MAX_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
COLORS = {
    "person": "#61a8ff",
    "pistol": "#ff6577",
    "knife": "#ffac5c",
    "rifle": "#ff8a80",
    "smartphone": "#bc9bff",
    "purse": "#e9c66c",
    "banknote": "#4cdbad",
    "card": "#5dd6ed",
}
WeaponModelId = Literal["v0", "v1"]
MODEL_NAMES: dict[WeaponModelId, str] = {
    "v0": "Weapon detection model v0",
    "v1": "Weapon detection model v1",
}
MODEL_FILENAMES: dict[WeaponModelId, str] = {"v0": "best.pt", "v1": "last.pt"}


class ModelInfo(BaseModel):
    id: WeaponModelId
    name: str
    filename: str


class InvalidImage(ValueError):
    pass


class Detection(BaseModel):
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    source: Literal["object", "person"]


class DetectionResult(BaseModel):
    width: int
    height: int
    inference_ms: float
    detections: list[Detection]
    counts: dict[str, int]
    weapon_count: int
    false_detection_count: int = 0
    annotated_image: str
    model_id: WeaponModelId = "v0"
    model_name: str = MODEL_NAMES["v0"]
    rifle_model: str | None = None


def decode_image(content: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("Unsupported image. Use a JPEG, PNG, or WebP image.")
                if image.width * image.height > MAX_PIXELS:
                    raise InvalidImage("Image is too large. Maximum resolution is 20 megapixels.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Image is corrupt, unsupported, or exceeds the 20-megapixel limit.") from exc


def draw_annotations(image: Image.Image, detections: list[Detection]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    scale = max(1, round(max(image.size) / 650))
    font = ImageFont.load_default(size=13 * scale)
    for detection in detections:
        color = COLORS[detection.label]
        x1, y1, x2, y2 = detection.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2 * scale)
        text = f"{detection.label} {detection.confidence:.0%}"
        bounds = draw.textbbox((0, 0), text, font=font)
        text_width = bounds[2] - bounds[0] + 8 * scale
        text_height = bounds[3] - bounds[1] + 8 * scale
        left = max(0, min(x1, image.width - text_width))
        top = max(0, y1 - text_height)
        draw.rectangle((left, top, left + text_width, top + text_height), fill=color)
        draw.text((left + 4 * scale, top + 2 * scale - bounds[1]),
                  text, fill="#071017", font=font)
    return output


def annotate(image: Image.Image, detections: list[Detection]) -> str:
    buffer = io.BytesIO()
    draw_annotations(image, detections).save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def person_match_rank(weapon: Detection, person: Detection) -> tuple[bool, float, float]:
    w, p = weapon.box, person.box
    overlap = max(0, min(w[2], p[2]) - max(w[0], p[0])) * max(0, min(w[3], p[3]) - max(w[1], p[1]))
    union = box_area(w) + box_area(p) - overlap
    iou = overlap / union if union else 0.0
    distance = ((w[0] + w[2] - p[0] - p[2]) ** 2 + (w[1] + w[3] - p[1] - p[3]) ** 2)
    return overlap > 0, iou, -distance


def filter_oversized_weapons(detections: list[Detection], people: list[Detection]) -> tuple[list[Detection], int]:
    kept = []
    rejected = 0
    for detection in detections:
        if detection.label in WEAPON_CLASSES and (
            not people or box_area(detection.box) > box_area(
                max(people, key=lambda person: person_match_rank(detection, person)).box
            )
        ):
            rejected += 1
        else:
            kept.append(detection)
    return kept, rejected


class Detector:
    def __init__(
        self, objects: YOLO, people: YOLO, device: str,
        object_v1: YOLO | None = None, rifle: YOLO | None = None,
    ) -> None:
        self.objects = objects
        self.object_models: dict[WeaponModelId, YOLO] = {"v0": objects}
        self.model_filenames = dict(MODEL_FILENAMES)
        if object_v1 is not None:
            self.object_models["v1"] = object_v1
        self.people = people
        self.rifle = rifle
        self.rifle_filename = RIFLE_MODEL_FILENAME
        self.device = device
        self.lock = threading.Lock()

    @classmethod
    def load(cls) -> "Detector":
        object_path = Path(os.getenv("OBJECT_MODEL_PATH", str(MODEL_DIR / "best.pt")))
        v1_path = Path(os.getenv("OBJECT_MODEL_V1_PATH", str(MODEL_DIR / "last.pt")))
        person_path = Path(os.getenv("PERSON_MODEL_PATH", str(MODEL_DIR / "yolo11s.pt")))
        rifle_path = Path(os.getenv("RIFLE_MODEL_PATH", str(MODEL_DIR / RIFLE_MODEL_FILENAME)))
        for path in (object_path, v1_path, person_path, rifle_path):
            if not path.is_file():
                raise RuntimeError(f"Model missing: {path}. Follow the model setup in README.md.")
        def load_objects(path: Path) -> YOLO:
            model = YOLO(str(path))
            names = [model.names[index] for index in range(len(model.names))]
            if model.task != "detect" or names != list(OBJECT_CLASSES):
                raise RuntimeError(f"Expected SoHAS classes {OBJECT_CLASSES} in {path.name}, received {names}.")
            return model

        objects = load_objects(object_path)
        object_v1 = load_objects(v1_path)
        rifle = YOLO(str(rifle_path))
        if rifle.task != "detect" or rifle.names != {0: "rifle"}:
            raise RuntimeError(f"Expected the single rifle class in {rifle_path.name}, received {rifle.names}.")
        people = YOLO(str(person_path))
        if people.task != "detect" or people.names.get(0) != "person" or len(people.names) != 80:
            raise RuntimeError("The person model must be the original 80-class COCO YOLO11s model.")
        # CPU is portable; opt into mps/cuda explicitly after testing your hardware.
        device = os.getenv("DETECTION_DEVICE", "cpu")
        LOGGER.info("Loaded v0/v1 object models, rifle model, and shared person detector on %s", device)
        detector = cls(objects, people, device, object_v1, rifle)
        detector.model_filenames = {"v0": object_path.name, "v1": v1_path.name}
        detector.rifle_filename = rifle_path.name
        return detector

    def available_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=model_id, name=MODEL_NAMES[model_id], filename=self.model_filenames[model_id])
                for model_id in ("v1", "v0") if model_id in self.object_models]

    def detect(
        self, image: Image.Image, classes: list[str], object_conf: float, person_conf: float,
        model_id: WeaponModelId = "v0", *, encode_annotation: bool = True,
    ) -> DetectionResult:
        if model_id not in self.object_models:
            raise ValueError(f"Weapon model {model_id} is not loaded.")
        if "rifle" in classes and self.rifle is None:
            raise ValueError("Rifle detection model is not loaded.")
        object_model = self.object_models[model_id]
        detections: list[Detection] = []
        people: list[Detection] = []
        with self.lock:
            start = time.perf_counter()
            selected_objects = [index for index, name in enumerate(OBJECT_CLASSES) if name in classes]
            models = []
            if selected_objects:
                models.append((object_model, selected_objects, object_conf, "object"))
            if "rifle" in classes and self.rifle is not None:
                models.append((self.rifle, [0], object_conf, "object"))
            if "person" in classes or any(name in WEAPON_CLASSES for name in classes):
                models.append((self.people, [0], person_conf, "person"))
            for model, class_ids, confidence, source in models:
                result = model.predict(
                    image, classes=class_ids, conf=confidence, imgsz=640,
                    device=self.device, verbose=False, save=False, max_det=100,
                )[0]
                if result.boxes is None:
                    continue
                for x1, y1, x2, y2, score, class_id in result.boxes.data.cpu().tolist():
                    label = model.names[int(class_id)]
                    if label not in classes and label != "person":
                        continue
                    detection = Detection(
                        label=label, confidence=score,
                        box=(max(0, min(x1, image.width)), max(0, min(y1, image.height)),
                             max(0, min(x2, image.width)), max(0, min(y2, image.height))),
                        source="person" if source == "person" else "object",
                    )
                    if source == "person":
                        people.append(detection)
                    if label in classes:
                        detections.append(detection)
            detections, rejected = filter_oversized_weapons(detections, people)
            elapsed = (time.perf_counter() - start) * 1000
        detections.sort(key=lambda item: item.confidence, reverse=True)
        counts = {name: sum(item.label == name for item in detections) for name in classes}
        return DetectionResult(
            width=image.width, height=image.height, inference_ms=round(elapsed, 1),
            detections=detections, counts=counts,
            weapon_count=sum(item.label in WEAPON_CLASSES for item in detections),
            false_detection_count=rejected,
            annotated_image=annotate(image, detections) if encode_annotation else "",
            model_id=model_id, model_name=MODEL_NAMES[model_id],
            rifle_model=self.rifle_filename if "rifle" in classes else None,
        )

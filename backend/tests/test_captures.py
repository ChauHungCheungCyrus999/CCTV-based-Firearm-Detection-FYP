from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from backend.captures import Capture, CapturePolicy, CaptureStore
from backend.detector import Detection, DetectionResult, annotate


@pytest.fixture
def clock(monkeypatch):
    clock = MagicMock(wraps=datetime)
    clock.now.return_value = datetime(2026, 11, 6, 14, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("backend.captures.datetime", clock)
    return clock


@pytest.fixture
def result():
    detections = [
        Detection(label="person", confidence=0.8, box=(0, 0, 30, 30), source="person"),
        Detection(label="smartphone", confidence=0.8, box=(5, 5, 15, 15), source="object"),
    ]
    return DetectionResult(
        width=32, height=32, inference_ms=5, detections=detections,
        counts={"person": 1, "smartphone": 1}, weapon_count=0,
        annotated_image=annotate(Image.new("RGB", (32, 32)), detections),
    )


def test_ten_second_cooldown_and_history_survive_restart(tmp_path, result, clock):
    store = CaptureStore(tmp_path)
    first = store.save_if_eligible(result, "image")
    assert first is not None
    assert store.save_if_eligible(result, "camera") is None
    store = CaptureStore(tmp_path)
    assert store.list(12, 0).items[0] == first
    clock.now.return_value += timedelta(seconds=9, milliseconds=999)
    assert store.save_if_eligible(result, "camera") is None
    clock.now.return_value += timedelta(milliseconds=1)
    second = store.save_if_eligible(result, "camera")
    assert second is not None
    assert second.id != first.id
    assert store.list(1, 0).total == 2
    assert store.list(1, 0).items[0].id == second.id
    assert store.list(1, 1).items[0].id == first.id
    assert store.list(1, 2).items == []
    assert len(list(tmp_path.glob("*.jpg"))) == 2


def test_concurrent_frames_save_only_one_capture(tmp_path, result):
    store = CaptureStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        captures = list(executor.map(lambda _: store.save_if_eligible(result, "camera"), range(4)))
    assert sum(capture is not None for capture in captures) == 1
    assert store.list(12, 0).total == 1
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_write_failure_does_not_create_capture_or_consume_cooldown(tmp_path, result, monkeypatch):
    store = CaptureStore(tmp_path)
    with monkeypatch.context() as context:
        context.setattr(Path, "open", MagicMock(side_effect=PermissionError("Not writable")))
        with pytest.raises(PermissionError):
            store.save_if_eligible(result, "camera")
    assert store.list(12, 0).total == 0
    assert list(tmp_path.glob("*.jpg")) == []
    assert store.save_if_eligible(result, "camera") is not None


def test_configurable_thresholds_classes_warning_and_cooldown(tmp_path, result, clock):
    store = CaptureStore(tmp_path)
    assert store.save_if_eligible(result, "camera", CapturePolicy(classes=["pistol"])) is None
    assert store.save_if_eligible(result, "camera", CapturePolicy(object_conf=0.81)) is None
    assert store.save_if_eligible(result, "camera", CapturePolicy(person_conf=0.81)) is None
    policy = CapturePolicy(classes=["smartphone"], warning_classes=["smartphone"], cooldown_seconds=20)
    capture = store.save_if_eligible(result, "camera", policy)
    assert capture is not None and capture.is_warning
    assert capture.policy.cooldown_seconds == 20
    clock.now.return_value += timedelta(seconds=19)
    assert store.save_if_eligible(result, "camera", policy) is None
    clock.now.return_value += timedelta(seconds=1)
    assert store.save_if_eligible(result, "camera", policy) is not None


def test_delete_removes_image_metadata_but_preserves_cooldown(tmp_path, result, clock):
    store = CaptureStore(tmp_path)
    capture = store.save_if_eligible(result, "image")
    assert store.delete(capture.id)
    assert not (tmp_path / capture.filename).exists()
    assert not list(tmp_path.glob(".deleting-*"))
    assert store.get(capture.id) is None
    assert store.list(12, 0).total == 0
    assert not store.delete(capture.id)
    restarted = CaptureStore(tmp_path)
    assert restarted.save_if_eligible(result, "camera") is None
    clock.now.return_value += timedelta(seconds=10)
    assert restarted.save_if_eligible(result, "camera") is not None


def test_delete_failure_preserves_capture_for_retry(tmp_path, result, monkeypatch):
    store = CaptureStore(tmp_path)
    capture = store.save_if_eligible(result, "image")
    with monkeypatch.context() as context:
        context.setattr(Path, "replace", MagicMock(side_effect=PermissionError("Not writable")))
        with pytest.raises(PermissionError):
            store.delete(capture.id)
    assert store.get(capture.id) == capture
    assert (tmp_path / capture.filename).exists()
    assert store.delete(capture.id)


def test_delete_restores_image_if_database_commit_fails(tmp_path, result, monkeypatch):
    import sqlite3
    store = CaptureStore(tmp_path)
    capture = store.save_if_eligible(result, "image")
    original_connect = sqlite3.connect

    class FailingCommit(sqlite3.Connection):
        def commit(self):
            raise sqlite3.OperationalError("Disk full")

    with monkeypatch.context() as context:
        context.setattr("backend.captures.sqlite3.connect", lambda *args, **kwargs: original_connect(*args, **kwargs, factory=FailingCommit))
        with pytest.raises(sqlite3.OperationalError):
            store.delete(capture.id)
    assert store.get(capture.id) == capture
    assert (tmp_path / capture.filename).exists()
    assert not list(tmp_path.glob(".deleting-*"))


def test_older_capture_metadata_gets_default_policy_and_warning_classification(tmp_path, result):
    result.detections[1] = Detection(label="knife", confidence=0.8, box=(1, 1, 20, 20), source="object")
    capture = CaptureStore(tmp_path).save_if_eligible(result, "camera")
    metadata = capture.model_dump()
    del metadata["policy"]
    del metadata["is_warning"]
    del metadata["model_id"]
    del metadata["model_name"]
    restored = Capture.model_validate(metadata)
    assert restored.is_warning
    assert restored.policy.object_conf == 0.8
    assert restored.model_id == "v0"
    assert restored.model_name == "Weapon detection model v0"

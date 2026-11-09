import io
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.detector import OBJECT_CLASSES, Detector, InvalidImage, decode_image
from backend.main import MAX_UPLOAD_BYTES, create_app
from backend.captures import CaptureStore


def model(names, rows):
    result = MagicMock()
    result.boxes.data.cpu.return_value.tolist.return_value = rows
    instance = MagicMock()
    instance.names = names
    instance.predict.return_value = [result]
    return instance


@pytest.fixture
def detectors():
    objects = model(dict(enumerate(OBJECT_CLASSES)), [
        [1, 2, 45, 50, 0.85, 0], [3, 5, 40, 45, 0.65, 2],
    ])
    people = model({0: "person"}, [[0, 0, 60, 60, 0.92, 0]])
    v1 = model(dict(enumerate(OBJECT_CLASSES)), [[1, 2, 45, 50, 0.95, 0]])
    rifle = model({0: "rifle"}, [[4, 6, 50, 55, 0.9, 0]])
    return Detector(objects, people, "cpu", v1, rifle)


@pytest.fixture
def capture_store(tmp_path):
    return CaptureStore(tmp_path / "captures")


@pytest.fixture
def client(detectors, capture_store):
    with TestClient(create_app(detectors, capture_store, capture_store.directory.parent / "videos")) as test_client:
        yield test_client


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "PNG")
    return buffer.getvalue()


def post(client, **data):
    return client.post("/api/detect", files={"file": ("sample.png", image_bytes(), "image/png")}, data=data)


def test_uploaded_video_frames_keep_capture_source_and_model(client):
    response = post(client, source="video", model_id="v1", auto_capture="true")
    assert response.status_code == 200
    capture = response.json()["capture"]
    assert capture["source"] == "video"
    assert capture["model_id"] == "v1"
    assert capture["is_warning"] is True
    assert client.get("/api/captures").json()["items"][0]["source"] == "video"
    assert client.get(capture["image_url"]).status_code == 200
    assert post(client, source="camera", auto_capture="true").json()["capture"] is None


def test_unknown_capture_source_is_rejected(client, detectors):
    assert post(client, source="unknown").status_code == 422
    detectors.objects.predict.assert_not_called()


def test_health_and_class_metadata(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert {item["id"] for item in client.get("/api/config").json()} == {"person", *OBJECT_CLASSES, "rifle"}
    assert client.get("/api/health").json()["rifle_model"] == "rifle_detector.pt"


def test_independent_thresholds_and_selected_classes(client, detectors):
    response = post(client, classes=json.dumps(["person", "pistol"]),
                    object_conf="0.6", person_conf="0.8")
    assert response.status_code == 200
    result = response.json()
    assert result["width"] == result["height"] == 64
    assert result["counts"] == {"person": 1, "pistol": 1}
    assert result["weapon_count"] == 1
    assert result["annotated_image"].startswith("data:image/jpeg;base64,")
    assert {item["source"] for item in result["detections"]} == {"object", "person"}
    assert detectors.objects.predict.call_args.kwargs["classes"] == [0]
    assert detectors.objects.predict.call_args.kwargs["conf"] == 0.6
    assert detectors.people.predict.call_args.kwargs["conf"] == 0.8
    assert detectors.people.predict.call_args.kwargs["classes"] == [0]
    assert detectors.objects.predict.call_args.args[0] is detectors.people.predict.call_args.args[0]


def test_oversized_weapon_uses_overlapping_person_not_larger_distant_person(client, detectors):
    detectors.object_models["v0"] = model(dict(enumerate(OBJECT_CLASSES)), [
        [0, 0, 20, 20, 0.9, 0], [1, 1, 10, 10, 0.9, 2],
    ])
    detectors.people = model({0: "person"}, [
        [0, 0, 10, 10, 0.9, 0], [30, 30, 60, 60, 0.9, 0],
    ])
    result = post(client, classes='["person","pistol","knife"]').json()
    assert result["counts"] == {"person": 2, "pistol": 0, "knife": 1}
    assert result["weapon_count"] == 1
    assert result["false_detection_count"] == 1
    assert {item["label"] for item in result["detections"]} == {"person", "knife"}


def test_weapon_size_filter_keeps_equal_area_and_nonweapons(client, detectors):
    detectors.object_models["v0"] = model(dict(enumerate(OBJECT_CLASSES)), [
        [0, 0, 20, 20, 0.9, 0], [0, 0, 50, 50, 0.9, 1],
    ])
    detectors.people = model({0: "person"}, [[0, 0, 20, 20, 0.9, 0]])
    result = post(client, classes='["person","pistol","smartphone"]').json()
    assert result["counts"] == {"person": 1, "pistol": 1, "smartphone": 1}
    assert result["false_detection_count"] == 0


def test_weapon_size_filter_uses_nearest_person_when_nothing_overlaps(client, detectors):
    detectors.object_models["v0"] = model(dict(enumerate(OBJECT_CLASSES)), [[12, 12, 32, 32, 0.9, 0]])
    detectors.people = model({0: "person"}, [
        [0, 0, 10, 10, 0.9, 0], [35, 35, 64, 64, 0.9, 0],
    ])
    result = post(client, classes='["person","pistol"]').json()
    assert result["counts"]["pistol"] == 0
    assert result["false_detection_count"] == 1


def test_weapon_without_person_is_filtered_and_cannot_trigger_capture(client, detectors):
    detectors.people = model({0: "person"}, [])
    result = post(client, classes='["person","pistol","rifle"]', auto_capture="true").json()
    assert result["counts"] == {"person": 0, "pistol": 0, "rifle": 0}
    assert result["weapon_count"] == 0
    assert result["false_detection_count"] == 2
    assert result["detections"] == []
    assert result["capture"] is None


def test_person_model_runs_hidden_when_weapon_selected(client, detectors):
    result = post(client, classes='["pistol"]').json()
    assert result["counts"] == {"pistol": 1}
    assert [item["label"] for item in result["detections"]] == ["pistol"]
    assert result["false_detection_count"] == 0
    detectors.people.predict.assert_called_once()


def test_hidden_person_filters_oversized_weapon_without_returning_person(client, detectors):
    detectors.people = model({0: "person"}, [[0, 0, 5, 5, 0.9, 0]])
    result = post(client, classes='["pistol"]').json()
    assert result["counts"] == {"pistol": 0}
    assert result["detections"] == []
    assert result["false_detection_count"] == 1
    detectors.people.predict.assert_called_once()


def test_nonweapon_without_person_is_unaffected(client, detectors):
    detectors.object_models["v0"] = model(dict(enumerate(OBJECT_CLASSES)), [[0, 0, 50, 50, 0.9, 1]])
    result = post(client, classes='["smartphone"]').json()
    assert result["counts"] == {"smartphone": 1}
    assert result["false_detection_count"] == 0
    detectors.people.predict.assert_not_called()


@pytest.mark.parametrize("selection,unused", [(["person"], "objects"), (["smartphone"], "people")])
def test_unselected_model_is_not_run(client, detectors, selection, unused):
    assert post(client, classes=json.dumps(selection)).status_code == 200
    getattr(detectors, unused).predict.assert_not_called()


@pytest.mark.parametrize("classes", ["[]", '["unknown"]', "{}", "bad", "[1]", '["person",{}]'])
def test_invalid_class_selection(client, classes):
    assert post(client, classes=classes).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("object_conf", "0"), ("object_conf", "1.01"), ("person_conf", "-0.1"),
    ("person_conf", "nan"), ("object_conf", "inf"),
])
def test_invalid_thresholds(client, field, value):
    assert post(client, **{field: value}).status_code == 422


@pytest.mark.parametrize("content,status", [
    (b"", 400), (b"not an image", 415), (b"x" * (MAX_UPLOAD_BYTES + 1), 413),
])
def test_invalid_uploads(client, content, status):
    response = client.post("/api/detect", files={"file": ("bad.png", content)})
    assert response.status_code == status
    assert response.json()["detail"]


def test_no_detection_is_valid(client, detectors):
    detectors.objects.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = []
    result = post(client, classes='["pistol"]').json()
    assert result["detections"] == []
    assert result["weapon_count"] == 0


def test_reject_unsupported_image():
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, "GIF")
    with pytest.raises(InvalidImage, match="Unsupported"):
        decode_image(buffer.getvalue())


def test_exif_orientation_is_normalized_before_inference():
    buffer = io.BytesIO()
    image = Image.new("RGB", (64, 32))
    exif = image.getexif()
    exif[274] = 6
    image.save(buffer, "JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).size == (32, 64)


def test_model_class_contract_is_validated(monkeypatch, tmp_path):
    from backend.detector import Detector

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setenv("OBJECT_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("OBJECT_MODEL_V1_PATH", str(checkpoint))
    monkeypatch.setenv("PERSON_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("RIFLE_MODEL_PATH", str(checkpoint))
    invalid_model = MagicMock(task="detect", names={0: "unexpected"})
    monkeypatch.setattr("backend.detector.YOLO", lambda _: invalid_model)
    with pytest.raises(RuntimeError, match="Expected SoHAS classes"):
        Detector.load()


def test_server_failure_is_visible(detectors, capture_store):
    detectors.objects.predict.side_effect = RuntimeError("inference failed")
    with TestClient(create_app(detectors, capture_store), raise_server_exceptions=False) as client:
        assert post(client).status_code == 500


def test_capture_is_opt_in_for_api_clients(client, capture_store):
    assert post(client).json()["capture"] is None
    assert capture_store.list(12, 0).total == 0


def test_auto_capture_saves_and_serves_annotated_jpeg(client, capture_store):
    result = post(client, auto_capture="true", source="camera").json()
    capture = result["capture"]
    assert result["capture_error"] is None
    assert capture["source"] == "camera"
    assert capture["object_confidence"] == 0.85
    assert capture["person_confidence"] == 0.92
    file = capture_store.directory / capture["filename"]
    assert file.is_file()
    with Image.open(file) as image:
        assert image.format == "JPEG"
        assert image.size == (64, 64)
    gallery = client.get("/api/captures").json()
    assert gallery["total"] == 1
    assert gallery["items"][0]["id"] == capture["id"]
    inline = client.get(capture["image_url"])
    assert inline.status_code == 200
    assert inline.headers["content-type"] == "image/jpeg"
    assert inline.headers["content-disposition"].startswith("inline;")
    assert inline.content == file.read_bytes()
    downloaded = client.get(capture["image_url"] + "?download=true")
    assert downloaded.headers["content-disposition"].startswith("attachment;")


@pytest.mark.parametrize("object_conf,person_conf,eligible", [
    (0.8, 0.8, True),
    (0.79999, 0.99, False),
    (0.99, 0.79999, False),
    (0.8, 0.1, False),
    (0.1, 0.8, False),
])
def test_capture_exact_thresholds(client, detectors, object_conf, person_conf, eligible):
    detectors.objects.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = [
        [1, 2, 45, 50, object_conf, 0],
    ]
    detectors.people.predict.return_value[0].boxes.data.cpu.return_value.tolist.return_value = [
        [0, 0, 60, 60, person_conf, 0],
    ]
    response = post(client, auto_capture="true")
    assert response.status_code == 200
    assert (response.json()["capture"] is not None) is eligible


@pytest.mark.parametrize("classes", ['["person"]', '["pistol"]', '["knife"]'])
def test_capture_requires_both_models_in_same_frame(client, classes):
    assert post(client, auto_capture="true", classes=classes).json()["capture"] is None


def test_capture_does_not_combine_objects_and_people_from_different_frames(client, detectors):
    objects = detectors.objects.predict.return_value[0].boxes.data.cpu.return_value
    people = detectors.people.predict.return_value[0].boxes.data.cpu.return_value
    people.tolist.return_value = []
    assert post(client, auto_capture="true").json()["capture"] is None
    objects.tolist.return_value = []
    people.tolist.return_value = [[0, 0, 60, 60, 0.95, 0]]
    assert post(client, auto_capture="true").json()["capture"] is None


def test_capture_gallery_validation_and_missing_files(client, capture_store):
    assert client.get("/api/captures?limit=0").status_code == 422
    assert client.get("/api/captures?limit=101").status_code == 422
    assert client.get("/api/captures?offset=-1").status_code == 422
    assert client.get("/api/captures/not-a-uuid/image").status_code == 422
    assert client.get("/api/captures/00000000-0000-0000-0000-000000000000/image").status_code == 404
    capture = post(client, auto_capture="true").json()["capture"]
    (capture_store.directory / capture["filename"]).unlink()
    assert client.get(capture["image_url"]).status_code == 404


def test_capture_failure_is_visible_without_losing_detection(client, capture_store, monkeypatch):
    def fail(*args):
        raise OSError("Disk full")
    monkeypatch.setattr(capture_store, "save_if_eligible", fail)
    response = post(client, auto_capture="true")
    assert response.status_code == 200
    assert response.json()["detections"]
    assert response.json()["capture"] is None
    assert "could not be saved" in response.json()["capture_error"]


def test_configurable_capture_rules_and_warning_classes(client):
    assert post(client, auto_capture="true", capture_object_conf="0.86").json()["capture"] is None
    assert post(client, auto_capture="true", capture_classes='["knife"]').json()["capture"] is None
    result = post(client, auto_capture="true", capture_classes='["knife"]',
                  capture_object_conf="0.6", capture_person_conf="0.9",
                  capture_cooldown="20", warning_classes='["knife"]').json()
    capture = result["capture"]
    assert capture["object_confidence"] == 0.65
    assert capture["is_warning"]
    assert capture["policy"]["classes"] == ["knife"]
    assert capture["policy"]["cooldown_seconds"] == 20


@pytest.mark.parametrize("field,value", [
    ("capture_object_conf", "0"), ("capture_object_conf", "nan"),
    ("capture_person_conf", "1.1"), ("capture_cooldown", "0"),
    ("capture_cooldown", "61"), ("capture_cooldown", "1.5"),
    ("capture_classes", "[]"), ("capture_classes", '["person"]'),
    ("warning_classes", '["unknown"]'), ("warning_classes", "bad"),
])
def test_capture_settings_validation(client, field, value):
    assert post(client, auto_capture="true", **{field: value}).status_code == 422


def test_user_can_disable_warning_classification(client):
    capture = post(client, auto_capture="true", warning_classes="[]").json()["capture"]
    assert capture is not None
    assert not capture["is_warning"]


def test_delete_capture_endpoint(client, capture_store):
    capture = post(client, auto_capture="true").json()["capture"]
    response = client.delete(f'/api/captures/{capture["id"]}')
    assert response.status_code == 200
    assert response.json()["deleted_id"] == capture["id"]
    assert not (capture_store.directory / capture["filename"]).exists()
    assert client.get("/api/captures").json()["total"] == 0
    assert client.get(capture["image_url"]).status_code == 404
    assert client.delete(f'/api/captures/{capture["id"]}').status_code == 404
    assert client.delete("/api/captures/../../README.md").status_code in {404, 405}
    assert client.delete("/api/captures/not-a-uuid").status_code == 422


def test_delete_failure_is_visible(client, capture_store, monkeypatch):
    capture = post(client, auto_capture="true").json()["capture"]
    def fail(*args):
        raise OSError("Not writable")
    monkeypatch.setattr(capture_store, "delete", fail)
    response = client.delete(f'/api/captures/{capture["id"]}')
    assert response.status_code == 500
    assert "Could not delete" in response.json()["detail"]
    assert client.get("/api/captures").json()["total"] == 1


def test_available_model_versions(client):
    assert client.get("/api/models").json() == [
        {"id": "v1", "name": "Weapon detection model v1", "filename": "last.pt"},
        {"id": "v0", "name": "Weapon detection model v0", "filename": "best.pt"},
    ]


def test_default_model_is_v0(client, detectors):
    result = post(client).json()
    assert result["model_id"] == "v0"
    assert result["model_name"] == "Weapon detection model v0"
    detectors.objects.predict.assert_called_once()
    detectors.object_models["v1"].predict.assert_not_called()


def test_v1_selection_runs_only_v1_and_shared_person_model(client, detectors):
    result = post(client, model_id="v1", object_conf="0.6", person_conf="0.7").json()
    assert result["model_id"] == "v1"
    assert result["model_name"] == "Weapon detection model v1"
    detectors.objects.predict.assert_not_called()
    detectors.object_models["v1"].predict.assert_called_once()
    assert detectors.object_models["v1"].predict.call_args.kwargs["conf"] == 0.6
    assert detectors.people.predict.call_args.kwargs["conf"] == 0.7
    assert detectors.object_models["v1"].predict.call_args.args[0] is detectors.people.predict.call_args.args[0]


@pytest.mark.parametrize("model_id", ["v2", "../last.pt", "best.pt", ""])
def test_reject_invalid_model_selection(client, model_id):
    assert post(client, model_id=model_id).status_code == 422


def test_reject_unavailable_model_without_falling_back(client, detectors):
    del detectors.object_models["v1"]
    response = post(client, model_id="v1")
    assert response.status_code == 422
    assert "not loaded" in response.json()["detail"]
    detectors.objects.predict.assert_not_called()


def test_model_version_is_saved_with_capture(client, capture_store):
    capture = post(client, model_id="v1", auto_capture="true").json()["capture"]
    assert capture["model_id"] == "v1"
    assert capture["model_name"] == "Weapon detection model v1"
    assert capture_store.get(capture["id"]).model_id == "v1"
    assert client.get("/api/captures").json()["items"][0]["model_name"] == "Weapon detection model v1"


def test_selected_model_is_not_run_for_person_only(client, detectors):
    assert post(client, model_id="v1", classes='["person"]').status_code == 200
    detectors.objects.predict.assert_not_called()
    detectors.object_models["v1"].predict.assert_not_called()
    detectors.people.predict.assert_called_once()


def test_switching_versions_is_per_request(client, detectors):
    assert post(client, model_id="v1").json()["model_id"] == "v1"
    assert post(client, model_id="v0").json()["model_id"] == "v0"
    assert detectors.objects.predict.call_count == 1
    assert detectors.object_models["v1"].predict.call_count == 1


def test_v1_class_contract_is_validated_without_substitution(monkeypatch, tmp_path):
    paths = [tmp_path / filename for filename in ("best.pt", "last.pt", "yolo11s.pt", "rifle_detector.pt")]
    for path in paths:
        path.touch()
    for variable, path in zip(("OBJECT_MODEL_PATH", "OBJECT_MODEL_V1_PATH", "PERSON_MODEL_PATH", "RIFLE_MODEL_PATH"), paths):
        monkeypatch.setenv(variable, str(path))
    valid_v0 = MagicMock(task="detect", names=dict(enumerate(OBJECT_CLASSES)))
    invalid_v1 = MagicMock(task="detect", names={0: "unexpected"})
    loader = MagicMock(side_effect=[valid_v0, invalid_v1])
    monkeypatch.setattr("backend.detector.YOLO", loader)
    with pytest.raises(RuntimeError, match="last.pt"):
        Detector.load()
    assert loader.call_count == 2


@pytest.mark.parametrize("version", ["v0", "v1"])
def test_rifle_runs_alongside_selected_weapon_model_on_the_same_frame(client, detectors, version):
    result = post(client, classes='["person","pistol","rifle"]', model_id=version,
                  object_conf="0.6", person_conf="0.7").json()
    assert result["counts"] == {"person": 1, "pistol": 1, "rifle": 1}
    assert result["weapon_count"] == 2
    assert result["rifle_model"] == "rifle_detector.pt"
    assert result["model_id"] == version
    selected = detectors.object_models[version]
    detectors.object_models["v1" if version == "v0" else "v0"].predict.assert_not_called()
    assert detectors.rifle.predict.call_args.kwargs["classes"] == [0]
    assert detectors.rifle.predict.call_args.kwargs["conf"] == 0.6
    assert detectors.rifle.predict.call_args.args[0] is selected.predict.call_args.args[0]
    assert detectors.rifle.predict.call_args.args[0] is detectors.people.predict.call_args.args[0]


def test_unselected_rifle_model_is_skipped(client, detectors):
    result = post(client, classes='["person","pistol"]').json()
    assert result["rifle_model"] is None
    detectors.rifle.predict.assert_not_called()


def test_rifle_capture_is_a_warning_and_preserves_both_model_identities(client, detectors, capture_store):
    result = post(client, classes='["person","rifle"]', model_id="v1",
                  auto_capture="true", capture_classes='["rifle"]').json()
    capture = result["capture"]
    assert capture["is_warning"] is True
    assert capture["rifle_model"] == "rifle_detector.pt"
    assert capture["model_id"] == "v1"
    assert capture_store.get(capture["id"]).rifle_model == "rifle_detector.pt"
    assert client.get("/api/captures").json()["items"][0]["rifle_model"] == "rifle_detector.pt"
    detectors.objects.predict.assert_not_called()
    detectors.object_models["v1"].predict.assert_not_called()


def test_missing_rifle_model_is_an_explicit_request_error(client, detectors):
    detectors.rifle = None
    response = post(client, classes='["person","rifle"]')
    assert response.status_code == 422
    assert "Rifle detection model is not loaded" in response.json()["detail"]
    detectors.people.predict.assert_not_called()


def test_load_validates_rifle_class_and_preserves_shared_detector(monkeypatch, tmp_path):
    paths = [tmp_path / filename for filename in ("best.pt", "last.pt", "yolo11s.pt", "rifle_detector.pt")]
    for variable, path in zip(("OBJECT_MODEL_PATH", "OBJECT_MODEL_V1_PATH", "PERSON_MODEL_PATH", "RIFLE_MODEL_PATH"), paths):
        path.touch()
        monkeypatch.setenv(variable, str(path))
    objects = MagicMock(task="detect", names=dict(enumerate(OBJECT_CLASSES)))
    rifle = MagicMock(task="detect", names={0: "rifle"})
    people = MagicMock(task="detect", names={index: "person" if index == 0 else f"class-{index}" for index in range(80)})
    loader = MagicMock(side_effect=[objects, objects, rifle, people])
    monkeypatch.setattr("backend.detector.YOLO", loader)
    detector = Detector.load()
    assert detector.rifle is rifle
    assert detector.people is people
    assert detector.rifle_filename == "rifle_detector.pt"
    rifle.names = {0: "pistol"}
    loader.side_effect = [objects, objects, rifle]
    with pytest.raises(RuntimeError, match="single rifle class"):
        Detector.load()

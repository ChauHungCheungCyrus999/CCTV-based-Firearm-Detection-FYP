# CCTV-based Firearm Detection

A local full-stack detection workspace branded **Visible Weapon Detection for Hong Kong Surveillance**, built with React, TypeScript, Material UI, and FastAPI. Choose between two trained object models, with a shared rifle detector and the notebook's separate COCO `yolo11s.pt` model for person detection.

## Features

- Upload, drag-and-drop, or paste a JPEG, PNG, or WebP image (10 MB / 20 megapixel limit).
- Side-by-side original and annotated previews, confidence scores, detection table, and downloadable results.
- Select any combination of person, pistol, rifle, smartphone, knife, purse, banknote, and card.
- Pistol, knife, and rifle boxes are suppressed if no person is found or if their clipped box area is strictly larger than the associated person's box. Association favors the most-overlapping person (highest IoU), falling back to the nearest box center. Person inference still runs when weapons are selected but Person is hidden; hidden person boxes/counts are not displayed. Rejected boxes do not trigger captures and are counted in API results and cumulative video-job metadata as `false_detection_count`, not shown in the UI. This is a size heuristic, not verified ground truth; it can reject real weapons when people are missed or framed differently.
- **Rifle detection model** (`rifle_detector.pt`, copied from your rifle-only `best (1).pt`) runs alongside the selected v0/v1 model when Rifle is selected. Rifle is enabled by default in the website.
- Independent `object_conf` and `person_conf` sliders; defaults match the notebook (0.25). `object_conf` applies to both the selected object model and the rifle detector.
- **Weapon detection model v0** (`best.pt`) or **Weapon detection model v1** (`last.pt`) selected independently per image, camera frame, or whole-video job.
- Browser webcam preview with automatic, sequential frame inference and stop/release controls.
- Upload or drop an MP4, WebM, or MOV CCTV video (100 MiB limit), click **Run detection**, and play/download a separate annotated H.264 MP4 with original audio retained. Includes progress, cancellation, recovered recent jobs, and the same screenshot settings.
- Automatic screenshots with adjustable object/person capture thresholds, capture classes, and cooldown (website defaults: **35% object**, **55% person**, pistol/knife/rifle only, **2 seconds**).
- **Check captures** gallery with timestamps, full-size previews, downloads, and **X** delete buttons with permanent-deletion confirmation.
- Configurable warning-class pop-up bubbles (pistol, knife, and rifle by default), normal capture notifications, and a notification toggle. Responsive notifications have a separate close button and slide in/out; reduced-motion preferences are respected.
- Class-specific icons for person, pistol, knife, smartphone, purse, banknote, and card.
- Visible upload, API, model, capture-storage, and camera permission errors.

## Setup

Requirements: Python 3.11–3.13, Node.js 22.18+ and npm. Commands below run from the project directory.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
npm install

mkdir -p backend/models
cp ~/Downloads/best.pt backend/models/best.pt
cp ~/Downloads/last.pt backend/models/last.pt
cp "$HOME/Downloads/best (1).pt" backend/models/rifle_detector.pt
curl --fail --location \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt \
  --output backend/models/yolo11s.pt
```

Only load trusted PyTorch checkpoints. Your supplied models are copied, not modified. Both object checkpoints must have the notebook's exact six-class order. The API validates all four models at startup and does not silently substitute weights or accept client-provided model paths.

## Run

Terminal 1:

```sh
npm run backend
```

Terminal 2:

```sh
npm run dev
```

Open **http://127.0.0.1:5173**. The Vite server proxies `/api` to FastAPI on port 8000, including in `npm run preview`. Initial model loading / first inference can take longer. If the interface is opened before the API is ready, use **Reconnect**.

1. Choose **Weapon detection model v1** or **Weapon detection model v0** in the model selector, select classes, and adjust the two confidence sliders. v1 is selected by default in the website.
2. Upload, drop, or paste an image; click **Run detection**. The original appears on the left and the annotated result on the right.
3. Or select **Live camera**, click **Open camera**, and grant browser permission. Detection starts automatically. Changed settings apply to the next frame. **Stop camera** releases the webcam.
   For recorded CCTV footage, select **Video**, upload or drop a video, and click **Run detection**. Watch progress, then play the separate **Annotated video** or click **Download video**. **Cancel processing** stops a running job. Preview playback does not trigger inference. Switching tabs pauses playback but leaves processing running; return to Video to check progress. Recent jobs can be selected after a browser reload.
   When auto-capture is enabled and a processed frame meets the object, person, and cooldown rules, the existing capture notification appears while processing. It stays visible until dismissed or the job finishes; the annotated video has no alert banner and becomes playable only when processing finishes.
4. **Auto-capture screenshots** is enabled in the website by default. Keep **Person** and your chosen capture object classes selected in detection settings. A matching object and person in the same frame save an annotated JPEG when they meet the capture thresholds.
5. Open **Capture settings** in the sidebar or **Check captures** gallery to choose capture classes, object/person capture thresholds (1–100%), cooldown (1–60 seconds), warning classes, and whether pop-up notifications are enabled. Changes apply to the next request; an in-flight frame or entire video job keeps its submitted settings. These dashboard controls reset to defaults on browser reload.
6. Saved captures matching a configured warning class above the object capture threshold display a **Warning capture saved** pop-up bubble. Other captures display a normal notification. The bubble links to the gallery and can be dismissed.
7. Click **Check captures** (or **View captures** in the notification) to browse saved screenshots, open a full-size preview, or download them. Use the **X** on a thumbnail or full-size view and confirm **Delete permanently** to remove both the image and its gallery entry. Cancelling preserves it. The gallery persists across browser and backend restarts.

The camera uses video only (no microphone). Browser camera access requires localhost or HTTPS; plain HTTP on a LAN IP will not work. The camera view shows the live source plus the most recently processed frame; it is not a synchronized high-FPS video stream. Frames are capped at 1280 pixels wide, inferred at 640 pixels, and sent one at a time without a growing queue. Each request encodes/uploads a JPEG, runs up to three sequential detectors on CPU by default, and returns an annotated JPEG. The previous artificial 120 ms pause has been removed: the next request starts after the previous response. This reduces avoidable latency but does not make inference a 30 FPS stream. Deselecting unused classes skips their detector. Processing a video simultaneously competes for the same model lock and can slow the camera.

Videos are uploaded in full to the local FastAPI backend, not a cloud service. Every decoded frame is analyzed using a fixed settings snapshot. The output preserves source frame presentation times, including variable-frame-rate clips, and resolution (odd dimensions are padded for H.264 compatibility), draws boxes, and retains source audio when present, transcoded to AAC with aligned timestamps. Clips without audio remain silent. Source browser preview depends on its codec, but server processing can still run without a playable source preview. Encoding uses bundled FFmpeg; no system FFmpeg installation is required.

Only one video job runs at a time. Long clips can take longer than their original duration on CPU. Outputs and job metadata persist in `video_jobs/`; uploaded sources and temporary encoding files are removed when a job ends. Completed outputs are retained for replay/download, so monitor disk space. Interrupted jobs are marked failed after a backend restart; partial outputs are never offered as completed results. Saved screenshots appear in the gallery as **Uploaded video** and use the shared thresholds, warning rules, and wall-clock cooldown.

## Model versions

| Dashboard name | Object checkpoint | Person checkpoint |
| --- | --- | --- |
| Weapon detection model v0 | `backend/models/best.pt` | Shared `backend/models/yolo11s.pt` |
| Weapon detection model v1 | `backend/models/last.pt` | Shared `backend/models/yolo11s.pt` |

All models are loaded once at startup. Only the selected v0/v1 object model runs for each request; the shared rifle and person detectors run when their classes are selected. Version changes apply to the next camera frame or next **Run detection**, not an already running video job. In-flight results and saved captures retain the version that actually processed them. Choices are per browser/request, not a server-wide switch. Model selection resets to v1 on browser reload or **Reset to defaults**; API requests that omit `model_id` still default to v0. Both versions support pistol, smartphone, knife, purse, banknote, and card.

The v1 name identifies your newly supplied checkpoint; it does not assert better accuracy. `last.pt` is a final-epoch checkpoint, while `best.pt` may be selected by validation metrics. Compare the models on representative data before choosing a deployment model.

### Shared rifle detector

The supplied rifle checkpoint is installed as `backend/models/rifle_detector.pt` without changing the original download or either existing weapon checkpoint. It has exactly one class: `0: rifle`. It is an auxiliary detector, not a third version in the dropdown. Keep **Rifle** selected to run it on the same original frame as the selected v0/v1 model and shared person detector. Deselecting Rifle skips rifle inference. The existing v0/v1 choice, confidence filters, image/camera/video modes, and shared inference lock are preserved.

Rifles count as weapon predictions and can trigger automatic captures and warning notifications together with a sufficiently confident person in the same frame. Rifle is included in the new default capture and warning filters. Results and saved captures record the auxiliary checkpoint in `rifle_model`; older captures default to no rifle model and retain their original saved policies.

## Capture storage

Screenshots are real annotated image/frame files, not screenshots of the browser UI. They are stored in the project's [`captures/`](./captures/) folder, next to `backend/` and `src/`. Filenames contain a UTC timestamp and a unique ID. `captures.sqlite3` in the same folder indexes their metadata for the gallery; retain it with the JPEGs. Both are excluded from source control.

The trigger uses unrounded prediction scores: at least one object from the capture-class filter plus a person must meet their independent capture thresholds in the same inference result. These classes must also be enabled on the detection dashboard. Higher detection confidence sliders still filter predictions before capture rules are evaluated. Capture metadata records the model version, rule snapshot, and warning classification; older captures retain v0 and the original 80% defaults.

The cooldown compares the current request's configured interval with the last successful capture across the backend, shared between uploaded images, videos, cameras, and clients. Its timestamp survives server restarts and deletion of the latest image. The website defaults to 2 seconds; API requests that omit capture settings still default to 10 seconds. Unchecking auto-capture stops saving for subsequent requests but does not delete existing files. With no capture classes selected, automatic capture is paused.

If the disk is full or unwritable, detection continues and the website displays the capture error. Deletion errors are displayed in the confirmation dialog for retry. Storage is not automatically pruned; monitor disk usage and retain images only as appropriate for your privacy obligations. Warning-class matches are model predictions, not proof of danger or possession; always verify the saved image.

## Configuration

Set environment variables before starting the backend:

| Variable | Default | Description |
| --- | --- | --- |
| `OBJECT_MODEL_PATH` | `backend/models/best.pt` | Trusted six-class checkpoint |
| `OBJECT_MODEL_V1_PATH` | `backend/models/last.pt` | Trusted v1 six-class checkpoint |
| `PERSON_MODEL_PATH` | `backend/models/yolo11s.pt` | Original COCO person detector |
| `RIFLE_MODEL_PATH` | `backend/models/rifle_detector.pt` | Trusted single-class rifle checkpoint, shared with v0/v1 |
| `DETECTION_DEVICE` | `cpu` | `cpu`, `mps` (Apple Silicon), or CUDA device such as `0` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed browser origins |

CPU is the portable default; test `mps` or CUDA on your machine before relying on it.

## API

- `GET /api/health`: loaded model status and device.
- `GET /api/config`: supported class names, categories, and colors.
- `GET /api/models`: loaded object model IDs, dashboard names, and checkpoint filenames.
- `POST /api/detect`: multipart image/frame `file`, `model_id` (`v0` or `v1`, default `v0`), `classes` (JSON array), `object_conf`, `person_conf` (each 0.01–1.0), `auto_capture` (boolean, API default `false`), and `source` (`image`, `camera`, or `video`, default `image`).
- `POST /api/videos`: whole-video multipart `file` (up to 100 MiB) plus the same detection/capture settings; returns a background job. Busy processing returns 409.
- `GET /api/videos`: recent job records for recovering progress/results.
- `GET /api/videos/{id}`: status, processed/total frames, progress, latest capture, capture count, and explicit errors.
- `POST /api/videos/{id}/cancel`: request cancellation.
- `GET /api/videos/{id}/result`: completed H.264 MP4 with range support for playback; add `?download=true` to download.
- Capture form fields: `capture_object_conf`, `capture_person_conf` (0.01–1.0, defaults 0.8), `capture_cooldown` (integer 1–60, default 10), `capture_classes` (JSON array of object names, defaults all six), and `warning_classes` (JSON array, defaults pistol/knife; empty disables warnings). An empty capture-class array is allowed only when auto-capture is disabled.
- `GET /api/captures?limit=12&offset=0`: newest-first capture metadata (`items`, `total`); limit 1–100.
- `GET /api/captures/{id}/image`: saved JPEG; add `?download=true` for a file download.
- `DELETE /api/captures/{id}`: permanently remove image and metadata; returns `deleted_id`. A missing capture returns 404.

```sh
curl --fail http://127.0.0.1:8000/api/detect \
  -F 'file=@/path/to/image.jpg' \
  -F 'model_id=v1' \
  -F 'classes=["person","pistol","knife"]' \
  -F 'object_conf=0.25' -F 'person_conf=0.35' \
  -F 'auto_capture=true' -F 'source=image' \
  -F 'capture_object_conf=0.8' -F 'capture_person_conf=0.8' \
  -F 'capture_cooldown=10' -F 'capture_classes=["pistol","knife"]' \
  -F 'warning_classes=["pistol","knife"]'
```

Returns `model_id`, `model_name`, `width`, `height`, `inference_ms`, `detections` (label, confidence, pixel `xyxy` box, model source), `counts`, `weapon_count`, `false_detection_count`, an annotated JPEG data URL, `capture` (saved metadata including model version, `policy` and `is_warning`, or `null`), and `capture_error` (storage error or `null`). No capture is created for nonmatching frames or frames within the cooldown. The selected object model and, when Person or a weapon is selected, the shared person model receive the same EXIF-corrected RGB image. Other unselected models are skipped. Inference is serialized to protect the shared model instances. Uploads are decoded by content, not trusted by filename or MIME type.

## Validation

```sh
.venv/bin/python -m pytest backend/tests -q
npm test
npm run build
```

## Limits and responsible use

This is a local development application, not an authenticated or hardened public service. The servers bind to loopback by default. Before exposing it, add authentication, request-body limits at the reverse proxy, TLS, rate limiting, resource controls, and an appropriate privacy policy. Do not expose the FastAPI port directly to the internet.

Confidence scores are model predictions, not calibrated safety probabilities. The notebook's 0.25 thresholds are preview defaults, not deployment recommendations. False positives and missed detections are possible. A person box does not establish who holds a weapon, and no detection does not mean a scene is safe. Use human verification; do not use this prototype as an autonomous safety or enforcement system.

Check Ultralytics' AGPL-3.0 / enterprise licensing and your training dataset's usage terms before distribution or commercial deployment.

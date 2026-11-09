import type { Capture, DetectionResult, Settings, VideoJob } from './types';

export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
export const MAX_VIDEO_BYTES = 100 * 1024 * 1024;

export function validateImage(file: File): string | null {
  if (!IMAGE_TYPES.includes(file.type)) return 'Choose a JPEG, PNG, or WebP image.';
  if (!file.size) return 'This image is empty. Choose another file.';
  if (file.size > MAX_IMAGE_BYTES) return 'Image exceeds the 10 MB upload limit.';
  return null;
}

export function validateVideo(file: File): string | null {
  if (!['video/mp4', 'video/webm', 'video/quicktime'].includes(file.type)) return 'Choose an MP4, WebM, or MOV video.';
  if (!file.size) return 'This video is empty. Choose another file.';
  if (file.size > MAX_VIDEO_BYTES) return 'Video exceeds the 100 MB upload limit.';
  return null;
}

export async function captureVideoFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<Blob> {
  canvas.width = Math.min(video.videoWidth, 1280);
  canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Your browser does not support video frame capture.');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((value) => value ? resolve(value)
      : reject(new Error('Could not capture the video frame.')), 'image/jpeg', 0.85);
  });
}

export async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    if (body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string') {
      throw new Error(body.detail);
    }
    throw new Error(response.status === 422
      ? 'Invalid detection settings. Select classes and use confidence values from 1% to 100%.'
      : `Server request failed (${response.status}). Check the backend and try again.`);
  }
  return response.json() as Promise<T>;
}

function detectionForm(settings: Settings): FormData {
  const form = new FormData();
  form.append('model_id', settings.modelId);
  form.append('classes', JSON.stringify(settings.classes));
  form.append('object_conf', String(settings.objectConf));
  form.append('person_conf', String(settings.personConf));
  form.append('auto_capture', String(settings.autoCapture && settings.capture.classes.length > 0));
  form.append('capture_object_conf', String(settings.capture.objectConf));
  form.append('capture_person_conf', String(settings.capture.personConf));
  form.append('capture_cooldown', String(settings.capture.cooldownSeconds));
  form.append('capture_classes', JSON.stringify(settings.capture.classes));
  form.append('warning_classes', JSON.stringify(settings.capture.warningClasses));
  return form;
}

export async function detectImage(
  image: Blob, settings: Settings, signal: AbortSignal, source: Capture['source'] = 'image',
): Promise<DetectionResult> {
  const form = detectionForm(settings);
  form.append('file', image, 'frame.jpg');
  form.append('source', source);
  const response = await fetch('/api/detect', {
    method: 'POST', body: form,
    signal: AbortSignal.any([signal, AbortSignal.timeout(90_000)]),
  });
  return readResponse<DetectionResult>(response);
}

export async function submitVideo(file: File, settings: Settings, signal: AbortSignal): Promise<VideoJob> {
  const form = detectionForm(settings);
  form.append('file', file, file.name);
  const response = await fetch('/api/videos', {
    method: 'POST', body: form,
    signal: AbortSignal.any([signal, AbortSignal.timeout(600_000)]),
  });
  return readResponse<VideoJob>(response);
}

export async function readVideoJob(id: string, signal: AbortSignal): Promise<VideoJob> {
  return readResponse<VideoJob>(await fetch(`/api/videos/${id}`, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(10_000)]),
  }));
}

export async function cancelVideoJob(id: string): Promise<VideoJob> {
  return readResponse<VideoJob>(await fetch(`/api/videos/${id}/cancel`, {
    method: 'POST', signal: AbortSignal.timeout(10_000),
  }));
}

export function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === 'TimeoutError') {
    return 'The API timed out. Check that the backend is ready, then try again.';
  }
  if (error instanceof TypeError) return 'Cannot reach the API. Start the FastAPI backend and retry.';
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
}

export interface ClassInfo {
  id: string;
  label: string;
  color: string;
  category: 'weapon' | 'person' | 'object';
}

export interface Settings {
  modelId: WeaponModelId;
  classes: string[];
  objectConf: number;
  personConf: number;
  autoCapture: boolean;
  capture: CaptureSettings;
}

export type WeaponModelId = 'v0' | 'v1';

export interface WeaponModelInfo {
  id: WeaponModelId;
  name: string;
  filename: string;
}

export interface CaptureSettings {
  objectConf: number;
  personConf: number;
  cooldownSeconds: number;
  classes: string[];
  warningClasses: string[];
  notifications: boolean;
}

export const DEFAULT_CAPTURE_SETTINGS: CaptureSettings = {
  objectConf: 0.35, personConf: 0.55, cooldownSeconds: 2,
  classes: ['pistol', 'knife', 'rifle'],
  warningClasses: ['pistol', 'knife', 'rifle'], notifications: true,
};

export interface Detection {
  label: string;
  confidence: number;
  box: [number, number, number, number];
  source: 'object' | 'person';
}

export interface DetectionResult {
  model_id: WeaponModelId;
  model_name: string;
  rifle_model?: string | null;
  width: number;
  height: number;
  inference_ms: number;
  detections: Detection[];
  counts: Record<string, number>;
  weapon_count: number;
  false_detection_count: number;
  annotated_image: string;
  capture: Capture | null;
  capture_error: string | null;
}

export interface Capture {
  model_id: WeaponModelId;
  model_name: string;
  rifle_model?: string | null;
  id: string;
  filename: string;
  created_at: string;
  image_url: string;
  source: 'image' | 'camera' | 'video';
  width: number;
  height: number;
  object_confidence: number;
  person_confidence: number;
  detections: Detection[];
  is_warning: boolean;
  policy: {
    object_conf: number;
    person_conf: number;
    cooldown_seconds: number;
    classes: string[];
    warning_classes: string[];
  };
}

export interface CapturePage {
  items: Capture[];
  total: number;
}

export interface VideoJob {
  id: string;
  filename: string;
  model_id?: WeaponModelId | null;
  model_name?: string | null;
  rifle_model?: string | null;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  frames_processed: number;
  total_frames: number | null;
  fps: number | null;
  progress: number | null;
  video_url: string | null;
  error: string | null;
  capture: Capture | null;
  capture_count: number;
  capture_error: string | null;
  false_detection_count: number;
}

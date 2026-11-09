import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert, Box, Button, Checkbox, Chip, CircularProgress, Collapse, Divider,
  FormControlLabel, IconButton, LinearProgress, Paper, Slider, Stack, Tab, Tabs,
  MenuItem, TextField, Tooltip, Typography,
} from '@mui/material';
import {
  ArrowForwardRounded, CenterFocusStrongRounded, CheckRounded,
  CloseRounded, CloudUploadOutlined, ContentPasteRounded, DownloadRounded,
  ImageOutlined, InfoOutlined, LockOutlined, MemoryRounded, PhotoCameraOutlined,
  PlayArrowRounded, RefreshRounded, SettingsOutlined, ShieldOutlined, StopRounded,
  TuneRounded, VideocamOutlined, VisibilityOutlined,
} from '@mui/icons-material';
import { captureVideoFrame, detectImage, errorMessage, readResponse, validateImage, validateVideo } from './api';
import ClassIcon from './ClassIcon';
import CaptureNotification from './CaptureNotification';
import UploadedVideo from './UploadedVideo';
import { DEFAULT_CAPTURE_SETTINGS } from './types';
import type { Capture, ClassInfo, DetectionResult, Settings, WeaponModelInfo } from './types';

const CapturesGallery = lazy(() => import('./CapturesGallery'));
const CaptureSettingsDialog = lazy(() => import('./CaptureSettingsDialog'));

const INITIAL_SETTINGS: Settings = {
  modelId: 'v1',
  classes: ['person', 'pistol', 'knife', 'rifle'], objectConf: 0.25, personConf: 0.25, autoCapture: true,
  capture: DEFAULT_CAPTURE_SETTINGS,
};
const DEFAULT_CLASSES: ClassInfo[] = [
  { id: 'person', label: 'Person', color: '#61a8ff', category: 'person' },
  { id: 'pistol', label: 'Pistol', color: '#ff6577', category: 'weapon' },
  { id: 'smartphone', label: 'Smartphone', color: '#bc9bff', category: 'object' },
  { id: 'knife', label: 'Knife', color: '#ffac5c', category: 'weapon' },
  { id: 'rifle', label: 'Rifle', color: '#ff8a80', category: 'weapon' },
  { id: 'purse', label: 'Purse', color: '#e9c66c', category: 'object' },
  { id: 'banknote', label: 'Banknote', color: '#4cdbad', category: 'object' },
  { id: 'card', label: 'Card', color: '#5dd6ed', category: 'object' },
];

interface ResultView {
  data: DetectionResult;
  settingsKey: string;
}

function ConfidenceControl({ label, code, value, disabled, onChange }: {
  label: string; code: string; value: number; disabled: boolean;
  onChange: (value: number) => void;
}) {
  return <Box className={`confidence-control ${disabled ? 'muted' : ''}`}>
    <Stack direction="row" justifyContent="space-between" alignItems="center">
      <Typography variant="body2" fontWeight={550}>{label}</Typography>
      <span className="threshold-value">{Math.round(value * 100)}<small>%</small></span>
    </Stack>
    <Typography className="code-label">{code}</Typography>
    <Slider
      aria-label={label} min={1} max={100} value={Math.round(value * 100)}
      disabled={disabled} onChange={(_, next) => onChange(Number(next) / 100)}
      valueLabelDisplay="auto" valueLabelFormat={(next) => `${next}%`}
      sx={{ mt: 0.6, mb: 0, py: 1.6, '& .MuiSlider-thumb': { width: 13, height: 13 } }}
    />
    <Stack direction="row" justifyContent="space-between" className="range-labels">
      <span>More detections</span><span>Higher certainty</span>
    </Stack>
  </Box>;
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(INITIAL_SETTINGS);
  const [classes, setClasses] = useState(DEFAULT_CLASSES);
  const [models, setModels] = useState<WeaponModelInfo[]>([]);
  const [health, setHealth] = useState<'connecting' | 'online' | 'offline'>('connecting');
  const [device, setDevice] = useState('');
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [mode, setMode] = useState<'image' | 'camera' | 'video'>('image');
  const [file, setFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoProcessing, setVideoProcessing] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ResultView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraPending, setCameraPending] = useState(false);
  const [fps, setFps] = useState(0);
  const [helpOpen, setHelpOpen] = useState(false);
  const [capturesOpen, setCapturesOpen] = useState(false);
  const [latestCapture, setLatestCapture] = useState<Capture | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [captureSettingsOpen, setCaptureSettingsOpen] = useState(false);
  const [captureNotice, setCaptureNotice] = useState<Capture | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generation = useRef(0);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  function receiveCapture(data: Pick<DetectionResult, 'capture' | 'capture_error'>) {
    if (data.capture) {
      setLatestCapture(data.capture);
      if (settingsRef.current.capture.notifications) setCaptureNotice(data.capture);
    }
    if (data.capture_error) setCaptureError(data.capture_error);
    else if (data.capture) setCaptureError(null);
  }

  const checkConnection = useCallback(async (signal?: AbortSignal) => {
    setHealth('connecting');
    setConnectionError(null);
    try {
      const requestSignal = signal
        ? AbortSignal.any([signal, AbortSignal.timeout(10_000)])
        : AbortSignal.timeout(10_000);
      const [healthData, classData, modelData] = await Promise.all([
        fetch('/api/health', { signal: requestSignal }).then(
          (response) => readResponse<{ device: string }>(response)),
        fetch('/api/config', { signal: requestSignal }).then(
          (response) => readResponse<ClassInfo[]>(response)),
        fetch('/api/models', { signal: requestSignal }).then(
          (response) => readResponse<WeaponModelInfo[]>(response)),
      ]);
      if (signal?.aborted) return;
      setDevice(healthData.device);
      setClasses(classData);
      setModels(modelData);
      setHealth('online');
    } catch (failure) {
      if (signal?.aborted) return;
      setHealth('offline');
      setConnectionError(errorMessage(failure));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void checkConnection(controller.signal);
    return () => controller.abort();
  }, [checkConnection]);

  const stopCamera = useCallback(() => {
    generation.current += 1;
    requestRef.current?.abort();
    if (timerRef.current) clearTimeout(timerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setCameraPending(false);
    setBusy(false);
    setFps(0);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const acceptFile = useCallback((next: File) => {
    const validation = validateImage(next);
    if (validation) {
      setError(validation);
      return;
    }
    stopCamera();
    setMode('image');
    setFile(next);
    setResult(null);
    setError(null);
  }, [stopCamera]);

  const acceptVideo = useCallback((next: File) => {
    if (videoProcessing) { setError('Cancel the current video job before replacing the file.'); return; }
    const validation = validateVideo(next);
    if (validation) { setError(validation); return; }
    stopCamera();
    setMode('video');
    setVideoFile(next);
    setResult(null);
    setError(null);
  }, [stopCamera, videoProcessing]);

  useEffect(() => {
    const paste = (event: ClipboardEvent) => {
      const images = Array.from(event.clipboardData?.items ?? [])
        .filter((item) => item.kind === 'file' && item.type.startsWith('image/'));
      const image = images[0]?.getAsFile();
      if (image) {
        event.preventDefault();
        acceptFile(image);
      }
    };
    window.addEventListener('paste', paste);
    return () => window.removeEventListener('paste', paste);
  }, [acceptFile]);

  async function runDetection() {
    if (!file || !settings.classes.length) return;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const token = ++generation.current;
    const snapshot = settingsRef.current;
    setBusy(true);
    setError(null);
    try {
      const data = await detectImage(file, snapshot, controller.signal);
      if (token === generation.current) {
        receiveCapture(data);
        setResult({ data, settingsKey: JSON.stringify(snapshot) });
      }
    } catch (failure) {
      if (!controller.signal.aborted && token === generation.current) setError(errorMessage(failure));
    } finally {
      if (token === generation.current) setBusy(false);
    }
  }

  async function startCamera() {
    stopCamera();
    setMode('camera');
    setResult(null);
    setError(null);
    setCameraPending(true);
    const token = generation.current;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera access needs localhost or HTTPS and a supported browser.');
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false,
      });
      if (token !== generation.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Camera preview could not initialize. Please try again.');
      video.srcObject = stream;
      await video.play();
      if (token !== generation.current) return;
      setCameraActive(true);
      setCameraPending(false);
      stream.getVideoTracks().forEach((track) => track.addEventListener('ended', () => {
        if (token !== generation.current) return;
        stopCamera();
        setError('The camera disconnected. Reconnect it and open the camera again.');
      }, { once: true }));
      const canvas = document.createElement('canvas');

      async function scan() {
        if (token !== generation.current) return;
        if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight || !settingsRef.current.classes.length) {
          timerRef.current = setTimeout(() => void scan(), 200);
          return;
        }
        try {
          const started = performance.now();
          const blob = await captureVideoFrame(video, canvas);
          if (token !== generation.current) return;
          const snapshot = settingsRef.current;
          const controller = new AbortController();
          requestRef.current = controller;
          setBusy(true);
          const data = await detectImage(blob, snapshot, controller.signal, 'camera');
          if (token !== generation.current) return;
          receiveCapture(data);
          setResult({ data, settingsKey: JSON.stringify(snapshot) });
          setFps(1000 / Math.max(1, performance.now() - started));
          setBusy(false);
          // Wait for each response before capturing another frame; never queue video frames.
          timerRef.current = setTimeout(() => void scan(), 0);
        } catch (failure) {
          if (token !== generation.current) return;
          stopCamera();
          setError(errorMessage(failure));
        }
      }
      void scan();
    } catch (failure) {
      if (token !== generation.current) return;
      stopCamera();
      if (failure instanceof DOMException && failure.name === 'NotAllowedError') {
        setError('Camera permission was denied. Allow camera access in your browser and try again.');
      } else if (failure instanceof DOMException && failure.name === 'NotFoundError') {
        setError('No camera was found. Connect a webcam and try again.');
      } else if (failure instanceof DOMException && failure.name === 'NotReadableError') {
        setError('Camera is unavailable or in use by another application. Close it there and retry.');
      } else {
        setError(errorMessage(failure));
      }
    }
  }

  function changeMode(next: 'image' | 'camera' | 'video') {
    if (next === mode) return;
    stopCamera();
    setMode(next);
    setResult(null);
    setError(null);
  }

  function toggleClass(id: string) {
    setSettings((current) => ({
      ...current,
      classes: current.classes.includes(id)
        ? current.classes.filter((item) => item !== id) : [...current.classes, id],
    }));
  }

  const resultData = result?.data;
  const stale = Boolean(result && result.settingsKey !== JSON.stringify(settings));
  const cameraRunning = cameraActive || cameraPending;
  const selectedObjects = settings.classes.some((name) => name !== 'person');
  const busyImage = busy && mode === 'image';
  const selectedModel = models.find((model) => model.id === settings.modelId);

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="/" aria-label="Visible Weapon Detection for Hong Kong Surveillance home">
        <span className="brand-mark"><CenterFocusStrongRounded /></span>
        <span className="brand-name">Visible Weapon Detection for Hong Kong Surveillance</span>
      </a>
      <div className="topbar-right">
        <span className="local-label"><LockOutlined sx={{ fontSize: 14 }} /> Local inference</span>
        <Chip size="small" className={`connection-chip ${health}`}
          icon={<span className="status-dot" />}
          label={health === 'online' ? 'API connected' : health === 'connecting' ? 'Connecting' : 'API offline'} />
      </div>
    </header>

    <main>
      <section className="page-heading">
        <div>
          <Typography variant="h4" component="h1">CCTV-based Firearm Detection</Typography>
          <Typography color="text.secondary" className="page-description">
            Upload an image or CCTV video, or connect a live camera.
          </Typography>
        </div>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        <Button variant="outlined" startIcon={<PhotoCameraOutlined />} onClick={() => setCapturesOpen(true)}>
          Check captures
        </Button>
        <Button variant="outlined" color="secondary" startIcon={<InfoOutlined />}
          onClick={() => setHelpOpen((open) => !open)} aria-expanded={helpOpen}>
          How it works
        </Button>
        </Stack>
      </section>

      <Collapse in={helpOpen}>
        <Alert severity="info" sx={{ mb: 2 }}>
          Choose Weapon detection model v1 (last.pt) or v0 (best.pt) for the six SoHAS object classes. v1 is selected by default.
          Both versions use the same separate COCO YOLO11s model for people.
          When Rifle is selected, the separate rifle_detector.pt model also runs on the same frame.
          Select classes, adjust each model&apos;s threshold, then run image/video detection or open your camera.
          Uploaded videos process every frame into a separate downloadable MP4, keeping original audio.
          Live camera requests run sequentially; speed depends on your hardware and the selected models.
          With auto-capture enabled, matching frames are saved to the captures/ folder.
        </Alert>
      </Collapse>
      {connectionError && <Alert severity="error" sx={{ mb: 2 }}
        action={<Button color="inherit" size="small" onClick={() => void checkConnection()}>Reconnect</Button>}>
        {connectionError}
      </Alert>}
      {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>{error}</Alert>}
      {captureError && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setCaptureError(null)}>{captureError}</Alert>}
      {mode === 'camera' && videoProcessing && <Alert severity="info" sx={{ mb: 2 }}>
        A video is processing in the background. It shares the detection models and may slow live camera analysis.
      </Alert>}
      {latestCapture && <Alert severity={latestCapture.is_warning ? 'warning' : 'success'} sx={{ mb: 2 }}
        action={<Button color="inherit" size="small" onClick={() => setCapturesOpen(true)}>View captures</Button>}>
        Capture saved at {new Date(latestCapture.created_at).toLocaleTimeString()} — object {(latestCapture.object_confidence * 100).toFixed(1)}%
        {' '}and person {(latestCapture.person_confidence * 100).toFixed(1)}%.
        {' '}{latestCapture.model_name}{latestCapture.rifle_model ? ' + Rifle detection model' : ''}.
      </Alert>}

      <div className="workspace-grid">
        <Paper component="aside" className="settings-panel">
          <div className="panel-title"><TuneRounded sx={{ fontSize: 19 }} /><Typography variant="h6">Detection settings</Typography></div>
          <Divider />
          <div className="settings-section">
            <Typography className="section-label" sx={{ mb: 1.5 }}>MODEL VERSION</Typography>
            <TextField select fullWidth size="small" label="Weapon detection model"
              value={selectedModel ? settings.modelId : ''} disabled={health !== 'online'}
              onChange={(event) => {
                const model = models.find((item) => item.id === event.target.value);
                if (!model) { setError('Choose a loaded weapon detection model.'); return; }
                setSettings((current) => ({ ...current, modelId: model.id }));
              }}
              sx={{ '& .MuiSelect-select': { fontSize: 11, py: 1.3 } }}>
              {models.map((model) => <MenuItem key={model.id} value={model.id} sx={{ fontSize: 12 }}>{model.name}</MenuItem>)}
            </TextField>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1, lineHeight: 1.8, fontSize: 10 }}>
              {selectedModel ? `${selectedModel.filename} · changes apply to the next camera frame or detection run.` : 'Loading available models…'}
              {' '}Person detection uses the same COCO model.
              {' '}Rifle detection runs alongside when Rifle is selected, using object_conf.
            </Typography>
          </div>
          <Divider />
          <div className="settings-section">
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography className="section-label">DETECT CLASSES</Typography>
              <span className="count-badge">{settings.classes.length} / {classes.length}</span>
            </Stack>
            <Typography className="settings-caption">Choose what to look for in your scene.</Typography>
            <Stack direction="row" spacing={0.5} sx={{ mt: 1, mb: 1.2 }}>
              <Button size="small" onClick={() => setSettings((current) => ({ ...current, classes: classes.map((item) => item.id) }))}>Select all</Button>
              <Button size="small" color="secondary" onClick={() => setSettings((current) => ({ ...current, classes: [] }))}>Clear</Button>
            </Stack>
            <div className="class-list">
              {classes.map((item) => <div key={item.id} className={`class-row ${settings.classes.includes(item.id) ? 'selected' : ''}`}>
                <FormControlLabel
                  control={<Checkbox size="small" checked={settings.classes.includes(item.id)}
                    onChange={() => toggleClass(item.id)} sx={{ color: '#4b5b6d', p: 0.7 }} />}
                  label={<span className="class-label"><ClassIcon label={item.id} color={item.color} />{item.label}</span>}
                />
                {item.category === 'weapon' && <span className="weapon-tag">WEAPON</span>}
              </div>)}
            </div>
            {!settings.classes.length && <Alert severity="warning" sx={{ mt: 1.5, py: 0 }}>Select at least one class.</Alert>}
          </div>
          <Divider />
          <div className="settings-section thresholds">
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
              <Typography className="section-label">CONFIDENCE THRESHOLDS</Typography>
              <Tooltip title="A higher threshold filters out less-confident predictions. Lower values can increase false positives.">
                <InfoOutlined sx={{ fontSize: 16, color: 'text.secondary' }} />
              </Tooltip>
            </Stack>
            <ConfidenceControl label="Object confidence" code="object_conf" value={settings.objectConf}
              disabled={!selectedObjects}
              onChange={(objectConf) => setSettings((current) => ({ ...current, objectConf }))} />
            <ConfidenceControl label="Person confidence" code="person_conf" value={settings.personConf}
              disabled={!settings.classes.includes('person')}
              onChange={(personConf) => setSettings((current) => ({ ...current, personConf }))} />
            <div className="threshold-note"><InfoOutlined sx={{ fontSize: 15 }} /><span>Thresholds apply independently to each model.</span></div>
          </div>
          <Divider />
          <div className="settings-section">
            <FormControlLabel
              control={<Checkbox checked={settings.autoCapture}
                onChange={(_, autoCapture) => setSettings((current) => ({ ...current, autoCapture }))} />}
              label={<Typography variant="body2" fontSize={12}>Auto-capture screenshots</Typography>}
            />
            <Typography variant="caption" color="text.secondary" display="block" sx={{ lineHeight: 1.8 }}>
              Save when a capture object scores ≥{Math.round(settings.capture.objectConf * 100)}% and person ≥{Math.round(settings.capture.personConf * 100)}%.
              {' '}Cooldown: {settings.capture.cooldownSeconds}s.
            </Typography>
            {settings.autoCapture && (!settings.classes.includes('person')
              || !settings.capture.classes.some((name) => settings.classes.includes(name)))
              && <Alert severity="warning" sx={{ mt: 1, py: 0 }}>Select Person and a configured capture class to capture.</Alert>}
            <Button size="small" startIcon={<SettingsOutlined />} onClick={() => setCaptureSettingsOpen(true)} sx={{ mt: 1 }}>
              Capture settings
            </Button>
            <Button size="small" startIcon={<PhotoCameraOutlined />} onClick={() => setCapturesOpen(true)} sx={{ mt: 1 }}>
              Check captures
            </Button>
          </div>
          <Divider />
          <Button className="reset-button" color="secondary" startIcon={<RefreshRounded sx={{ fontSize: 16 }} />}
            onClick={() => setSettings({ ...INITIAL_SETTINGS, classes: [...INITIAL_SETTINGS.classes] })}>
            Reset to defaults
          </Button>
        </Paper>

        <div className="studio-main">
          <Paper className="stage-panel">
            <div className="stage-toolbar">
              <Tabs value={mode} onChange={(_, next: 'image' | 'camera' | 'video') => changeMode(next)} aria-label="Input mode"
                variant="scrollable" scrollButtons="auto">
                <Tab value="image" icon={<ImageOutlined sx={{ fontSize: 18 }} />} iconPosition="start" label="Image" />
                <Tab value="video" icon={<PlayArrowRounded sx={{ fontSize: 19 }} />} iconPosition="start" label="Video" />
                <Tab value="camera" icon={<VideocamOutlined sx={{ fontSize: 19 }} />} iconPosition="start" label="Live camera" />
              </Tabs>
              <Tooltip title={selectedModel?.name ?? 'Choose a model in detection settings'}>
                <span className="model-label"><MemoryRounded sx={{ fontSize: 16 }} /> YOLO11s · {settings.modelId}</span>
              </Tooltip>
            </div>
            <Divider />
            <div hidden={mode !== 'video'}>
              <UploadedVideo file={videoFile} settings={settings} enabled={health === 'online'} visible={mode === 'video'}
                onUpload={acceptVideo} onRemove={() => { setVideoFile(null); setError(null); }}
                onCapture={receiveCapture} onRunning={setVideoProcessing} />
            </div>
            <div hidden={mode === 'video'}>
            <div className="preview-grid">
              <section className="preview-column">
                <div className="preview-heading">
                  <span><span className="step-number">01</span> {mode === 'image' ? 'Source image' : 'Camera feed'}</span>
                  <span className="preview-meta">{mode === 'image' ? 'INPUT' : cameraActive ? 'LIVE' : 'INPUT'}</span>
                </div>
                <div
                  className={`source-stage media-stage ${dragging ? 'dragging' : ''} ${preview && mode === 'image' ? 'has-image' : ''}`}
                  onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                  onDragLeave={(event) => {
                    if (!(event.relatedTarget instanceof Node)
                      || !event.currentTarget.contains(event.relatedTarget)) setDragging(false);
                  }}
                  onDrop={(event) => {
                    event.preventDefault(); setDragging(false);
                    const dropped = event.dataTransfer.files[0];
                    if (dropped) {
                      if (dropped.type.startsWith('video/')) acceptVideo(dropped);
                      else acceptFile(dropped);
                    } else setError('Drop a local image or video file, not a link.');
                  }}
                >
                  <video ref={videoRef} autoPlay muted playsInline aria-label="Live camera feed"
                    style={{ display: mode === 'camera' && cameraRunning ? 'block' : 'none' }} />
                  {mode === 'image' && (preview
                    ? <>
                      <img src={preview} alt="Original uploaded image" onError={() => setError('This image cannot be displayed. Choose a valid JPEG, PNG, or WebP.')} />
                      <Tooltip title="Remove image"><IconButton className="remove-image" size="small" aria-label="Remove image"
                        onClick={() => { stopCamera(); setFile(null); setResult(null); setError(null); }}>
                        <CloseRounded fontSize="small" />
                      </IconButton></Tooltip>
                    </>
                    : <div className="drop-content" role="button" tabIndex={0} aria-label="Upload an image"
                      onClick={() => fileInput.current?.click()}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.current?.click(); }
                      }}>
                      <div className="upload-illustration">
                        <span className="illustration-back" />
                        <span className="illustration-front"><CloudUploadOutlined sx={{ fontSize: 34 }} /></span>
                        <span className="illustration-plus">+</span>
                      </div>
                      <Typography variant="h6">Drop your image here</Typography>
                      <Typography variant="body2" color="text.secondary">or click to browse your files</Typography>
                      <span className="file-format">JPG, PNG, WEBP <span>·</span> up to 10 MB</span>
                      <span className="paste-hint"><ContentPasteRounded sx={{ fontSize: 13 }} /> You can also paste an image</span>
                    </div>)}
                  {mode === 'camera' && !cameraRunning && <div className="drop-content camera-empty">
                    <div className="empty-icon"><VideocamOutlined sx={{ fontSize: 34 }} /></div>
                    <Typography variant="h6">Your camera, in focus</Typography>
                    <Typography variant="body2" color="text.secondary">Open your camera to detect live.</Typography>
                    <span className="file-format">Camera permission is required</span>
                  </div>}
                  {mode === 'camera' && cameraPending && <div className="camera-pending"><CircularProgress size={26} /><span>Waiting for camera permission…</span></div>}
                  {mode === 'camera' && cameraActive && <span className="live-badge"><span /> LIVE</span>}
                </div>
                <div className="preview-footnote">
                  {mode === 'image' && file
                    ? <><ImageOutlined sx={{ fontSize: 14 }} /><span className="filename">{file.name}</span><span>{(file.size / 1024).toFixed(0)} KB</span></>
                    : <><LockOutlined sx={{ fontSize: 14 }} /><span>{mode === 'camera' ? 'Video only. Audio is never requested.' : 'Your original image stays unchanged.'}</span></>}
                </div>
              </section>

              <div className="preview-arrow"><ArrowForwardRounded sx={{ fontSize: 17 }} /></div>
              <section className="preview-column">
                <div className="preview-heading">
                  <span><span className="step-number">02</span> Detection result</span>
                  <span className={`preview-meta ${result ? 'accent-text' : ''}`}>{result ? stale ? 'SETTINGS CHANGED' : 'PROCESSED' : 'OUTPUT'}</span>
                </div>
                <div className="result-stage media-stage" aria-busy={busy} aria-label="Detection result">
                  {resultData
                    ? <img src={resultData.annotated_image} alt="Annotated detection result" />
                    : <div className="result-placeholder">
                      <div className={`reticle ${busy ? 'scanning' : ''}`}><CenterFocusStrongRounded sx={{ fontSize: 38 }} /></div>
                      <Typography variant="body2" fontWeight={550}>{busy ? 'Analyzing your scene' : 'Ready when you are'}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {busy ? 'Running your detection models…' : 'Your detections will appear here'}
                      </Typography>
                    </div>}
                  {busy && <LinearProgress className="stage-progress" />}
                </div>
                <div className="preview-footnote">
                  <CenterFocusStrongRounded sx={{ fontSize: 14 }} />
                  <span>{resultData ? `${resultData.width} × ${resultData.height} px · ${mode === 'camera' && !cameraRunning ? 'Last processed frame' : stale ? 'Run again to apply new settings' : 'Bounding boxes + confidence scores'}` : 'Bounding boxes + confidence scores'}</span>
                </div>
              </section>
            </div>

            <div className="stage-actions">
              <span className="action-hint">
                <span className={`small-dot ${cameraActive ? 'active' : ''}`} />
                {cameraPending ? 'Waiting for permission' : cameraActive
                  ? settings.classes.length ? `Live analysis · ${fps.toFixed(1)} processed fps` : 'Paused · select a class'
                  : busyImage ? 'Processing image…' : file && mode === 'image' ? 'Image ready for analysis' : 'Awaiting input'}
              </span>
              <Stack direction="row" spacing={1}>
                {mode === 'image' && preview && <Button color="secondary" onClick={() => fileInput.current?.click()}>Replace</Button>}
                {mode === 'image'
                  ? <Button variant="contained" startIcon={busyImage ? <CircularProgress size={16} color="inherit" /> : <PlayArrowRounded />}
                    disabled={!file || busy || !settings.classes.length || health !== 'online'} onClick={() => void runDetection()}>
                    {busyImage ? 'Detecting…' : 'Run detection'}
                  </Button>
                  : mode === 'camera' && <Button variant={cameraRunning ? 'outlined' : 'contained'}
                    color={cameraRunning ? 'error' : 'primary'}
                    startIcon={cameraRunning ? <StopRounded /> : <PhotoCameraOutlined />}
                    disabled={!cameraRunning && (!settings.classes.length || health !== 'online')}
                    onClick={() => cameraRunning ? stopCamera() : void startCamera()}>
                    {cameraRunning ? 'Stop camera' : 'Open camera'}
                  </Button>}
              </Stack>
            </div>
            <input ref={fileInput} hidden type="file" accept="image/jpeg,image/png,image/webp" aria-label="Image file"
              onChange={(event) => {
                const next = event.target.files?.[0];
                if (next) acceptFile(next);
                event.target.value = '';
              }} />
            </div>
          </Paper>

          <div hidden={mode === 'video'}>
          <div className="metrics-grid">
            <Paper className="metric-card"><span className="metric-icon green"><CenterFocusStrongRounded /></span>
              <div><span className="metric-label">Objects detected</span><strong>{resultData?.detections.length ?? '—'}</strong></div>
              {resultData && <span className="metric-detail">{Object.values(resultData.counts).filter((count) => count > 0).length} classes</span>}
            </Paper>
            <Paper className="metric-card"><span className="metric-icon blue"><VisibilityOutlined /></span>
              <div><span className="metric-label">Inference time</span><strong>{resultData ? Math.round(resultData.inference_ms) : '—'}{resultData && <small>ms</small>}</strong></div>
              <span className="metric-detail">on {device || 'device'}</span>
            </Paper>
            <Paper className="metric-card"><span className={`metric-icon ${resultData?.weapon_count ? 'red' : 'neutral'}`}><ShieldOutlined /></span>
              <div><span className="metric-label">Weapon predictions</span><strong>{resultData?.weapon_count ?? '—'}</strong></div>
              <span className="metric-detail">pistol + knife + rifle</span>
            </Paper>
          </div>

          <Paper className="detections-panel">
            <div className="detections-header">
              <Stack spacing={0.5} sx={{ minWidth: 0 }}>
                <Stack direction="row" spacing={1.2} alignItems="center">
                  <Typography variant="h6">Detected objects</Typography>
                  <span className="count-badge">{resultData?.detections.length ?? 0}</span>
                  {stale && <Chip size="small" label="Previous settings" color="warning" variant="outlined" />}
                </Stack>
                {resultData && <Typography variant="caption" color="text.secondary" sx={{ fontSize: 9 }}>
                  Processed with {resultData.model_name}{resultData.rifle_model ? ' + Rifle detection model' : ''}
                </Typography>}
              </Stack>
              <Button size="small" color="secondary" startIcon={<DownloadRounded sx={{ fontSize: 16 }} />}
                disabled={!resultData} component="a" href={resultData?.annotated_image} download="sentinel-detection.jpg">
                Save image
              </Button>
            </div>
            <Divider />
            {resultData?.detections.length
              ? <div className="detection-table-wrap"><table className="detection-table">
                <thead><tr><th>OBJECT</th><th>CONFIDENCE</th><th>MODEL</th><th>BOUNDING BOX</th></tr></thead>
                <tbody>{resultData.detections.map((detection, index) => <tr key={`${detection.label}-${index}`}>
                  <td><span className="detection-name"><ClassIcon label={detection.label}
                    color={classes.find((item) => item.id === detection.label)?.color} size={16} />{detection.label}</span></td>
                  <td><div className="confidence-cell"><span className="confidence-track"><span style={{ width: `${detection.confidence * 100}%` }} /></span>{(detection.confidence * 100).toFixed(1)}%</div></td>
                  <td><span className="source-tag" title={detection.source === 'person' ? 'COCO person detector'
                    : detection.label === 'rifle' ? `Rifle detection model · ${resultData.rifle_model}` : resultData.model_name}>
                    {detection.source === 'person' ? 'COCO · Person' : detection.label === 'rifle' ? 'Rifle detector' : `SoHAS · ${resultData.model_id}`}
                  </span></td>
                  <td className="box-coordinates">{detection.box.map(Math.round).join(', ')}</td>
                </tr>)}</tbody>
              </table></div>
              : <div className="no-detections"><span className="no-detections-icon">{resultData ? <CheckRounded /> : <SettingsOutlined />}</span>
                <span>{resultData ? 'No selected objects detected at these thresholds.' : 'No detections yet. Add an image or start your camera to begin.'}</span>
              </div>}
          </Paper>
          </div>
          <div className="responsibility-note"><InfoOutlined sx={{ fontSize: 15 }} />
            <span>Model predictions can be wrong. No detection does not mean a scene is safe. Always verify results; person boxes do not establish who holds an object.</span>
          </div>
        </div>
      </div>
      {capturesOpen && <Suspense fallback={<Alert severity="info" sx={{ mt: 2 }}>Loading capture gallery…</Alert>}>
        <CapturesGallery open={capturesOpen} onClose={() => setCapturesOpen(false)} latestCaptureId={latestCapture?.id}
          settings={settings.capture} onOpenSettings={() => setCaptureSettingsOpen(true)}
          onDeleted={(id) => {
            setLatestCapture((current) => current?.id === id ? null : current);
            setCaptureNotice((current) => current?.id === id ? null : current);
          }} />
      </Suspense>}
      {captureSettingsOpen && <Suspense fallback={<Alert severity="info" sx={{ mt: 2 }}>Loading capture settings…</Alert>}>
        <CaptureSettingsDialog open={captureSettingsOpen} onClose={() => setCaptureSettingsOpen(false)}
          settings={settings.capture} classes={classes}
          onChange={(capture) => {
            setSettings((current) => ({ ...current, capture }));
            if (!capture.notifications) setCaptureNotice(null);
          }} />
      </Suspense>}
      <CaptureNotification capture={captureNotice} persist={videoProcessing && captureNotice?.source === 'video'}
        onClose={() => setCaptureNotice(null)}
        onView={() => setCapturesOpen(true)} />
      <footer><span><span className="footer-dot" /> Visible Weapon Detection for Hong Kong Surveillance</span>
        <span>Your models. One workspace. <span className="footer-divider">|</span> React + FastAPI</span></footer>
    </main>
  </div>;
}

import { useEffect, useRef, useState } from 'react';
import {
  Alert, Button, CircularProgress, LinearProgress, MenuItem, Stack, TextField, Typography,
} from '@mui/material';
import {
  ArrowForwardRounded, CloudUploadOutlined, DownloadRounded, PlayArrowRounded, StopRounded,
} from '@mui/icons-material';
import { cancelVideoJob, errorMessage, readResponse, readVideoJob, submitVideo } from './api';
import type { DetectionResult, Settings, VideoJob } from './types';

function isActive(job: VideoJob | null): boolean {
  return job?.status === 'queued' || job?.status === 'running';
}

export default function UploadedVideo({ file, settings, enabled, visible, onUpload, onRemove, onCapture, onRunning }: {
  file: File | null; settings: Settings; enabled: boolean; visible: boolean;
  onUpload: (file: File) => void; onRemove: () => void;
  onCapture: (data: Pick<DetectionResult, 'capture' | 'capture_error'>) => void;
  onRunning: (running: boolean) => void;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [job, setJob] = useState<VideoJob | null>(null);
  const [history, setHistory] = useState<VideoJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [settingsKey, setSettingsKey] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const sourceVideo = useRef<HTMLVideoElement>(null);
  const outputVideo = useRef<HTMLVideoElement>(null);
  const request = useRef<AbortController | null>(null);
  const statusRequest = useRef<AbortController | null>(null);
  const previousFile = useRef(file);
  const fileRef = useRef(file);
  const jobRef = useRef(job);
  const historyLoaded = useRef(false);
  const latestCapture = useRef<string | null>(null);
  const callbacks = useRef({ onCapture, onRunning });
  callbacks.current = { onCapture, onRunning };
  fileRef.current = file;
  jobRef.current = job;
  const working = uploading || isActive(job);

  function updateJob(next: VideoJob) {
    setJob(next);
    setHistory((items) => [next, ...items.filter((item) => item.id !== next.id)].slice(0, 20));
  }

  useEffect(() => {
    if (previousFile.current !== file) {
      setJob(null);
      setSettingsKey(null);
      setError(null);
      setPollError(null);
      setPreviewError(null);
      previousFile.current = file;
    }
    if (!file) { setUrl(null); return; }
    const objectUrl = URL.createObjectURL(file);
    const video = sourceVideo.current;
    setUrl(objectUrl);
    return () => { video?.pause(); URL.revokeObjectURL(objectUrl); };
  }, [file]);

  useEffect(() => {
    if (!visible) { sourceVideo.current?.pause(); outputVideo.current?.pause(); }
  }, [visible]);

  useEffect(() => {
    callbacks.current.onRunning(working);
  }, [working]);
  useEffect(() => () => {
    request.current?.abort();
    callbacks.current.onRunning(false);
  }, []);

  useEffect(() => {
    if (!visible || !enabled || historyLoaded.current) return;
    const controller = new AbortController();
    let finished = false;
    historyLoaded.current = true;
    void fetch('/api/videos', { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]) })
      .then((response) => readResponse<VideoJob[]>(response))
      .then((jobs) => {
        if (controller.signal.aborted) return;
        if (!Array.isArray(jobs)) throw new Error('The API returned invalid video job history.');
        setHistory(jobs);
        if (!fileRef.current && !jobRef.current && jobs.length) {
          const recovered = jobs.find(isActive) ?? jobs[0];
          latestCapture.current = recovered.capture?.id ?? null;
          setJob(recovered);
        }
        finished = true;
      })
      .catch((failure) => {
        if (controller.signal.aborted) return;
        setPollError(errorMessage(failure));
        historyLoaded.current = false;
      });
    return () => { controller.abort(); if (!finished) historyLoaded.current = false; };
  }, [visible, enabled, retry]);

  useEffect(() => {
    if (!job || !isActive(job)) return;
    const id = job.id;
    const controller = new AbortController();
    statusRequest.current = controller;
    let timer: ReturnType<typeof setTimeout> | null = null;
    async function poll() {
      try {
        const next = await readVideoJob(id, controller.signal);
        if (controller.signal.aborted) return;
        updateJob(next);
        setPollError(null);
        if (isActive(next)) timer = setTimeout(() => void poll(), 750);
      } catch (failure) {
        if (!controller.signal.aborted) setPollError(errorMessage(failure));
      }
    }
    timer = setTimeout(() => void poll(), 250);
    return () => { controller.abort(); if (timer) clearTimeout(timer); };
  }, [job?.id, job?.status, retry]);

  useEffect(() => {
    if (!job) return;
    if (job.capture?.id !== latestCapture.current && job.capture) {
      latestCapture.current = job.capture.id;
      callbacks.current.onCapture({ capture: job.capture, capture_error: job.capture_error });
    } else if (job.capture_error) {
      callbacks.current.onCapture({ capture: null, capture_error: job.capture_error });
    }
  }, [job?.capture?.id, job?.capture_error]);

  async function run() {
    if (!file || working || !enabled || !settings.classes.length) return;
    const controller = new AbortController();
    request.current = controller;
    setUploading(true);
    setError(null);
    setPollError(null);
    setJob(null);
    latestCapture.current = null;
    setSettingsKey(JSON.stringify(settings));
    try {
      const next = await submitVideo(file, settings, controller.signal);
      if (!controller.signal.aborted) updateJob(next);
    } catch (failure) {
      if (!controller.signal.aborted) setError(errorMessage(failure));
    } finally {
      if (!controller.signal.aborted) setUploading(false);
    }
  }

  async function cancel() {
    if (!job || !isActive(job)) return;
    setCancelling(true);
    statusRequest.current?.abort();
    try { updateJob(await cancelVideoJob(job.id)); }
    catch (failure) { setError(errorMessage(failure)); }
    finally { setCancelling(false); setRetry((value) => value + 1); }
  }

  const changed = Boolean(job && settingsKey && settingsKey !== JSON.stringify(settings));
  return <>
    <div className="preview-grid">
      <section className="preview-column">
        <div className="preview-heading"><span><span className="step-number">01</span> Source video</span><span className="preview-meta">INPUT</span></div>
        <div className="source-stage media-stage" onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            if (working) { setError('Cancel the current video job before replacing the file.'); return; }
            const next = event.dataTransfer.files[0];
            if (next) onUpload(next);
            else setError('Drop a local video file, not a link.');
          }}>
          {file ? <video ref={sourceVideo} src={url ?? undefined} controls playsInline aria-label="Uploaded CCTV video"
            onError={() => setPreviewError('Your browser cannot preview this source codec. You can still run server-side detection.')} />
            : <div className="drop-content" role="button" tabIndex={0} aria-label="Upload a video"
              onClick={() => input.current?.click()} onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.current?.click(); }
              }}>
              <div className="empty-icon"><CloudUploadOutlined sx={{ fontSize: 34 }} /></div>
              <Typography variant="h6">Drop your CCTV video here</Typography>
              <Typography variant="body2" color="text.secondary">or click to browse your files</Typography>
              <span className="file-format">MP4, WebM, MOV · up to 100 MB</span>
              <span className="paste-hint">Run detection to create a separate annotated video</span>
            </div>}
        </div>
        <div className="preview-footnote"><span className="filename">{file?.name ?? 'Original video stays unchanged'}</span>
          {file && <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span>}</div>
      </section>
      <div className="preview-arrow"><ArrowForwardRounded sx={{ fontSize: 17 }} /></div>
      <section className="preview-column">
        <div className="preview-heading"><span><span className="step-number">02</span> Annotated video</span>
          <span className="preview-meta">{job?.status === 'completed' ? 'COMPLETE' : 'OUTPUT'}</span></div>
        <div className="result-stage media-stage" aria-label="Annotated video output" aria-busy={working}>
          {job?.status === 'completed' && job.video_url
            ? <video key={job.id} ref={outputVideo} src={job.video_url} controls playsInline aria-label="Annotated detection video"
              onError={() => setError('The processed video could not play in this browser. Download it to open locally.')} />
            : <div className="result-placeholder">
              {working ? <CircularProgress size={32} /> : <PlayArrowRounded sx={{ fontSize: 38 }} />}
              <Typography variant="body2">{uploading ? 'Uploading video…' : isActive(job) ? 'Processing every frame…' : 'Run detection to generate your video'}</Typography>
              <Typography variant="caption" color="text.secondary">A separate MP4 with bounding boxes and original audio</Typography>
            </div>}
        </div>
        <div className="preview-footnote"><span className="filename">{job?.filename ?? 'Output keeps source timing and audio'}</span></div>
      </section>
    </div>
    <div className="video-progress">
      {previewError && <Alert severity="info" sx={{ mb: 1 }}>{previewError}</Alert>}
      {(error || job?.error) && <Alert severity="error" sx={{ mb: 1 }}>{error ?? job?.error}</Alert>}
      {pollError && <Alert severity="error" sx={{ mb: 1 }}
        action={<Button size="small" color="inherit" onClick={() => setRetry((value) => value + 1)}>Retry status</Button>}>
        {pollError} Processing may still be running on the backend.
      </Alert>}
      {job && <Stack spacing={1}>
        {job.model_name && <Typography variant="caption" color="text.secondary">
          Processed with {job.model_name}{job.rifle_model ? ' + Rifle detection model' : ''}
        </Typography>}
        <Typography variant="body2">
          {job.status === 'completed' ? 'Video processing complete' : job.status === 'cancelled' ? 'Video processing cancelled'
            : job.status === 'failed' ? 'Video processing failed' : 'Processing video'}
          {' · '}{job.frames_processed}{job.total_frames ? ` / ${job.total_frames}` : ''} frames
          {job.fps ? ` · ${job.fps.toFixed(2)} fps (source average)` : ''}
          {job.capture_count ? ` · ${job.capture_count} captures saved` : ''}
        </Typography>
        <LinearProgress aria-label="Video processing progress" variant={job.progress === null ? 'indeterminate' : 'determinate'}
          value={job.progress ?? undefined} />
      </Stack>}
      {changed && <Alert severity="info" sx={{ mt: 1 }}>This job keeps its submitted settings. Run detection again to apply changed settings.</Alert>}
      {history.length > 0 && <TextField select fullWidth size="small" label="Video jobs" value={job?.id ?? ''} disabled={working}
        onChange={(event) => {
          const selected = history.find((item) => item.id === event.target.value);
          if (selected) {
            latestCapture.current = selected.capture?.id ?? null;
            setJob(selected); setSettingsKey(null); setError(null);
          }
        }} sx={{ mt: 2 }}>
        {history.map((item) => <MenuItem key={item.id} value={item.id}>{item.filename} · {item.status}</MenuItem>)}
      </TextField>}
    </div>
    <div className="stage-actions">
      <span className="action-hint">Every frame is analyzed. Long clips may take longer than their playback duration.</span>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {file && <Button color="secondary" disabled={working} onClick={onRemove}>Remove video</Button>}
        {file && <Button color="secondary" disabled={working} onClick={() => input.current?.click()}>Replace</Button>}
        {job?.status === 'completed' && job.video_url && <Button component="a" href={`${job.video_url}?download=true`}
          startIcon={<DownloadRounded />} variant="outlined">Download video</Button>}
        {isActive(job) && <Button color="error" variant="outlined" startIcon={<StopRounded />} disabled={cancelling}
          onClick={() => void cancel()}>{cancelling ? 'Cancelling…' : 'Cancel processing'}</Button>}
        <Button variant="contained" startIcon={working ? <CircularProgress size={16} color="inherit" /> : <PlayArrowRounded />}
          disabled={!file || working || !enabled || !settings.classes.length} onClick={() => void run()}>
          {uploading ? 'Uploading…' : isActive(job) ? 'Processing…' : 'Run detection'}
        </Button>
      </Stack>
    </div>
    <input ref={input} hidden type="file" accept="video/mp4,video/webm,video/quicktime" aria-label="Video file"
      onChange={(event) => {
        const next = event.target.files?.[0];
        if (next) onUpload(next);
        event.target.value = '';
      }} />
  </>;
}

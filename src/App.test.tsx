import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { Capture } from './types';
import { MAX_VIDEO_BYTES } from './api';

const config = [
  { id: 'person', label: 'Person', color: '#61a8ff', category: 'person' },
  { id: 'pistol', label: 'Pistol', color: '#ff6577', category: 'weapon' },
  { id: 'knife', label: 'Knife', color: '#ffac5c', category: 'weapon' },
  { id: 'rifle', label: 'Rifle', color: '#ff8a80', category: 'weapon' },
  { id: 'smartphone', label: 'Smartphone', color: '#bc9bff', category: 'object' },
];
const result = {
  model_id: 'v0', model_name: 'Weapon detection model v0',
  width: 64, height: 64, inference_ms: 20, weapon_count: 0,
  detections: [], counts: { person: 0, pistol: 0, knife: 0 },
  annotated_image: 'data:image/jpeg;base64,dGVzdA==',
  capture: null, capture_error: null,
};
const savedCapture: Capture = {
  model_id: 'v0', model_name: 'Weapon detection model v0',
  id: 'capture-id', filename: 'capture_test.jpg', created_at: '2026-11-06T14:00:00Z',
  image_url: '/api/captures/capture-id/image', source: 'image', width: 64, height: 64,
  object_confidence: 0.85, person_confidence: 0.92,
  detections: [
    { label: 'pistol', confidence: 0.85, box: [1, 1, 20, 20], source: 'object' },
    { label: 'person', confidence: 0.92, box: [0, 0, 60, 60], source: 'person' },
  ],
  is_warning: true,
  policy: { object_conf: 0.8, person_conf: 0.8, cooldown_seconds: 10,
    classes: ['pistol', 'knife', 'smartphone'], warning_classes: ['pistol', 'knife'] },
};
const modelOptions = [
  { id: 'v1', name: 'Weapon detection model v1', filename: 'last.pt' },
  { id: 'v0', name: 'Weapon detection model v0', filename: 'best.pt' },
];
const completedVideo = {
  id: 'video-id', filename: 'cctv.mp4', status: 'completed', frames_processed: 24,
  total_frames: 24, fps: 12, progress: 100, video_url: '/api/videos/video-id/result',
  error: null, capture: null, capture_count: 0, capture_error: null,
};

async function chooseModel(version: 'v0' | 'v1') {
  fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Weapon detection model' }));
  fireEvent.click(screen.getByRole('option', { name: `Weapon detection model ${version}` }));
  await waitFor(() => expect(screen.queryByRole('listbox')).not.toBeInTheDocument());
}

async function mount() {
  render(<App />);
  await screen.findByText('API connected');
}

function upload() {
  const image = new File(['image'], 'scene.png', { type: 'image/png' });
  fireEvent.change(screen.getByLabelText('Image file'), { target: { files: [image] } });
  return image;
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (url === '/api/health') return new Response('{"device":"cpu"}');
    if (url === '/api/config') return new Response(JSON.stringify(config));
    if (url === '/api/models') return new Response(JSON.stringify(modelOptions));
    if (url === '/api/videos') return new Response(JSON.stringify(init?.method === 'POST' ? completedVideo : []));
    if (url.startsWith('/api/videos/')) return new Response(JSON.stringify(completedVideo));
    if (url.startsWith('/api/captures')) return new Response('{"items":[],"total":0}');
    const version = init?.body instanceof FormData ? init.body.get('model_id') : 'v0';
    return new Response(JSON.stringify({ ...result, model_id: version, model_name: `Weapon detection model ${version}` }));
  }));
  vi.stubGlobal('URL', Object.assign(URL, {
    createObjectURL: vi.fn(() => 'blob:test-image'), revokeObjectURL: vi.fn(),
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('image workspace', () => {
  it('previews uploaded image beside detection output and marks changed settings', async () => {
    await mount();
    expect(screen.getByRole('button', { name: 'Run detection' })).toBeDisabled();
    upload();
    expect(await screen.findByAltText('Original uploaded image')).toHaveAttribute('src', 'blob:test-image');
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByAltText('Annotated detection result')).toBeVisible();
    const body = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/detect')?.[1]?.body;
    if (!(body instanceof FormData)) throw new Error('Expected image detection request');
    expect(body.get('capture_object_conf')).toBe('0.35');
    expect(body.get('capture_person_conf')).toBe('0.55');
    expect(JSON.parse(String(body.get('capture_classes')))).toEqual(['pistol', 'knife', 'rifle']);
    expect(screen.getByText('No selected objects detected at these thresholds.')).toBeVisible();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Knife' }));
    expect(screen.getByText('Previous settings')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(screen.getByRole('button', { name: 'Run detection' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Reset to defaults' }));
    expect(screen.getByRole('checkbox', { name: 'Knife' })).toBeChecked();
  });

  it('accepts dropped and pasted images', async () => {
    await mount();
    const image = new File(['image'], 'drop.webp', { type: 'image/webp' });
    const dropzone = screen.getByRole('button', { name: 'Upload an image' }).parentElement;
    if (!dropzone) throw new Error('Missing dropzone');
    fireEvent.drop(dropzone, { dataTransfer: { files: [image] } });
    expect(await screen.findByText('drop.webp')).toBeInTheDocument();
    const pasted = new File(['image'], 'paste.png', { type: 'image/png' });
    fireEvent.paste(window, { clipboardData: { items: [
      { kind: 'file', type: 'image/png', getAsFile: () => pasted },
    ] } });
    expect(await screen.findByText('paste.png')).toBeInTheDocument();
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });

  it('shows invalid file and server errors', async () => {
    await mount();
    fireEvent.change(screen.getByLabelText('Image file'), {
      target: { files: [new File(['bad'], 'bad.gif', { type: 'image/gif' })] },
    });
    expect(screen.getByText('Choose a JPEG, PNG, or WebP image.')).toBeVisible();
    upload();
    vi.mocked(fetch).mockResolvedValueOnce(new Response('{"detail":"Invalid image content"}', { status: 415 }));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText('Invalid image content')).toBeVisible();
  });

  it('lets the user disable automatic captures and reports saved captures', async () => {
    await mount();
    const autoCapture = screen.getByRole('checkbox', { name: 'Auto-capture screenshots' });
    expect(autoCapture).toBeChecked();
    fireEvent.click(autoCapture);
    upload();
    const capture = savedCapture;
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ ...result, capture })));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText(/Capture saved at/)).toBeVisible();
    const request = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/detect');
    const body = request?.[1]?.body;
    expect(body).toBeInstanceOf(FormData);
    if (!(body instanceof FormData)) throw new Error('Expected multipart body');
    expect(body.get('auto_capture')).toBe('false');
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ items: [capture], total: 1 })));
    fireEvent.click(screen.getAllByRole('button', { name: 'View captures' })[0]);
    expect(await screen.findByText('Saved captures (1)')).toBeVisible();
    expect(screen.getByRole('button', { name: 'View capture capture_test.jpg' })).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'View capture capture_test.jpg' }));
    expect(await screen.findByText('capture_test.jpg')).toBeVisible();
    expect(screen.getByRole('link', { name: 'Download capture' })).toHaveAttribute('href', '/api/captures/capture-id/image?download=true');
  });

  it('shows capture save failures while preserving the detection result', async () => {
    await mount();
    upload();
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
      ...result, capture_error: 'Detection succeeded, but the capture could not be saved.',
    })));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText(/capture could not be saved/)).toBeVisible();
    expect(screen.getByAltText('Annotated detection result')).toBeVisible();
  });

  it('shows an empty gallery and visible gallery request errors', async () => {
    await mount();
    fireEvent.click(screen.getAllByRole('button', { name: 'Check captures' })[0]);
    expect(await screen.findByText(/No captures yet/)).toBeVisible();
    vi.mocked(fetch).mockResolvedValueOnce(new Response('{"detail":"Could not read captures"}', { status: 500 }));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh captures' }));
    expect(await screen.findByText('Could not read captures')).toBeVisible();
  });

  it('uses the dashboard title and class icons', async () => {
    await mount();
    expect(screen.getByRole('heading', { name: 'CCTV-based Firearm Detection' })).toBeVisible();
    expect(screen.getByRole('link', { name: 'Visible Weapon Detection for Hong Kong Surveillance home' })).toHaveAttribute('href', '/');
    expect(screen.getAllByText('Visible Weapon Detection for Hong Kong Surveillance')).toHaveLength(2);
    expect(screen.queryByText('Workspace')).not.toBeInTheDocument();
    expect(screen.queryByText('OBJECT & PERSON RECOGNITION')).not.toBeInTheDocument();
    for (const name of ['Person', 'Pistol', 'Knife', 'Rifle', 'Smartphone']) {
      const label = screen.getByRole('checkbox', { name }).closest('label');
      expect(label?.querySelectorAll('svg').length).toBeGreaterThanOrEqual(2);
    }
  });

  it('shows a warning bubble for warning captures and a normal bubble for other captures', async () => {
    await mount();
    upload();
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ ...result, capture: savedCapture })));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText('Warning capture saved')).toBeVisible();
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
      ...result, capture: { ...savedCapture, id: 'normal-id', is_warning: false },
    })));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run detection' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText('Screenshot captured')).toBeVisible();
    expect(screen.queryByText('Warning capture saved')).not.toBeInTheDocument();
  });

  it('keeps a dismissed notification mounted until its slide-out finishes', async () => {
    await mount();
    upload();
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ ...result, capture: savedCapture })));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    const title = await screen.findByText('Warning capture saved');
    const bubble = title.closest<HTMLElement>('[role="alert"]');
    if (!bubble) throw new Error('Missing notification alert');
    expect(within(bubble).getByRole('button', { name: 'View captures' })).toBeVisible();
    fireEvent.click(within(bubble).getByRole('button', { name: 'Close' }));
    expect(title).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('Warning capture saved')).not.toBeInTheDocument());
  });

  it('enables rifle detection and capture rules by default and labels the auxiliary detector', async () => {
    await mount();
    expect(screen.getByRole('checkbox', { name: 'Rifle' })).toBeChecked();
    upload();
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
      ...result, rifle_model: 'rifle_detector.pt',
      detections: [{ label: 'rifle', confidence: 0.93, box: [1, 1, 20, 20], source: 'object' }],
      counts: { rifle: 1 }, weapon_count: 1,
      capture: { ...savedCapture, rifle_model: 'rifle_detector.pt' },
    })));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText('Rifle detector')).toHaveAttribute('title', 'Rifle detection model · rifle_detector.pt');
    expect(screen.getByText('Processed with Weapon detection model v0 + Rifle detection model')).toBeVisible();
    expect(await screen.findByText('Warning capture saved')).toBeVisible();
    const body = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/detect')?.[1]?.body;
    if (!(body instanceof FormData)) throw new Error('Expected rifle-enabled request');
    for (const field of ['classes', 'capture_classes', 'warning_classes']) {
      expect(JSON.parse(String(body.get(field)))).toContain('rifle');
    }
    fireEvent.click(screen.getByRole('checkbox', { name: 'Rifle' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run detection' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run detection' })).toBeEnabled());
    const nextBody = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/detect').at(-1)?.[1]?.body;
    if (!(nextBody instanceof FormData)) throw new Error('Expected follow-up request');
    expect(JSON.parse(String(nextBody.get('classes')))).not.toContain('rifle');
  });

  it('applies capture settings, configurable warning classes, and the notification toggle to the next request', async () => {
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'Capture settings' }));
    const dialog = await screen.findByRole('dialog', { name: 'Capture settings' });
    fireEvent.change(within(dialog).getByRole('slider', { name: 'Object capture confidence' }), { target: { value: '65' } });
    fireEvent.change(within(dialog).getByRole('slider', { name: 'Person capture confidence' }), { target: { value: '90' } });
    fireEvent.change(within(dialog).getByRole('slider', { name: 'Capture cooldown' }), { target: { value: '20' } });
    const captureClasses = within(dialog).getByRole('group', { name: 'Capture classes' });
    fireEvent.click(within(captureClasses).getByRole('checkbox', { name: 'Pistol' }));
    const warningClasses = within(dialog).getByRole('group', { name: 'Warning classes' });
    fireEvent.click(within(warningClasses).getByRole('checkbox', { name: 'Pistol' }));
    fireEvent.click(within(warningClasses).getByRole('checkbox', { name: 'Smartphone' }));
    fireEvent.click(within(dialog).getByRole('switch', { name: 'Pop-up capture notifications' }));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Done' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Capture settings' })).not.toBeInTheDocument());
    upload();
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ ...result, capture: savedCapture })));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText(/Capture saved at/)).toBeVisible();
    expect(screen.queryByText('Warning capture saved')).not.toBeInTheDocument();
    const request = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/detect');
    const body = request?.[1]?.body;
    if (!(body instanceof FormData)) throw new Error('Expected multipart body');
    expect(body.get('capture_object_conf')).toBe('0.65');
    expect(body.get('capture_person_conf')).toBe('0.9');
    expect(body.get('capture_cooldown')).toBe('20');
    expect(JSON.parse(String(body.get('capture_classes')))).not.toContain('pistol');
    expect(JSON.parse(String(body.get('warning_classes')))).toEqual(['knife', 'rifle', 'smartphone']);
  });

  it('defaults to v1 first, switches image requests to v0, and resets to v1', async () => {
    await mount();
    expect(screen.getByRole('combobox', { name: 'Weapon detection model' })).toHaveTextContent('Weapon detection model v1');
    fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Weapon detection model' }));
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
      'Weapon detection model v1', 'Weapon detection model v0',
    ]);
    fireEvent.keyDown(screen.getByRole('listbox'), { key: 'Escape' });
    upload();
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    await screen.findByAltText('Annotated detection result');
    const firstBody = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/detect')?.[1]?.body;
    if (!(firstBody instanceof FormData)) throw new Error('Expected initial image request');
    expect(firstBody.get('model_id')).toBe('v1');
    await chooseModel('v0');
    expect(screen.getByText('Previous settings')).toBeInTheDocument();
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
      ...result, model_id: 'v0', model_name: 'Weapon detection model v0',
      detections: savedCapture.detections,
    })));
    fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
    expect(await screen.findByText('SoHAS · v0')).toHaveAttribute('title', 'Weapon detection model v0');
    expect(screen.getByText('Processed with Weapon detection model v0')).toBeVisible();
    const requests = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/detect');
    const body = requests[1]?.[1]?.body;
    if (!(body instanceof FormData)) throw new Error('Expected multipart body');
    expect(body.get('model_id')).toBe('v0');
    fireEvent.click(screen.getByRole('button', { name: 'Reset to defaults' }));
    expect(screen.getByRole('combobox', { name: 'Weapon detection model' })).toHaveTextContent('Weapon detection model v1');
  });

  it('surfaces model-list failures rather than pretending v0 is available', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response('{"device":"cpu"}'))
      .mockResolvedValueOnce(new Response(JSON.stringify(config)))
      .mockResolvedValueOnce(new Response('{"detail":"Models unavailable"}', { status: 500 }));
    render(<App />);
    expect(await screen.findByText('Models unavailable')).toBeVisible();
    expect(screen.getByRole('combobox', { name: 'Weapon detection model' })).toHaveAttribute('aria-disabled', 'true');
  });
});

describe('camera lifecycle', () => {
  it('shows camera permission errors', async () => {
    const getUserMedia = vi.fn().mockRejectedValue(new DOMException('Denied', 'NotAllowedError'));
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } });
    await mount();
    fireEvent.click(screen.getByRole('tab', { name: 'Live camera' }));
    fireEvent.click(screen.getByRole('button', { name: 'Open camera' }));
    expect(await screen.findByText(/Camera permission was denied/)).toBeVisible();
    expect(getUserMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
  });

  describe('uploaded CCTV video', () => {
    beforeEach(() => {
      vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    });

    async function uploadVideo() {
      await mount();
      fireEvent.click(screen.getByRole('tab', { name: 'Video' }));
      fireEvent.change(screen.getByLabelText('Video file'), {
        target: { files: [new File(['video'], 'cctv.mp4', { type: 'video/mp4' })] },
      });
      await screen.findByLabelText('Uploaded CCTV video');
      await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/videos', expect.anything()));
    }

    function videoRequests() {
      return vi.mocked(fetch).mock.calls.filter(([url, init]) => url === '/api/videos' && init?.method === 'POST');
    }

    it('runs detection explicitly and creates a separate playable downloadable video', async () => {
      await uploadVideo();
      const video = screen.getByLabelText('Uploaded CCTV video');
      fireEvent.play(video);
      fireEvent.playing(video);
      expect(videoRequests()).toHaveLength(0);
      expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/detect')).toBe(false);
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      expect(await screen.findByLabelText('Annotated detection video')).toHaveAttribute('src', completedVideo.video_url);
      expect(screen.getByRole('link', { name: 'Download video' })).toHaveAttribute('href', `${completedVideo.video_url}?download=true`);
      expect(screen.getByLabelText('Uploaded CCTV video')).toHaveAttribute('src', 'blob:test-image');
      expect(screen.getByText(/Video processing complete.*24 \/ 24 frames/)).toBeVisible();
      const body = videoRequests()[0]?.[1]?.body;
      if (!(body instanceof FormData)) throw new Error('Missing whole-video multipart request');
      expect(body.get('file')).toBeInstanceOf(File);
      expect(body.get('model_id')).toBe('v1');
      expect(body.get('auto_capture')).toBe('true');
      expect(body.get('capture_object_conf')).toBe('0.35');
      expect(body.get('capture_person_conf')).toBe('0.55');
      expect(JSON.parse(String(body.get('capture_classes')))).toEqual(['pistol', 'knife', 'rifle']);
    });

    it('validates videos, accepts dropped footage, and removes its local preview', async () => {
      await mount();
      fireEvent.click(screen.getByRole('tab', { name: 'Video' }));
      fireEvent.change(screen.getByLabelText('Video file'), {
        target: { files: [new File(['bad'], 'bad.avi', { type: 'video/x-msvideo' })] },
      });
      expect(screen.getByText('Choose an MP4, WebM, or MOV video.')).toBeVisible();
      fireEvent.change(screen.getByLabelText('Video file'), {
        target: { files: [new File([], 'empty.mp4', { type: 'video/mp4' })] },
      });
      expect(screen.getByText('This video is empty. Choose another file.')).toBeVisible();
      const dropzone = screen.getByRole('button', { name: 'Upload a video' }).parentElement;
      if (!dropzone) throw new Error('Missing video dropzone');
      fireEvent.drop(dropzone, { dataTransfer: { files: [new File(['video'], 'drop.webm', { type: 'video/webm' })] } });
      expect(await screen.findByLabelText('Uploaded CCTV video')).toHaveAttribute('src', 'blob:test-image');
      expect(videoRequests()).toHaveLength(0);
      const oversized = new File(['video'], 'large.mp4', { type: 'video/mp4' });
      Object.defineProperty(oversized, 'size', { value: MAX_VIDEO_BYTES + 1 });
      fireEvent.change(screen.getByLabelText('Video file'), { target: { files: [oversized] } });
      expect(screen.getByText('Video exceeds the 100 MB upload limit.')).toBeVisible();
      const maximum = new File(['video'], 'maximum.mp4', { type: 'video/mp4' });
      Object.defineProperty(maximum, 'size', { value: MAX_VIDEO_BYTES });
      fireEvent.change(screen.getByLabelText('Video file'), { target: { files: [maximum] } });
      expect(screen.queryByText('Video exceeds the 100 MB upload limit.')).not.toBeInTheDocument();
      expect(screen.getByText('maximum.mp4')).toBeVisible();
      fireEvent.click(screen.getByRole('button', { name: 'Remove video' }));
      expect(screen.getByRole('button', { name: 'Upload a video' })).toBeVisible();
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-image');
    });

    it('uses a fixed settings snapshot and reruns with the newly selected model', async () => {
      await uploadVideo();
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      await screen.findByLabelText('Annotated detection video');
      await chooseModel('v0');
      expect(screen.getByText(/This job keeps its submitted settings/)).toBeVisible();
      expect(videoRequests()).toHaveLength(1);
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      await waitFor(() => expect(videoRequests()).toHaveLength(2));
      const body = videoRequests()[1]?.[1]?.body;
      if (!(body instanceof FormData)) throw new Error('Missing video settings');
      expect(body.get('model_id')).toBe('v0');
    });

    it('retains a running job across tabs, prevents input replacement, and cancels it', async () => {
      await uploadVideo();
      const running = { ...completedVideo, status: 'running', progress: 25, frames_processed: 6, video_url: null };
      vi.mocked(fetch).mockImplementation(async (url, init) => {
        if (String(url).endsWith('/cancel')) return new Response(JSON.stringify({ ...running, status: 'cancelled' }));
        if (url === '/api/videos' && init?.method === 'POST' || url === '/api/videos/video-id') return new Response(JSON.stringify(running));
        return new Response('[]');
      });
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      expect(await screen.findByRole('button', { name: 'Cancel processing' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Remove video' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Replace' })).toBeDisabled();
      fireEvent.click(screen.getByRole('tab', { name: 'Live camera' }));
      expect(screen.getByText(/A video is processing in the background/)).toBeVisible();
      fireEvent.click(screen.getByRole('tab', { name: 'Video' }));
      expect(screen.getByRole('button', { name: 'Cancel processing' })).toBeEnabled();
      fireEvent.click(screen.getByRole('button', { name: 'Cancel processing' }));
      expect(await screen.findByText(/Video processing cancelled/)).toBeVisible();
      expect(screen.getByRole('button', { name: 'Run detection' })).toBeEnabled();
      expect(screen.queryByLabelText('Annotated detection video')).not.toBeInTheDocument();
    });

    it('recovers completed results from server history after a reload', async () => {
      await mount();
      vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify([{ ...completedVideo, capture: savedCapture, capture_count: 1 }])));
      fireEvent.click(screen.getByRole('tab', { name: 'Video' }));
      expect(await screen.findByLabelText('Annotated detection video')).toHaveAttribute('src', completedVideo.video_url);
      expect(screen.getByRole('link', { name: 'Download video' })).toBeVisible();
      expect(screen.queryByText('Warning capture saved')).not.toBeInTheDocument();
    });

    it('surfaces upload, processing, and source preview errors without blocking conversion', async () => {
      await uploadVideo();
      vi.mocked(fetch).mockResolvedValueOnce(new Response('{"detail":"Video upload failed"}', { status: 500 }));
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      expect(await screen.findByText('Video upload failed')).toBeVisible();
      fireEvent.error(screen.getByLabelText('Uploaded CCTV video'));
      expect(screen.getByText(/Your browser cannot preview this source codec/)).toBeVisible();
      expect(screen.getByRole('button', { name: 'Run detection' })).toBeEnabled();
      vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ ...completedVideo, status: 'failed', video_url: null, error: 'Cannot decode video' })));
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      expect(await screen.findByText('Cannot decode video')).toBeVisible();
      expect(screen.queryByRole('link', { name: 'Download video' })).not.toBeInTheDocument();
    });

    it('shows a capture notification during video processing, before the output is ready', async () => {
      await uploadVideo();
      let finish!: (value: Response) => void;
      vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
        ...completedVideo, status: 'queued', progress: 0, frames_processed: 0, video_url: null,
      }))).mockResolvedValueOnce(new Response(JSON.stringify({
        ...completedVideo, status: 'running', progress: 25, frames_processed: 6, video_url: null,
        capture: { ...savedCapture, source: 'video' }, capture_count: 1,
      }))).mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }));
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      expect(await screen.findByRole('button', { name: 'Cancel processing' })).toBeVisible();
      expect(await screen.findByText('Warning capture saved')).toBeVisible();
      expect(screen.getByText(/Processing video.*6 \/ 24 frames/)).toBeVisible();
      expect(screen.queryByLabelText('Annotated detection video')).not.toBeInTheDocument();
      expect(screen.getByText(/1 captures saved/)).toBeVisible();
      await waitFor(() => expect(finish).toBeDefined());
      await act(async () => { finish(new Response(JSON.stringify({
        ...completedVideo, capture: { ...savedCapture, source: 'video' }, capture_count: 1,
      }))); });
      expect(await screen.findByLabelText('Annotated detection video')).toBeVisible();
    });

    it('reports status failures explicitly and resumes polling when retried', async () => {
      await uploadVideo();
      vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
        ...completedVideo, status: 'running', progress: 0, frames_processed: 0, video_url: null,
      }))).mockRejectedValueOnce(new TypeError('Network failure'));
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      expect(await screen.findByText(/Processing may still be running on the backend/)).toBeVisible();
      fireEvent.click(screen.getByRole('button', { name: 'Retry status' }));
      expect(await screen.findByLabelText('Annotated detection video')).toBeVisible();
    });

    it('does not overlap status requests when a poll is slow', async () => {
      await uploadVideo();
      let finish!: (value: Response) => void;
      vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
        ...completedVideo, status: 'running', progress: 0, frames_processed: 0, video_url: null,
      }))).mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }));
      fireEvent.click(screen.getByRole('button', { name: 'Run detection' }));
      await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/videos/video-id', expect.anything()));
      await act(async () => { await new Promise((resolve) => setTimeout(resolve, 850)); });
      expect(vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/videos/video-id')).toHaveLength(1);
      await act(async () => { finish(new Response(JSON.stringify(completedVideo))); });
      expect(await screen.findByLabelText('Annotated detection video')).toBeVisible();
    });
  });

  it('releases a camera if its permission request resolves after cancellation', async () => {
    let resolveCamera: (value: MediaStream) => void = () => { throw new Error('Not initialized'); };
    const pending = new Promise<MediaStream>((resolve) => { resolveCamera = resolve; });
    const stop = vi.fn();
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn(() => pending) } });
    await mount();
    fireEvent.click(screen.getByRole('tab', { name: 'Live camera' }));
    fireEvent.click(screen.getByRole('button', { name: 'Open camera' }));
    fireEvent.click(screen.getByRole('button', { name: 'Stop camera' }));
    await act(async () => { resolveCamera({ getTracks: () => [{ stop }] } as unknown as MediaStream); });
    expect(stop).toHaveBeenCalledOnce();
    expect(screen.getByRole('button', { name: 'Open camera' })).toBeEnabled();
  });

  it('captures frames sequentially and releases tracks when stopped', async () => {
    const track = { stop: vi.fn(), addEventListener: vi.fn() };
    const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn(async () => stream) } });
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
    Object.defineProperty(HTMLMediaElement.prototype, 'readyState', { configurable: true, get: () => 4 });
    Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { configurable: true, get: () => 640 });
    Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { configurable: true, get: () => 480 });
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => {
      callback(new Blob(['frame'], { type: 'image/jpeg' }));
    });
    await mount();
    fireEvent.click(screen.getByRole('tab', { name: 'Live camera' }));
    fireEvent.click(screen.getByRole('button', { name: 'Open camera' }));
    await screen.findByAltText('Annotated detection result');
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/detect', expect.objectContaining({ method: 'POST' })));
    const request = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/detect');
    const body = request?.[1]?.body;
    if (!(body instanceof FormData)) throw new Error('Expected camera multipart body');
    expect(body.get('source')).toBe('camera');
    expect(body.get('auto_capture')).toBe('true');
    expect(body.get('model_id')).toBe('v1');
    await chooseModel('v0');
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, init]) =>
      url === '/api/detect' && init?.body instanceof FormData && init.body.get('model_id') === 'v0')).toBe(true));
    fireEvent.click(screen.getByRole('button', { name: 'Stop camera' }));
    expect(track.stop).toHaveBeenCalledOnce();
    expect(screen.getByText(/Last processed frame/)).toBeInTheDocument();
  });
});

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CapturesGallery from './CapturesGallery';
import { DEFAULT_CAPTURE_SETTINGS } from './types';
import type { Capture } from './types';

const capture: Capture = {
  model_id: 'v0', model_name: 'Weapon detection model v0',
  id: 'capture-id', filename: 'capture_test.jpg', created_at: '2026-11-06T14:00:00Z',
  image_url: '/api/captures/capture-id/image', source: 'camera',
  width: 64, height: 64, object_confidence: 0.85, person_confidence: 0.92,
  detections: [], is_warning: true,
  policy: { object_conf: 0.8, person_conf: 0.8, cooldown_seconds: 10, classes: ['pistol'], warning_classes: ['pistol'] },
};

const deleted = vi.fn();
const openSettings = vi.fn();

async function mount() {
  render(<CapturesGallery open onClose={vi.fn()} settings={DEFAULT_CAPTURE_SETTINGS}
    onDeleted={deleted} onOpenSettings={openSettings} />);
  await screen.findByText('Saved captures (1)');
}

beforeEach(() => {
  deleted.mockClear();
  openSettings.mockClear();
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ items: [capture], total: 1 }))));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('capture deletion', () => {
  it('shows auxiliary rifle model provenance on saved captures', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
      items: [{ ...capture, rifle_model: 'rifle_detector.pt' }], total: 1,
    })));
    await mount();
    expect(screen.getByText('Weapon detection model v0 + Rifle detection model')).toBeVisible();
  });

  it('identifies uploaded-video captures and their selected model', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({
      items: [{ ...capture, source: 'video', model_id: 'v1', model_name: 'Weapon detection model v1' }], total: 1,
    })));
    await mount();
    expect(screen.getByText(/Uploaded video/)).toBeVisible();
    expect(screen.getByText('Weapon detection model v1')).toBeVisible();
  });

  it('requires confirmation and does nothing when deletion is cancelled', async () => {
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'Delete capture capture_test.jpg' }));
    const dialog = screen.getByRole('dialog', { name: 'Delete captured image?' });
    expect(within(dialog).getByText(/permanently deletes/)).toBeVisible();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Delete captured image?' })).not.toBeInTheDocument());
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(deleted).not.toHaveBeenCalled();
  });

  it('deletes through the API and refreshes the gallery only after success', async () => {
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'Delete capture capture_test.jpg' }));
    vi.mocked(fetch).mockResolvedValueOnce(new Response('{"deleted_id":"capture-id"}'))
      .mockResolvedValueOnce(new Response('{"items":[],"total":0}'));
    fireEvent.click(screen.getByRole('button', { name: 'Delete permanently' }));
    expect(await screen.findByText('Saved captures (0)')).toBeVisible();
    expect(await screen.findByText('Capture deleted from the folder and gallery.')).toBeVisible();
    expect(deleted).toHaveBeenCalledWith('capture-id');
    expect(fetch).toHaveBeenCalledWith('/api/captures/capture-id', expect.objectContaining({ method: 'DELETE' }));
  });

  it('preserves the capture and displays deletion failures for retry', async () => {
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'Delete capture capture_test.jpg' }));
    vi.mocked(fetch).mockResolvedValueOnce(new Response('{"detail":"Could not delete this capture."}', { status: 500 }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete permanently' }));
    expect(await screen.findByText('Could not delete this capture.')).toBeVisible();
    expect(deleted).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Delete permanently' })).toBeEnabled();
    expect(screen.getByText('Saved captures (1)')).toBeInTheDocument();
  });

  it('opens capture settings from the gallery', async () => {
    await mount();
    fireEvent.click(screen.getByRole('button', { name: 'Capture settings' }));
    expect(openSettings).toHaveBeenCalledOnce();
  });

  it('returns to the last valid page when the final capture on a page is deleted', async () => {
    const items = Array.from({ length: 12 }, (_, index) => ({ ...capture, id: `id-${index}`, filename: `capture_${index}.jpg` }));
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ items, total: 13 })));
    render(<CapturesGallery open onClose={vi.fn()} settings={DEFAULT_CAPTURE_SETTINGS}
      onDeleted={deleted} onOpenSettings={openSettings} />);
    await screen.findByText('Saved captures (13)');
    vi.mocked(fetch).mockResolvedValueOnce(new Response(JSON.stringify({ items: [capture], total: 13 })));
    fireEvent.click(screen.getByRole('button', { name: 'Go to page 2' }));
    await screen.findByRole('button', { name: 'Delete capture capture_test.jpg' });
    fireEvent.click(screen.getByRole('button', { name: 'Delete capture capture_test.jpg' }));
    vi.mocked(fetch).mockResolvedValueOnce(new Response('{"deleted_id":"capture-id"}'))
      .mockResolvedValueOnce(new Response('{"items":[],"total":12}'))
      .mockResolvedValueOnce(new Response(JSON.stringify({ items, total: 12 })));
    fireEvent.click(screen.getByRole('button', { name: 'Delete permanently' }));
    await waitFor(() => expect(screen.getByText('Saved captures (12)')).toBeVisible());
    expect(await screen.findByRole('button', { name: 'Delete capture capture_0.jpg' })).toBeVisible();
  });
});

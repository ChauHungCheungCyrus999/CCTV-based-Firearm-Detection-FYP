import { describe, expect, it, vi } from 'vitest';
import { detectImage, MAX_IMAGE_BYTES, readResponse, validateImage } from './api';
import { DEFAULT_CAPTURE_SETTINGS } from './types';

describe('image validation', () => {
  it('accepts supported images and rejects unsupported, empty and oversized files', () => {
    expect(validateImage(new File(['image'], 'a.png', { type: 'image/png' }))).toBeNull();
    expect(validateImage(new File(['image'], 'a.gif', { type: 'image/gif' }))).toContain('JPEG');
    expect(validateImage(new File([], 'a.png', { type: 'image/png' }))).toContain('empty');
    const oversized = new File([new Uint8Array(MAX_IMAGE_BYTES + 1)], 'big.png', { type: 'image/png' });
    expect(validateImage(oversized)).toContain('10 MB');
  });
});

describe('API errors', () => {
  it('surfaces server errors instead of showing an empty success state', async () => {
    await expect(readResponse(new Response(JSON.stringify({ detail: 'Image is corrupt' }), { status: 415 })))
      .rejects.toThrow('Image is corrupt');
    await expect(readResponse(new Response('Bad Gateway', { status: 502 }))).rejects.toThrow('502');
  });

  it('sends independent thresholds and selected classes as multipart fields', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}'));
    await detectImage(new Blob(['image']), { modelId: 'v1', classes: ['person', 'knife'], objectConf: 0.45, personConf: 0.7, autoCapture: true,
      capture: { ...DEFAULT_CAPTURE_SETTINGS, objectConf: 0.65, personConf: 0.9, cooldownSeconds: 20,
        classes: ['knife'], warningClasses: ['knife'] } },
      new AbortController().signal);
    const body = fetchMock.mock.calls[0][1]?.body;
    expect(body).toBeInstanceOf(FormData);
    if (!(body instanceof FormData)) throw new Error('Expected multipart body');
    expect(body.get('object_conf')).toBe('0.45');
    expect(body.get('person_conf')).toBe('0.7');
    expect(body.get('classes')).toBe('["person","knife"]');
    expect(body.get('auto_capture')).toBe('true');
    expect(body.get('source')).toBe('image');
    expect(body.get('model_id')).toBe('v1');
    expect(body.get('capture_object_conf')).toBe('0.65');
    expect(body.get('capture_person_conf')).toBe('0.9');
    expect(body.get('capture_cooldown')).toBe('20');
    expect(body.get('capture_classes')).toBe('["knife"]');
    expect(body.get('warning_classes')).toBe('["knife"]');
  });
});

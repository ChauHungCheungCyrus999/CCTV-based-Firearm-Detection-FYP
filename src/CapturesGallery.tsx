import { useEffect, useRef, useState } from 'react';
import {
  Alert, Box, Button, Card, CardActionArea, CardContent, CircularProgress,
  Chip, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Pagination, Stack, Tooltip, Typography,
} from '@mui/material';
import { ArrowBackRounded, CloseRounded, DownloadRounded, RefreshRounded, SettingsOutlined } from '@mui/icons-material';
import { errorMessage, readResponse } from './api';
import ClassIcon from './ClassIcon';
import type { Capture, CapturePage, CaptureSettings } from './types';

const PAGE_SIZE = 12;

function CaptureImage({ capture, fullSize = false }: { capture: Capture; fullSize?: boolean }) {
  const [failed, setFailed] = useState(false);
  return failed
    ? <Alert severity="error">Could not load {capture.filename}. Check the saved file and backend.</Alert>
    : <Box component="img" src={capture.image_url} alt={`Capture from ${new Date(capture.created_at).toLocaleString()}`}
      loading={fullSize ? 'eager' : 'lazy'} onError={() => setFailed(true)}
      sx={{ width: '100%', height: fullSize ? 'auto' : 180, maxHeight: fullSize ? '65vh' : undefined,
        objectFit: 'contain', display: 'block', bgcolor: '#080e16' }} />;
}

export default function CapturesGallery({ open, onClose, latestCaptureId, settings, onOpenSettings, onDeleted }: {
  open: boolean; onClose: () => void; latestCaptureId?: string; settings: CaptureSettings;
  onOpenSettings: () => void; onDeleted: (id: string) => void;
}) {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<CapturePage | null>(null);
  const [selected, setSelected] = useState<Capture | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [pendingDelete, setPendingDelete] = useState<Capture | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleteSuccess, setDeleteSuccess] = useState(false);
  const deleteController = useRef<AbortController | null>(null);
  useEffect(() => () => deleteController.current?.abort(), []);

  useEffect(() => {
    if (open) { setPage(1); setSelected(null); }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void fetch(`/api/captures?limit=${PAGE_SIZE}&offset=${(page - 1) * PAGE_SIZE}`, {
      signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]),
    }).then((response) => readResponse<CapturePage>(response))
      .then((next) => {
        if (!controller.signal.aborted) {
          const lastPage = Math.max(1, Math.ceil(next.total / PAGE_SIZE));
          if (page > lastPage) setPage(lastPage);
          else setData(next);
        }
      }).catch((failure: unknown) => {
        if (!controller.signal.aborted) setError(errorMessage(failure));
      }).finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [open, page, latestCaptureId, revision]);

  function requestDelete(capture: Capture) {
    setPendingDelete(capture);
    setDeleteError(null);
    setDeleteSuccess(false);
  }

  async function deleteCapture() {
    if (!pendingDelete || deleting) return;
    const capture = pendingDelete;
    const controller = new AbortController();
    deleteController.current = controller;
    setDeleting(true);
    setDeleteError(null);
    try {
      const response = await fetch(`/api/captures/${capture.id}`, {
        method: 'DELETE', signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]),
      });
      await readResponse<{ deleted_id: string }>(response);
      if (controller.signal.aborted) return;
      if (selected?.id === capture.id) setSelected(null);
      setPendingDelete(null);
      setDeleteSuccess(true);
      onDeleted(capture.id);
      setRevision((value) => value + 1);
    } catch (failure) {
      if (!controller.signal.aborted) setDeleteError(errorMessage(failure));
    } finally {
      if (!controller.signal.aborted) setDeleting(false);
    }
  }

  return <>
  <Dialog open={open} onClose={() => { if (!deleting) onClose(); }} maxWidth="lg" fullWidth aria-labelledby="captures-title">
    <DialogTitle id="captures-title">
      <Stack direction="row" alignItems="center" justifyContent="space-between" gap={1}>
        <span>Saved captures{data ? ` (${data.total})` : ''}</span>
        <Button size="small" startIcon={<SettingsOutlined />} onClick={onOpenSettings}>Capture settings</Button>
      </Stack>
    </DialogTitle>
    <DialogContent dividers>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Annotated screenshots are stored in the project&apos;s captures/ folder.
        Current trigger: selected capture object ≥{Math.round(settings.objectConf * 100)}% and person ≥{Math.round(settings.personConf * 100)}%
        {' '}in the same frame, with a {settings.cooldownSeconds}-second cooldown.
      </Typography>
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {deleteSuccess && <Alert severity="success" sx={{ mb: 2 }} onClose={() => setDeleteSuccess(false)}>Capture deleted from the folder and gallery.</Alert>}
      {loading && <Stack alignItems="center" sx={{ p: 3 }}><CircularProgress size={26} aria-label="Loading captures" /></Stack>}
      {!loading && !error && selected && <>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
          <Button startIcon={<ArrowBackRounded />} onClick={() => setSelected(null)}>Back to gallery</Button>
          <Tooltip title="Delete captured image"><IconButton color="error" aria-label={`Delete capture ${selected.filename}`}
            onClick={() => requestDelete(selected)}><CloseRounded /></IconButton></Tooltip>
        </Stack>
        <CaptureImage key={selected.id} capture={selected} fullSize />
        <Typography sx={{ mt: 2, overflowWrap: 'anywhere' }} variant="body2">{selected.filename}</Typography>
        <Typography variant="caption" color="text.secondary">
          {new Date(selected.created_at).toLocaleString()} · {selected.width} × {selected.height} · {selected.source}
        </Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>{selected.model_name}{selected.rifle_model ? ' + Rifle detection model' : ''}</Typography>
        <Typography variant="caption" color="text.secondary" display="block">
          Saved with object ≥{Math.round(selected.policy.object_conf * 100)}% and person ≥{Math.round(selected.policy.person_conf * 100)}%.
          {selected.is_warning ? ' Warning class matched.' : ' Normal capture.'}
        </Typography>
        <Stack direction="row" flexWrap="wrap" gap={1} sx={{ my: 2 }}>
          {selected.detections.map((item, index) => <Chip key={index} size="small"
            icon={<ClassIcon label={item.label} size={16} />} label={`${item.label} ${(item.confidence * 100).toFixed(1)}%`} />)}
        </Stack>
        <Button component="a" href={`${selected.image_url}?download=true`} startIcon={<DownloadRounded />}>Download capture</Button>
      </>}
      {!loading && !error && !selected && (data?.items.length
        ? <>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
            {data.items.map((capture) => <Card key={capture.id} variant="outlined" sx={{ position: 'relative' }}>
              <CardActionArea onClick={() => setSelected(capture)} aria-label={`View capture ${capture.filename}`}>
                <CaptureImage capture={capture} />
                <CardContent>
                  {capture.is_warning && <Chip size="small" color="warning" label="Warning" sx={{ mb: 1 }} />}
                  <Typography variant="body2">{new Date(capture.created_at).toLocaleString()}</Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Object {(capture.object_confidence * 100).toFixed(1)}% · Person {(capture.person_confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {capture.source === 'camera' ? 'Live camera' : capture.source === 'video' ? 'Uploaded video' : 'Uploaded image'} · {capture.width} × {capture.height}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {capture.model_name}{capture.rifle_model ? ' + Rifle detection model' : ''}
                  </Typography>
                </CardContent>
              </CardActionArea>
              <Tooltip title="Delete captured image"><IconButton size="small" color="error"
                aria-label={`Delete capture ${capture.filename}`} onClick={() => requestDelete(capture)}
                sx={{ position: 'absolute', top: 8, right: 8, bgcolor: '#0a1018e6', border: '1px solid', borderColor: 'divider',
                  '&:hover': { bgcolor: '#2b1720' } }}><CloseRounded fontSize="small" /></IconButton></Tooltip>
              <Button size="small" sx={{ m: 1 }} component="a" href={`${capture.image_url}?download=true`}
                startIcon={<DownloadRounded />}>Download</Button>
            </Card>)}
          </Box>
          {data.total > PAGE_SIZE && <Stack alignItems="center" sx={{ mt: 3 }}>
            <Pagination count={Math.ceil(data.total / PAGE_SIZE)} page={page}
              onChange={(_, next) => setPage(next)} color="primary" />
          </Stack>}
        </>
        : <Alert severity="info">No captures yet. Enable auto-capture and match the class and confidence rules in Capture settings.</Alert>)}
    </DialogContent>
    <DialogActions>
      <Button startIcon={<RefreshRounded />} onClick={() => setRevision((value) => value + 1)} disabled={loading}>Refresh captures</Button>
      <Button onClick={onClose} disabled={deleting}>Close</Button>
    </DialogActions>
  </Dialog>
  <Dialog open={Boolean(pendingDelete)} onClose={() => { if (!deleting) setPendingDelete(null); }}
    maxWidth="xs" fullWidth aria-labelledby="delete-capture-title">
    <DialogTitle id="delete-capture-title">Delete captured image?</DialogTitle>
    <DialogContent>
      <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>{pendingDelete?.filename}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
        This permanently deletes the saved image from the captures folder and its gallery entry. This cannot be undone.
      </Typography>
      {deleteError && <Alert severity="error" sx={{ mt: 2 }}>{deleteError}</Alert>}
    </DialogContent>
    <DialogActions>
      <Button onClick={() => setPendingDelete(null)} disabled={deleting}>Cancel</Button>
      <Button color="error" variant="contained" onClick={() => void deleteCapture()} disabled={deleting}
        startIcon={deleting ? <CircularProgress size={14} color="inherit" /> : <CloseRounded />}>
        {deleting ? 'Deleting…' : 'Delete permanently'}
      </Button>
    </DialogActions>
  </Dialog>
  </>;
}

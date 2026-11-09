import {
  Alert, Box, Button, Checkbox, Dialog, DialogActions, DialogContent, DialogTitle,
  Divider, FormControlLabel, Slider, Stack, Switch, Typography,
} from '@mui/material';
import ClassIcon from './ClassIcon';
import { DEFAULT_CAPTURE_SETTINGS } from './types';
import type { CaptureSettings, ClassInfo } from './types';

export default function CaptureSettingsDialog({ open, onClose, settings, onChange, classes }: {
  open: boolean; onClose: () => void; settings: CaptureSettings;
  onChange: (next: CaptureSettings) => void; classes: ClassInfo[];
}) {
  function toggle(field: 'classes' | 'warningClasses', id: string) {
    onChange({ ...settings, [field]: settings[field].includes(id)
      ? settings[field].filter((name) => name !== id) : [...settings[field], id] });
  }

  return <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth aria-labelledby="capture-settings-title">
    <DialogTitle id="capture-settings-title">Capture settings</DialogTitle>
    <DialogContent dividers>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Changes apply to the next image, camera, or video frame. Detection filters still apply:
        keep Person and your capture classes selected on the dashboard.
      </Typography>
      <Typography variant="subtitle2">Object capture confidence: {Math.round(settings.objectConf * 100)}%</Typography>
      <Slider aria-label="Object capture confidence" min={1} max={100} value={Math.round(settings.objectConf * 100)}
        valueLabelDisplay="auto" valueLabelFormat={(value) => `${value}%`}
        onChange={(_, value) => onChange({ ...settings, objectConf: Number(value) / 100 })} />
      <Typography variant="subtitle2">Person capture confidence: {Math.round(settings.personConf * 100)}%</Typography>
      <Slider aria-label="Person capture confidence" min={1} max={100} value={Math.round(settings.personConf * 100)}
        valueLabelDisplay="auto" valueLabelFormat={(value) => `${value}%`}
        onChange={(_, value) => onChange({ ...settings, personConf: Number(value) / 100 })} />
      <Typography variant="subtitle2">Capture cooldown: {settings.cooldownSeconds} seconds</Typography>
      <Slider aria-label="Capture cooldown" min={1} max={60} value={settings.cooldownSeconds}
        valueLabelDisplay="auto" valueLabelFormat={(value) => `${value}s`}
        onChange={(_, value) => onChange({ ...settings, cooldownSeconds: Number(value) })} />
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle2">Classes that trigger a capture</Typography>
      <Typography variant="caption" color="text.secondary">A matching object and person must be in the same frame.</Typography>
      <Box role="group" aria-label="Capture classes" sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', my: 1 }}>
        {classes.filter((item) => item.id !== 'person').map((item) => <FormControlLabel key={item.id}
          control={<Checkbox checked={settings.classes.includes(item.id)} onChange={() => toggle('classes', item.id)} />}
          label={<Stack direction="row" alignItems="center" spacing={1}><ClassIcon label={item.id} color={item.color} />
            <Typography variant="body2">{item.label}</Typography></Stack>} />)}
      </Box>
      {!settings.classes.length && <Alert severity="warning" sx={{ mb: 2 }}>Auto-capture is paused until you select a capture class.</Alert>}
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle2">Warning classes</Typography>
      <Typography variant="caption" color="text.secondary">
        Captures containing these classes above the object capture threshold show a warning bubble.
        Other saved captures show a normal notification.
      </Typography>
      <Box role="group" aria-label="Warning classes" sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', my: 1 }}>
        {classes.filter((item) => item.id !== 'person').map((item) => <FormControlLabel key={item.id}
          control={<Checkbox checked={settings.warningClasses.includes(item.id)} onChange={() => toggle('warningClasses', item.id)} />}
          label={<Stack direction="row" alignItems="center" spacing={1}><ClassIcon label={item.id} color={item.color} />
            <Typography variant="body2">{item.label}</Typography></Stack>} />)}
      </Box>
      <FormControlLabel control={<Switch checked={settings.notifications}
        onChange={(_, notifications) => onChange({ ...settings, notifications })} />}
        label="Pop-up capture notifications" />
      <Alert severity="info" sx={{ mt: 1 }}>Warnings are model predictions, not proof of danger or possession. Verify the image.</Alert>
    </DialogContent>
    <DialogActions>
      <Button onClick={() => onChange({
        ...DEFAULT_CAPTURE_SETTINGS, classes: [...DEFAULT_CAPTURE_SETTINGS.classes],
        warningClasses: [...DEFAULT_CAPTURE_SETTINGS.warningClasses],
      })}>Reset capture settings</Button>
      <Button onClick={onClose}>Done</Button>
    </DialogActions>
  </Dialog>;
}

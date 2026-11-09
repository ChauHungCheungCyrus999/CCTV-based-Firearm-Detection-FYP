import { useEffect, useState } from 'react';
import { Alert, AlertTitle, Button, Slide, Snackbar, Typography, useMediaQuery } from '@mui/material';
import type { SlideProps } from '@mui/material';
import { PhotoCameraOutlined } from '@mui/icons-material';
import type { Capture } from './types';

function SlideNotification(props: SlideProps) {
  return <Slide {...props} direction="left" />;
}

export default function CaptureNotification({ capture, persist, onClose, onView }: {
  capture: Capture | null; persist: boolean; onClose: () => void; onView: () => void;
}) {
  const [display, setDisplay] = useState(capture);
  const [open, setOpen] = useState(Boolean(capture));
  const reducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');
  useEffect(() => {
    if (!capture) setOpen(false);
    else if (!display) { setDisplay(capture); setOpen(true); }
    else if (capture.id !== display.id) setOpen(false);
  }, [capture, display]);

  function close() { setOpen(false); onClose(); }

  return <Snackbar open={open} autoHideDuration={persist ? null : 8000}
    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
    onClose={(_, reason) => { if (reason !== 'clickaway') close(); }}
    slots={{ transition: SlideNotification }}
    slotProps={{ transition: {
      timeout: reducedMotion ? 0 : { enter: 320, exit: 240 },
      onExited: () => {
        setDisplay(capture);
        if (capture) setOpen(true);
      },
    } }}
    sx={{ maxWidth: { xs: 'calc(100% - 16px)', sm: 380 } }}>
    <Alert className="capture-notification" variant="outlined"
      severity={display?.is_warning ? 'warning' : 'success'} onClose={close}
      sx={{
        width: 380, maxWidth: '100%', bgcolor: '#111c2a', color: 'text.primary',
        boxShadow: 6, alignItems: 'flex-start',
        '& .MuiAlert-message': { minWidth: 0, flex: 1 },
        '& .MuiAlert-action': { pt: 0.5 },
      }}>
      <AlertTitle sx={{ fontSize: 14, fontWeight: 650 }}>
        {display?.is_warning ? 'Warning capture saved' : 'Screenshot captured'}
      </AlertTitle>
      <Typography variant="body2" sx={{ fontSize: 12, lineHeight: 1.6 }}>
        {display?.is_warning
          ? 'A configured warning class was detected. Verify the saved image.'
          : 'An object and person matched your capture rules.'}
      </Typography>
      <Button color="inherit" size="small" startIcon={<PhotoCameraOutlined />}
        sx={{ mt: 1, whiteSpace: 'nowrap', fontSize: 11 }}
        onClick={() => { onView(); close(); }}>
        View captures
      </Button>
    </Alert>
  </Snackbar>;
}

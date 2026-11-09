import { SvgIcon } from '@mui/material';
import {
  CategoryOutlined, CreditCardOutlined, PaymentsOutlined, PersonOutlineRounded,
  SmartphoneRounded, WorkOutlineRounded,
} from '@mui/icons-material';

export default function ClassIcon({ label, color, size = 18 }: { label: string; color?: string; size?: number }) {
  const sx = { fontSize: size, color, flexShrink: 0 };
  if (label === 'pistol') return <SvgIcon sx={sx}>
    <path d="M3 5h15l3 2v4h-8l-2 3h-1l-1 7H5l1-9H3V5zm9 7h3v3h-4l1-3z" />
  </SvgIcon>;
  if (label === 'knife') return <SvgIcon sx={sx}>
    <path d="M20 2c1 5-1 10-6 13l-3-3 9-10zM10 13l3 3-6 6-3-3 6-6z" />
  </SvgIcon>;
  if (label === 'rifle') return <SvgIcon sx={sx}>
    <path d="M2 10h4V8h8V6h2v2h6v3h-7l-2 2H9l-1 7H5l1-8H2v-2zm8 4h3v3h-4l1-3z" />
  </SvgIcon>;
  const icons: Record<string, typeof PersonOutlineRounded> = {
    person: PersonOutlineRounded, smartphone: SmartphoneRounded, purse: WorkOutlineRounded,
    banknote: PaymentsOutlined, card: CreditCardOutlined,
  };
  const Icon = icons[label] ?? CategoryOutlined;
  return <Icon sx={sx} />;
}
